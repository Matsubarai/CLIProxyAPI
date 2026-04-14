package responses

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/tidwall/gjson"
)

func parseSSEEvent(t *testing.T, chunk []byte) (string, gjson.Result) {
	t.Helper()

	lines := strings.Split(string(chunk), "\n")
	if len(lines) < 2 {
		t.Fatalf("unexpected SSE chunk: %q", chunk)
	}

	event := strings.TrimSpace(strings.TrimPrefix(lines[0], "event:"))
	dataLine := strings.TrimSpace(strings.TrimPrefix(lines[1], "data:"))
	if !gjson.Valid(dataLine) {
		t.Fatalf("invalid SSE data JSON: %q", dataLine)
	}
	return event, gjson.Parse(dataLine)
}

func TestConvertGeminiResponseToOpenAIResponses_UnwrapAndAggregateText(t *testing.T) {
	// Vertex-style Gemini stream wraps the actual response payload under "response".
	// This test ensures we unwrap and that output_text.done contains the full text.
	in := []string{
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":""}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2,"cachedContentTokenCount":0},"modelVersion":"test-model","responseId":"req_vrtx_1"},"traceId":"t1"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"让"}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2,"cachedContentTokenCount":0},"modelVersion":"test-model","responseId":"req_vrtx_1"},"traceId":"t1"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"我先"}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2,"cachedContentTokenCount":0},"modelVersion":"test-model","responseId":"req_vrtx_1"},"traceId":"t1"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"了解"}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2,"cachedContentTokenCount":0},"modelVersion":"test-model","responseId":"req_vrtx_1"},"traceId":"t1"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"mcp__serena__list_dir","args":{"recursive":false,"relative_path":"internal"},"id":"toolu_1"}}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2,"cachedContentTokenCount":0},"modelVersion":"test-model","responseId":"req_vrtx_1"},"traceId":"t1"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15,"cachedContentTokenCount":2},"modelVersion":"test-model","responseId":"req_vrtx_1"},"traceId":"t1"}`,
	}

	originalReq := []byte(`{"instructions":"test instructions","model":"gpt-5","max_output_tokens":123}`)

	var param any
	var out [][]byte
	for _, line := range in {
		out = append(out, ConvertGeminiResponseToOpenAIResponses(context.Background(), "test-model", originalReq, nil, []byte(line), &param)...)
	}

	var (
		gotTextDone     bool
		gotMessageDone  bool
		gotResponseDone bool
		gotFuncDone     bool

		textDone     string
		messageText  string
		responseID   string
		instructions string
		cachedTokens int64

		funcName string
		funcArgs string

		posTextDone    = -1
		posPartDone    = -1
		posMessageDone = -1
		posFuncAdded   = -1
	)

	for i, chunk := range out {
		ev, data := parseSSEEvent(t, chunk)
		switch ev {
		case "response.output_text.done":
			gotTextDone = true
			if posTextDone == -1 {
				posTextDone = i
			}
			textDone = data.Get("text").String()
		case "response.content_part.done":
			if posPartDone == -1 {
				posPartDone = i
			}
		case "response.output_item.done":
			switch data.Get("item.type").String() {
			case "message":
				gotMessageDone = true
				if posMessageDone == -1 {
					posMessageDone = i
				}
				messageText = data.Get("item.content.0.text").String()
			case "function_call":
				gotFuncDone = true
				funcName = data.Get("item.name").String()
				funcArgs = data.Get("item.arguments").String()
			}
		case "response.output_item.added":
			if data.Get("item.type").String() == "function_call" && posFuncAdded == -1 {
				posFuncAdded = i
			}
		case "response.completed":
			gotResponseDone = true
			responseID = data.Get("response.id").String()
			instructions = data.Get("response.instructions").String()
			cachedTokens = data.Get("response.usage.input_tokens_details.cached_tokens").Int()
		}
	}

	if !gotTextDone {
		t.Fatalf("missing response.output_text.done event")
	}
	if posTextDone == -1 || posPartDone == -1 || posMessageDone == -1 || posFuncAdded == -1 {
		t.Fatalf("missing ordering events: textDone=%d partDone=%d messageDone=%d funcAdded=%d", posTextDone, posPartDone, posMessageDone, posFuncAdded)
	}
	if !(posTextDone < posPartDone && posPartDone < posMessageDone && posMessageDone < posFuncAdded) {
		t.Fatalf("unexpected message/function ordering: textDone=%d partDone=%d messageDone=%d funcAdded=%d", posTextDone, posPartDone, posMessageDone, posFuncAdded)
	}
	if !gotMessageDone {
		t.Fatalf("missing message response.output_item.done event")
	}
	if !gotFuncDone {
		t.Fatalf("missing function_call response.output_item.done event")
	}
	if !gotResponseDone {
		t.Fatalf("missing response.completed event")
	}

	if textDone != "让我先了解" {
		t.Fatalf("unexpected output_text.done text: got %q", textDone)
	}
	if messageText != "让我先了解" {
		t.Fatalf("unexpected message done text: got %q", messageText)
	}

	if responseID != "resp_req_vrtx_1" {
		t.Fatalf("unexpected response id: got %q", responseID)
	}
	if instructions != "test instructions" {
		t.Fatalf("unexpected instructions echo: got %q", instructions)
	}
	if cachedTokens != 2 {
		t.Fatalf("unexpected cached token count: got %d", cachedTokens)
	}

	if funcName != "mcp__serena__list_dir" {
		t.Fatalf("unexpected function name: got %q", funcName)
	}
	if !gjson.Valid(funcArgs) {
		t.Fatalf("invalid function arguments JSON: %q", funcArgs)
	}
	if gjson.Get(funcArgs, "recursive").Bool() != false {
		t.Fatalf("unexpected recursive arg: %v", gjson.Get(funcArgs, "recursive").Value())
	}
	if gjson.Get(funcArgs, "relative_path").String() != "internal" {
		t.Fatalf("unexpected relative_path arg: %q", gjson.Get(funcArgs, "relative_path").String())
	}
}

func TestConvertGeminiResponseToOpenAIResponses_ReasoningEncryptedContent(t *testing.T) {
	sig := "RXE0RENrZ0lDeEFDR0FJcVFOZDdjUzlleGFuRktRdFcvSzNyZ2MvWDNCcDQ4RmxSbGxOWUlOVU5kR1l1UHMrMGdkMVp0Vkg3ekdKU0g4YVljc2JjN3lNK0FrdGpTNUdqamI4T3Z0VVNETzdQd3pmcFhUOGl3U3hXUEJvTVFRQ09mWTFyMEtTWGZxUUlJakFqdmFGWk83RW1XRlBKckJVOVpkYzdDKw=="
	in := []string{
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"thought":true,"thoughtSignature":"` + sig + `","text":""}]}}],"modelVersion":"test-model","responseId":"req_vrtx_sig"},"traceId":"t1"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":"a"}]}}],"modelVersion":"test-model","responseId":"req_vrtx_sig"},"traceId":"t1"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]}}],"modelVersion":"test-model","responseId":"req_vrtx_sig"},"traceId":"t1"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}],"modelVersion":"test-model","responseId":"req_vrtx_sig"},"traceId":"t1"}`,
	}

	var param any
	var out [][]byte
	for _, line := range in {
		out = append(out, ConvertGeminiResponseToOpenAIResponses(context.Background(), "test-model", nil, nil, []byte(line), &param)...)
	}

	var (
		addedEnc string
		doneEnc  string
	)
	for _, chunk := range out {
		ev, data := parseSSEEvent(t, chunk)
		switch ev {
		case "response.output_item.added":
			if data.Get("item.type").String() == "reasoning" {
				addedEnc = data.Get("item.encrypted_content").String()
			}
		case "response.output_item.done":
			if data.Get("item.type").String() == "reasoning" {
				doneEnc = data.Get("item.encrypted_content").String()
			}
		}
	}

	if addedEnc != sig {
		t.Fatalf("unexpected encrypted_content in response.output_item.added: got %q", addedEnc)
	}
	if doneEnc != sig {
		t.Fatalf("unexpected encrypted_content in response.output_item.done: got %q", doneEnc)
	}
}

func TestConvertGeminiResponseToOpenAIResponses_FunctionCallEventOrder(t *testing.T) {
	in := []string{
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"tool0"}}]}}],"modelVersion":"test-model","responseId":"req_vrtx_1"},"traceId":"t1"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"tool1"}}]}}],"modelVersion":"test-model","responseId":"req_vrtx_1"},"traceId":"t1"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"tool2","args":{"a":1}}}]}}],"modelVersion":"test-model","responseId":"req_vrtx_1"},"traceId":"t1"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15,"cachedContentTokenCount":0},"modelVersion":"test-model","responseId":"req_vrtx_1"},"traceId":"t1"}`,
	}

	var param any
	var out [][]byte
	for _, line := range in {
		out = append(out, ConvertGeminiResponseToOpenAIResponses(context.Background(), "test-model", nil, nil, []byte(line), &param)...)
	}

	posAdded := []int{-1, -1, -1}
	posArgsDelta := []int{-1, -1, -1}
	posArgsDone := []int{-1, -1, -1}
	posItemDone := []int{-1, -1, -1}
	posCompleted := -1
	deltaByIndex := map[int]string{}

	for i, chunk := range out {
		ev, data := parseSSEEvent(t, chunk)
		switch ev {
		case "response.output_item.added":
			if data.Get("item.type").String() != "function_call" {
				continue
			}
			idx := int(data.Get("output_index").Int())
			if idx >= 0 && idx < len(posAdded) {
				posAdded[idx] = i
			}
		case "response.function_call_arguments.delta":
			idx := int(data.Get("output_index").Int())
			if idx >= 0 && idx < len(posArgsDelta) {
				posArgsDelta[idx] = i
				deltaByIndex[idx] = data.Get("delta").String()
			}
		case "response.function_call_arguments.done":
			idx := int(data.Get("output_index").Int())
			if idx >= 0 && idx < len(posArgsDone) {
				posArgsDone[idx] = i
			}
		case "response.output_item.done":
			if data.Get("item.type").String() != "function_call" {
				continue
			}
			idx := int(data.Get("output_index").Int())
			if idx >= 0 && idx < len(posItemDone) {
				posItemDone[idx] = i
			}
		case "response.completed":
			posCompleted = i

			output := data.Get("response.output")
			if !output.Exists() || !output.IsArray() {
				t.Fatalf("missing response.output in response.completed")
			}
			if len(output.Array()) != 3 {
				t.Fatalf("unexpected response.output length: got %d", len(output.Array()))
			}
			if data.Get("response.output.0.name").String() != "tool0" || data.Get("response.output.0.arguments").String() != "{}" {
				t.Fatalf("unexpected output[0]: %s", data.Get("response.output.0").Raw)
			}
			if data.Get("response.output.1.name").String() != "tool1" || data.Get("response.output.1.arguments").String() != "{}" {
				t.Fatalf("unexpected output[1]: %s", data.Get("response.output.1").Raw)
			}
			if data.Get("response.output.2.name").String() != "tool2" {
				t.Fatalf("unexpected output[2] name: %s", data.Get("response.output.2").Raw)
			}
			if !gjson.Valid(data.Get("response.output.2.arguments").String()) {
				t.Fatalf("unexpected output[2] arguments: %q", data.Get("response.output.2.arguments").String())
			}
		}
	}

	if posCompleted == -1 {
		t.Fatalf("missing response.completed event")
	}
	for idx := 0; idx < 3; idx++ {
		if posAdded[idx] == -1 || posArgsDelta[idx] == -1 || posArgsDone[idx] == -1 || posItemDone[idx] == -1 {
			t.Fatalf("missing function call events for output_index %d: added=%d argsDelta=%d argsDone=%d itemDone=%d", idx, posAdded[idx], posArgsDelta[idx], posArgsDone[idx], posItemDone[idx])
		}
		if !(posAdded[idx] < posArgsDelta[idx] && posArgsDelta[idx] < posArgsDone[idx] && posArgsDone[idx] < posItemDone[idx]) {
			t.Fatalf("unexpected ordering for output_index %d: added=%d argsDelta=%d argsDone=%d itemDone=%d", idx, posAdded[idx], posArgsDelta[idx], posArgsDone[idx], posItemDone[idx])
		}
		if idx > 0 && !(posItemDone[idx-1] < posAdded[idx]) {
			t.Fatalf("function call events overlap between %d and %d: prevDone=%d nextAdded=%d", idx-1, idx, posItemDone[idx-1], posAdded[idx])
		}
	}

	if deltaByIndex[0] != "{}" {
		t.Fatalf("unexpected delta for output_index 0: got %q", deltaByIndex[0])
	}
	if deltaByIndex[1] != "{}" {
		t.Fatalf("unexpected delta for output_index 1: got %q", deltaByIndex[1])
	}
	if deltaByIndex[2] == "" || !gjson.Valid(deltaByIndex[2]) || gjson.Get(deltaByIndex[2], "a").Int() != 1 {
		t.Fatalf("unexpected delta for output_index 2: got %q", deltaByIndex[2])
	}
	if !(posItemDone[2] < posCompleted) {
		t.Fatalf("response.completed should be after last output_item.done: last=%d completed=%d", posItemDone[2], posCompleted)
	}
}

func TestConvertGeminiResponseToOpenAIResponses_ResponseOutputOrdering(t *testing.T) {
	in := []string{
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"tool0","args":{"x":"y"}}}]}}],"modelVersion":"test-model","responseId":"req_vrtx_2"},"traceId":"t2"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]}}],"modelVersion":"test-model","responseId":"req_vrtx_2"},"traceId":"t2"}`,
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2,"cachedContentTokenCount":0},"modelVersion":"test-model","responseId":"req_vrtx_2"},"traceId":"t2"}`,
	}

	var param any
	var out [][]byte
	for _, line := range in {
		out = append(out, ConvertGeminiResponseToOpenAIResponses(context.Background(), "test-model", nil, nil, []byte(line), &param)...)
	}

	posFuncDone := -1
	posMsgAdded := -1
	posCompleted := -1

	for i, chunk := range out {
		ev, data := parseSSEEvent(t, chunk)
		switch ev {
		case "response.output_item.done":
			if data.Get("item.type").String() == "function_call" && data.Get("output_index").Int() == 0 {
				posFuncDone = i
			}
		case "response.output_item.added":
			if data.Get("item.type").String() == "message" && data.Get("output_index").Int() == 1 {
				posMsgAdded = i
			}
		case "response.completed":
			posCompleted = i
			if data.Get("response.output.0.type").String() != "function_call" {
				t.Fatalf("expected response.output[0] to be function_call: %s", data.Get("response.output.0").Raw)
			}
			if data.Get("response.output.1.type").String() != "message" {
				t.Fatalf("expected response.output[1] to be message: %s", data.Get("response.output.1").Raw)
			}
			if data.Get("response.output.1.content.0.text").String() != "hi" {
				t.Fatalf("unexpected message text in response.output[1]: %s", data.Get("response.output.1").Raw)
			}
		}
	}

	if posFuncDone == -1 || posMsgAdded == -1 || posCompleted == -1 {
		t.Fatalf("missing required events: funcDone=%d msgAdded=%d completed=%d", posFuncDone, posMsgAdded, posCompleted)
	}
	if !(posFuncDone < posMsgAdded) {
		t.Fatalf("expected function_call to complete before message is added: funcDone=%d msgAdded=%d", posFuncDone, posMsgAdded)
	}
	if !(posMsgAdded < posCompleted) {
		t.Fatalf("expected response.completed after message added: msgAdded=%d completed=%d", posMsgAdded, posCompleted)
	}
}

func TestConvertGeminiResponseToOpenAIResponses_CodeInterpreterStreamReopensMessage(t *testing.T) {
	ctx := context.Background()
	originalReq := []byte(`{
		"model":"gpt-5",
		"tools":[{"type":"code_interpreter","container":"container_req"}],
		"include":["code_interpreter_call.outputs"]
	}`)
	requestJSON := ConvertOpenAIResponsesRequestToGemini("gemini-2.5-flash", originalReq, true)
	chunks := [][]byte{
		[]byte(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"Let me calculate that."}]}}],"modelVersion":"gemini-2.5-flash","responseId":"resp_code_stream"},"traceId":"t1"}`),
		[]byte(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"executableCode":{"language":"PYTHON","code":"print(6 * 7)"}}]}}],"modelVersion":"gemini-2.5-flash","responseId":"resp_code_stream"},"traceId":"t1"}`),
		[]byte(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"42\n"}},{"text":"The answer is 42."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":7,"totalTokenCount":16},"modelVersion":"gemini-2.5-flash","responseId":"resp_code_stream"},"traceId":"t1"}`),
	}

	var param any
	var outputs [][]byte
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertGeminiResponseToOpenAIResponses(ctx, "gemini-2.5-flash", originalReq, requestJSON, chunk, &param)...)
	}

	var (
		messageDoneCount      int
		firstMessageText      string
		secondMessageText     string
		codeAddedPos          = -1
		codeInProgressPos     = -1
		codeDeltaPos          = -1
		codeCodeDonePos       = -1
		codeInterpretingPos   = -1
		codeCompletedPos      = -1
		codeItemDonePos       = -1
		firstMessageDone      = -1
		secondMessageAdd      = -1
		completedPos          = -1
		codeInterpreterItemID string
	)
	for i, chunk := range outputs {
		ev, data := parseSSEEvent(t, chunk)
		switch ev {
		case "response.output_item.added":
			switch data.Get("item.type").String() {
			case "code_interpreter_call":
				if codeAddedPos == -1 {
					codeAddedPos = i
				}
				codeInterpreterItemID = data.Get("item.id").String()
				if got := data.Get("item.status").String(); got != "in_progress" {
					t.Fatalf("code_interpreter_call status = %q, want %q: %s", got, "in_progress", data.Raw)
				}
				if data.Get("item.code").Exists() {
					t.Fatalf("did not expect code_interpreter_call code in initial added event: %s", data.Raw)
				}
				if got := data.Get("item.container_id").String(); got != "container_req" {
					t.Fatalf("code_interpreter_call container_id = %q, want %q: %s", got, "container_req", data.Raw)
				}
			case "message":
				if firstMessageDone >= 0 && secondMessageAdd == -1 {
					secondMessageAdd = i
				}
			}
		case "response.output_item.done":
			switch data.Get("item.type").String() {
			case "message":
				messageDoneCount++
				if messageDoneCount == 1 {
					firstMessageDone = i
					firstMessageText = data.Get("item.content.0.text").String()
				} else if messageDoneCount == 2 {
					secondMessageText = data.Get("item.content.0.text").String()
				}
			case "code_interpreter_call":
				codeItemDonePos = i
				if got := data.Get("item.status").String(); got != "completed" {
					t.Fatalf("code_interpreter_call done status = %q, want %q: %s", got, "completed", data.Raw)
				}
				if got := data.Get("item.code").String(); got != "print(6 * 7)" {
					t.Fatalf("code_interpreter_call done code = %q, want %q: %s", got, "print(6 * 7)", data.Raw)
				}
				if got := data.Get("item.outputs.0.type").String(); got != "logs" {
					t.Fatalf("code_interpreter_call output type = %q, want %q: %s", got, "logs", data.Raw)
				}
				if got := data.Get("item.outputs.0.logs").String(); got != "42\n" {
					t.Fatalf("code_interpreter_call logs = %q, want %q: %s", got, "42\\n", data.Raw)
				}
			}
		case "response.code_interpreter_call.in_progress":
			codeInProgressPos = i
			if got := data.Get("item_id").String(); got != codeInterpreterItemID {
				t.Fatalf("code_interpreter_call.in_progress item_id = %q, want %q: %s", got, codeInterpreterItemID, data.Raw)
			}
		case "response.code_interpreter_call_code.delta":
			codeDeltaPos = i
			if got := data.Get("item_id").String(); got != codeInterpreterItemID {
				t.Fatalf("code_interpreter_call_code.delta item_id = %q, want %q: %s", got, codeInterpreterItemID, data.Raw)
			}
			if got := data.Get("delta").String(); got != "print(6 * 7)" {
				t.Fatalf("code_interpreter_call_code.delta = %q, want %q: %s", got, "print(6 * 7)", data.Raw)
			}
		case "response.code_interpreter_call_code.done":
			codeCodeDonePos = i
			if got := data.Get("item_id").String(); got != codeInterpreterItemID {
				t.Fatalf("code_interpreter_call_code.done item_id = %q, want %q: %s", got, codeInterpreterItemID, data.Raw)
			}
			if got := data.Get("code").String(); got != "print(6 * 7)" {
				t.Fatalf("code_interpreter_call_code.done code = %q, want %q: %s", got, "print(6 * 7)", data.Raw)
			}
		case "response.code_interpreter_call.interpreting":
			codeInterpretingPos = i
			if got := data.Get("item_id").String(); got != codeInterpreterItemID {
				t.Fatalf("code_interpreter_call.interpreting item_id = %q, want %q: %s", got, codeInterpreterItemID, data.Raw)
			}
		case "response.code_interpreter_call.completed":
			codeCompletedPos = i
			if got := data.Get("item_id").String(); got != codeInterpreterItemID {
				t.Fatalf("code_interpreter_call.completed item_id = %q, want %q: %s", got, codeInterpreterItemID, data.Raw)
			}
		case "response.completed":
			completedPos = i
			if got := data.Get("response.output.0.type").String(); got != "message" {
				t.Fatalf("response.output[0].type = %q, want %q: %s", got, "message", data.Raw)
			}
			if got := data.Get("response.output.1.type").String(); got != "code_interpreter_call" {
				t.Fatalf("response.output[1].type = %q, want %q: %s", got, "code_interpreter_call", data.Raw)
			}
			if got := data.Get("response.output.1.container_id").String(); got != "container_req" {
				t.Fatalf("response.output[1].container_id = %q, want %q: %s", got, "container_req", data.Raw)
			}
			if got := data.Get("response.output.1.outputs.0.logs").String(); got != "42\n" {
				t.Fatalf("response.output[1].outputs[0].logs = %q, want %q: %s", got, "42\\n", data.Raw)
			}
			if got := data.Get("response.output.2.type").String(); got != "message" {
				t.Fatalf("response.output[2].type = %q, want %q: %s", got, "message", data.Raw)
			}
		}
	}

	if messageDoneCount != 2 {
		t.Fatalf("messageDoneCount = %d, want 2", messageDoneCount)
	}
	if firstMessageText != "Let me calculate that." {
		t.Fatalf("firstMessageText = %q, want %q", firstMessageText, "Let me calculate that.")
	}
	if secondMessageText != "The answer is 42." {
		t.Fatalf("secondMessageText = %q, want %q", secondMessageText, "The answer is 42.")
	}
	if !(firstMessageDone >= 0 && codeAddedPos >= 0 && codeInProgressPos >= 0 && codeDeltaPos >= 0 && codeCodeDonePos >= 0 && codeInterpretingPos >= 0 && codeCompletedPos >= 0 && codeItemDonePos >= 0 && secondMessageAdd >= 0 && completedPos >= 0) {
		t.Fatalf("missing ordering markers: firstMessageDone=%d codeAdded=%d codeInProgress=%d codeDelta=%d codeCodeDone=%d codeInterpreting=%d codeCompleted=%d codeItemDone=%d secondMessageAdd=%d completed=%d", firstMessageDone, codeAddedPos, codeInProgressPos, codeDeltaPos, codeCodeDonePos, codeInterpretingPos, codeCompletedPos, codeItemDonePos, secondMessageAdd, completedPos)
	}
	if !(firstMessageDone < codeAddedPos &&
		codeAddedPos < codeInProgressPos &&
		codeInProgressPos < codeDeltaPos &&
		codeDeltaPos < codeCodeDonePos &&
		codeCodeDonePos < codeInterpretingPos &&
		codeInterpretingPos < codeCompletedPos &&
		codeCompletedPos < codeItemDonePos &&
		codeItemDonePos < secondMessageAdd &&
		secondMessageAdd < completedPos) {
		t.Fatalf("unexpected code interpreter ordering: firstMessageDone=%d codeAdded=%d codeInProgress=%d codeDelta=%d codeCodeDone=%d codeInterpreting=%d codeCompleted=%d codeItemDone=%d secondMessageAdd=%d completed=%d", firstMessageDone, codeAddedPos, codeInProgressPos, codeDeltaPos, codeCodeDonePos, codeInterpretingPos, codeCompletedPos, codeItemDonePos, secondMessageAdd, completedPos)
	}
}

func TestConvertGeminiResponseToOpenAIResponsesNonStream_CodeInterpreterOutputsFollowInclude(t *testing.T) {
	rawJSON := []byte(`{
		"responseId":"resp_code_nonstream",
		"modelVersion":"gemini-2.5-flash",
		"candidates":[{
			"content":{"role":"model","parts":[
				{"executableCode":{"language":"PYTHON","code":"print(42)"}},
				{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"42\n"}},
				{"text":"The answer is 42."}
			]},
			"finishReason":"STOP"
		}]
	}`)

	withoutInclude := []byte(`{
		"model":"gpt-5",
		"tools":[{"type":"code_interpreter","container":"container_req"}]
	}`)
	withInclude := []byte(`{
		"model":"gpt-5",
		"tools":[{"type":"code_interpreter","container":"container_req"}],
		"include":["code_interpreter_call.outputs"]
	}`)

	outputWithoutInclude := ConvertGeminiResponseToOpenAIResponsesNonStream(context.Background(), "gemini-2.5-flash", withoutInclude, nil, rawJSON, nil)
	if got := gjson.GetBytes(outputWithoutInclude, "output.0.type").String(); got != "code_interpreter_call" {
		t.Fatalf("output[0].type = %q, want %q: %s", got, "code_interpreter_call", string(outputWithoutInclude))
	}
	if got := gjson.GetBytes(outputWithoutInclude, "output.0.container_id").String(); got != "container_req" {
		t.Fatalf("output[0].container_id = %q, want %q: %s", got, "container_req", string(outputWithoutInclude))
	}
	if gjson.GetBytes(outputWithoutInclude, "output.0.outputs").Exists() {
		t.Fatalf("did not expect code interpreter outputs without include: %s", string(outputWithoutInclude))
	}
	if got := gjson.GetBytes(outputWithoutInclude, "output.1.content.0.text").String(); got != "The answer is 42." {
		t.Fatalf("output[1].content[0].text = %q, want %q: %s", got, "The answer is 42.", string(outputWithoutInclude))
	}

	outputWithInclude := ConvertGeminiResponseToOpenAIResponsesNonStream(context.Background(), "gemini-2.5-flash", withInclude, nil, rawJSON, nil)
	if got := gjson.GetBytes(outputWithInclude, "output.0.outputs.0.type").String(); got != "logs" {
		t.Fatalf("output[0].outputs[0].type = %q, want %q: %s", got, "logs", string(outputWithInclude))
	}
	if got := gjson.GetBytes(outputWithInclude, "output.0.outputs.0.logs").String(); got != "42\n" {
		t.Fatalf("output[0].outputs[0].logs = %q, want %q: %s", got, "42\\n", string(outputWithInclude))
	}
}

func TestConvertGeminiResponseToOpenAIResponsesNonStream_GroundingMetadataAddsSearchCallAndAnnotations(t *testing.T) {
	originalReq := []byte(`{
		"model":"gpt-5",
		"tools":[{"type":"web_search"}]
	}`)
	rawJSON := []byte(`{
		"responseId":"resp_grounded",
		"modelVersion":"gemini-2.5-flash",
		"candidates":[{
			"content":{"role":"model","parts":[{"text":"Alpha beta"}]},
			"finishReason":"STOP",
			"groundingMetadata":{
				"webSearchQueries":["Alpha beta source"],
				"groundingChunks":[
					{"web":{"uri":"https://example.com/a","title":"Example A"}}
				],
				"groundingSupports":[
					{"segment":{"startIndex":0,"endIndex":10,"text":"Alpha beta"},"groundingChunkIndices":[0]}
				]
			}
		}]
	}`)

	output := ConvertGeminiResponseToOpenAIResponsesNonStream(context.Background(), "gemini-2.5-flash", originalReq, nil, rawJSON, nil)

	if got := gjson.GetBytes(output, "output.0.type").String(); got != "web_search_call" {
		t.Fatalf("output[0].type = %q, want %q: %s", got, "web_search_call", string(output))
	}
	if got := gjson.GetBytes(output, "output.0.action.type").String(); got != "search" {
		t.Fatalf("output[0].action.type = %q, want %q: %s", got, "search", string(output))
	}
	if got := gjson.GetBytes(output, "output.0.action.queries.0").String(); got != "Alpha beta source" {
		t.Fatalf("output[0].action.queries[0] = %q, want %q: %s", got, "Alpha beta source", string(output))
	}
	if got := gjson.GetBytes(output, "output.1.type").String(); got != "message" {
		t.Fatalf("output[1].type = %q, want %q: %s", got, "message", string(output))
	}
	if got := gjson.GetBytes(output, "output.1.content.0.annotations.0.type").String(); got != "url_citation" {
		t.Fatalf("annotations[0].type = %q, want %q: %s", got, "url_citation", string(output))
	}
	if got := gjson.GetBytes(output, "output.1.content.0.annotations.0.start_index").Int(); got != 0 {
		t.Fatalf("annotations[0].start_index = %d, want 0: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "output.1.content.0.annotations.0.end_index").Int(); got != 10 {
		t.Fatalf("annotations[0].end_index = %d, want 10: %s", got, string(output))
	}
}

func TestConvertGeminiResponseToOpenAIResponsesNonStream_GroundingSourcesFollowInclude(t *testing.T) {
	originalReq := []byte(`{
		"model":"gpt-5",
		"tools":[{"type":"web_search_preview"}],
		"include":["web_search_call.action.sources"]
	}`)
	rawJSON := []byte(`{
		"responseId":"resp_grounded_sources",
		"modelVersion":"gemini-2.5-flash",
		"candidates":[{
			"content":{"role":"model","parts":[{"text":"Alpha beta"}]},
			"finishReason":"STOP",
			"groundingMetadata":{
				"webSearchQueries":["Alpha beta source"],
				"groundingChunks":[
					{"web":{"uri":"https://example.com/a","title":"Example A"}}
				],
				"groundingSupports":[
					{"segment":{"startIndex":0,"endIndex":10,"text":"Alpha beta"},"groundingChunkIndices":[0]}
				]
			}
		}]
	}`)

	output := ConvertGeminiResponseToOpenAIResponsesNonStream(context.Background(), "gemini-2.5-flash", originalReq, nil, rawJSON, nil)

	if got := gjson.GetBytes(output, "output.0.action.sources.0.type").String(); got != "url" {
		t.Fatalf("action.sources[0].type = %q, want %q: %s", got, "url", string(output))
	}
	if got := gjson.GetBytes(output, "output.0.action.sources.0.url").String(); got != "https://example.com/a" {
		t.Fatalf("action.sources[0].url = %q, want %q: %s", got, "https://example.com/a", string(output))
	}
}

func TestConvertGeminiResponseToOpenAIResponsesNonStream_GroundingByteOffsets(t *testing.T) {
	originalReq := []byte(`{"model":"gpt-5","tools":[{"type":"web_search"}]}`)
	text := "中文 grounded 引用测试"
	segment := "grounded"
	byteStart := bytes.Index([]byte(text), []byte(segment))
	if byteStart < 0 {
		t.Fatalf("segment %q not found in %q", segment, text)
	}
	byteEnd := byteStart + len([]byte(segment))
	wantStart := utf8.RuneCountInString(text[:byteStart])
	wantEnd := wantStart + utf8.RuneCountInString(segment)

	rawJSON := []byte(fmt.Sprintf(`{
		"responseId":"resp_byte_offsets",
		"modelVersion":"gemini-2.5-flash",
		"candidates":[{
			"content":{"role":"model","parts":[{"text":%q}]},
			"finishReason":"STOP",
			"groundingMetadata":{
				"groundingChunks":[
					{"web":{"uri":"https://example.com/source","title":"Example Source"}}
				],
				"groundingSupports":[
					{"segment":{"startIndex":%d,"endIndex":%d,"text":%q},"groundingChunkIndices":[0]}
				]
			}
		}]
	}`, text, byteStart, byteEnd, segment))

	output := ConvertGeminiResponseToOpenAIResponsesNonStream(context.Background(), "gemini-2.5-flash", originalReq, nil, rawJSON, nil)

	if got := gjson.GetBytes(output, "output.1.content.0.annotations.0.start_index").Int(); got != int64(wantStart) {
		t.Fatalf("start_index = %d, want %d: %s", got, wantStart, string(output))
	}
	if got := gjson.GetBytes(output, "output.1.content.0.annotations.0.end_index").Int(); got != int64(wantEnd) {
		t.Fatalf("end_index = %d, want %d: %s", got, wantEnd, string(output))
	}
}

func TestConvertGeminiResponseToOpenAIResponses_StreamGroundingCompletesWithAnnotationsAndSearchCall(t *testing.T) {
	ctx := context.Background()
	originalReq := []byte(`{
		"model":"gpt-5",
		"tools":[{"type":"web_search_preview"}]
	}`)
	requestJSON := ConvertOpenAIResponsesRequestToGemini("gemini-2.5-flash", originalReq, true)
	var param any

	firstChunk := []byte(`{
		"responseId":"resp_stream_grounded",
		"modelVersion":"gemini-2.5-flash",
		"candidates":[{"content":{"role":"model","parts":[{"text":"Alpha beta"}]}}]
	}`)
	secondChunk := []byte(`{
		"responseId":"resp_stream_grounded",
		"modelVersion":"gemini-2.5-flash",
		"candidates":[{
			"finishReason":"STOP",
			"groundingMetadata":{
				"webSearchQueries":["Alpha beta source"],
				"groundingChunks":[
					{"web":{"uri":"https://example.com/a","title":"Example A"}}
				],
				"groundingSupports":[
					{"segment":{"startIndex":0,"endIndex":10,"text":"Alpha beta"},"groundingChunkIndices":[0]}
				]
			}
		}]
	}`)

	firstOutput := ConvertGeminiResponseToOpenAIResponses(ctx, "gemini-2.5-flash", originalReq, requestJSON, firstChunk, &param)
	if len(firstOutput) == 0 {
		t.Fatal("expected first chunk output")
	}
	var (
		sawCreated       bool
		sawInProgress    bool
		sawSearchAdded   bool
		sawSearchRunning bool
		sawTextDelta     bool
	)
	for _, chunk := range firstOutput {
		ev, data := parseSSEEvent(t, chunk)
		switch ev {
		case "response.created":
			sawCreated = true
		case "response.in_progress":
			sawInProgress = true
		case "response.output_item.added":
			if data.Get("item.type").String() == "web_search_call" && data.Get("item.status").String() == "in_progress" {
				sawSearchAdded = true
			}
		case "response.web_search_call.in_progress", "response.web_search_call.searching":
			sawSearchRunning = true
		case "response.output_text.delta":
			sawTextDelta = true
			if got := data.Get("delta").String(); got != "Alpha beta" {
				t.Fatalf("delta = %q, want %q: %s", got, "Alpha beta", data.Raw)
			}
		}
	}
	if !sawCreated || !sawInProgress || !sawSearchAdded || !sawSearchRunning || !sawTextDelta {
		t.Fatalf("missing first chunk events: created=%v inProgress=%v searchAdded=%v searchRunning=%v textDelta=%v", sawCreated, sawInProgress, sawSearchAdded, sawSearchRunning, sawTextDelta)
	}

	secondOutput := ConvertGeminiResponseToOpenAIResponses(ctx, "gemini-2.5-flash", originalReq, requestJSON, secondChunk, &param)
	if len(secondOutput) == 0 {
		t.Fatal("expected grounded completion events on finish chunk")
	}
	var (
		sawAnnotationAdded bool
		sawContentPartDone bool
		sawMessageDone     bool
		sawSearchCallDone  bool
		sawSearchCompleted bool
		sawCompleted       bool
		posSearchCompleted = -1
		posSearchDone      = -1
		posAnnotationAdded = -1
		posTextDone        = -1
		posPartDone        = -1
		posMessageDone     = -1
	)
	for i, chunk := range secondOutput {
		ev, data := parseSSEEvent(t, chunk)
		switch ev {
		case "response.web_search_call.completed":
			sawSearchCompleted = true
			posSearchCompleted = i
		case "response.output_text.annotation.added":
			sawAnnotationAdded = true
			if posAnnotationAdded == -1 {
				posAnnotationAdded = i
			}
			if got := data.Get("annotation.type").String(); got != "url_citation" {
				t.Fatalf("annotation.type = %q, want %q: %s", got, "url_citation", data.Raw)
			}
		case "response.output_text.done":
			posTextDone = i
		case "response.content_part.done":
			sawContentPartDone = true
			posPartDone = i
			if got := data.Get("part.annotations.0.type").String(); got != "url_citation" {
				t.Fatalf("part.annotations[0].type = %q, want %q: %s", got, "url_citation", data.Raw)
			}
		case "response.output_item.done":
			switch data.Get("item.type").String() {
			case "message":
				sawMessageDone = true
				posMessageDone = i
				if got := data.Get("item.content.0.annotations.0.url").String(); got != "https://example.com/a" {
					t.Fatalf("message annotation url = %q, want %q: %s", got, "https://example.com/a", data.Raw)
				}
			case "web_search_call":
				sawSearchCallDone = true
				posSearchDone = i
				if got := data.Get("item.action.queries.0").String(); got != "Alpha beta source" {
					t.Fatalf("search call action.queries[0] = %q, want %q: %s", got, "Alpha beta source", data.Raw)
				}
			}
		case "response.completed":
			sawCompleted = true
			if got := data.Get("response.output.0.type").String(); got != "web_search_call" {
				t.Fatalf("response.output[0].type = %q, want %q: %s", got, "web_search_call", data.Raw)
			}
			if got := data.Get("response.output.0.action.queries.0").String(); got != "Alpha beta source" {
				t.Fatalf("response.output[0].action.queries[0] = %q, want %q: %s", got, "Alpha beta source", data.Raw)
			}
			if got := data.Get("response.output.1.type").String(); got != "message" {
				t.Fatalf("response.output[1].type = %q, want %q: %s", got, "message", data.Raw)
			}
		}
	}

	if !sawSearchCompleted || !sawAnnotationAdded || !sawContentPartDone || !sawMessageDone || !sawSearchCallDone || !sawCompleted {
		t.Fatalf("missing grounded completion events: searchCompleted=%v annotationAdded=%v contentPartDone=%v messageDone=%v searchCallDone=%v completed=%v", sawSearchCompleted, sawAnnotationAdded, sawContentPartDone, sawMessageDone, sawSearchCallDone, sawCompleted)
	}
	if !(posSearchCompleted >= 0 && posSearchDone >= 0 && posAnnotationAdded >= 0 && posTextDone >= 0 && posPartDone >= 0 && posMessageDone >= 0) {
		t.Fatalf("missing event positions: searchCompleted=%d searchDone=%d annotationAdded=%d textDone=%d partDone=%d messageDone=%d", posSearchCompleted, posSearchDone, posAnnotationAdded, posTextDone, posPartDone, posMessageDone)
	}
	if !(posSearchCompleted < posSearchDone && posSearchDone < posAnnotationAdded && posAnnotationAdded < posTextDone && posTextDone < posPartDone && posPartDone < posMessageDone) {
		t.Fatalf("unexpected grounded completion ordering: searchCompleted=%d searchDone=%d annotationAdded=%d textDone=%d partDone=%d messageDone=%d", posSearchCompleted, posSearchDone, posAnnotationAdded, posTextDone, posPartDone, posMessageDone)
	}
}
