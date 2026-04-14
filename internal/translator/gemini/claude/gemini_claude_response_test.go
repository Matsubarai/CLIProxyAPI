package claude

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertGeminiResponseToClaudeNonStream_GroundedGoogleSearch(t *testing.T) {
	originalRequest := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"Who won Euro 2024?"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`)

	rawJSON := []byte(`{
		"responseId":"resp_1",
		"modelVersion":"gemini-3-flash-preview",
		"candidates":[{
			"content":{"role":"model","parts":[{"text":"Spain won Euro 2024, defeating England 2-1 in the final."}]},
			"finishReason":"STOP",
			"groundingMetadata":{
				"webSearchQueries":["UEFA Euro 2024 winner","who won euro 2024"],
				"groundingChunks":[
					{"web":{"uri":"https://example.com/a","title":"Example A"}},
					{"web":{"uri":"https://example.com/b","title":"Example B"}}
				],
				"groundingSupports":[
					{
						"segment":{"startIndex":0,"endIndex":57,"text":"Spain won Euro 2024, defeating England 2-1 in the final."},
						"groundingChunkIndices":[0,1]
					}
				]
			}
		}],
		"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":8}
	}`)

	output := ConvertGeminiResponseToClaudeNonStream(context.Background(), "gemini-3-flash-preview", originalRequest, nil, rawJSON, nil)
	if got := gjson.GetBytes(output, "content.0.type").String(); got != "text" {
		t.Fatalf("Expected cited text block first, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.0.citations.0.type").String(); got != "web_search_result_location" {
		t.Fatalf("Expected web_search_result_location citation on first text block, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.0.citations.0.encrypted_index").String(); got != "" {
		t.Fatalf("Expected synthetic encrypted_index to default to empty string, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.1.type").String(); got != "server_tool_use" {
		t.Fatalf("Expected server_tool_use block second, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.1.name").String(); got != "web_search" {
		t.Fatalf("Expected server_tool_use name web_search, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.1.input.query").String(); got != "UEFA Euro 2024 winner" {
		t.Fatalf("Expected first grounding query to be used, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.2.type").String(); got != "web_search_tool_result" {
		t.Fatalf("Expected web_search_tool_result block third, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.2.content.#").Int(); got != 2 {
		t.Fatalf("Expected 2 grounded results, got %d: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.2.content.0.encrypted_content").String(); got != "" {
		t.Fatalf("Expected encrypted_content to default to empty string, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "usage.server_tool_use.web_search_requests").Int(); got != 2 {
		t.Fatalf("Expected web_search_requests=2, got %d: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "stop_reason").String(); got != "end_turn" {
		t.Fatalf("Expected stop_reason end_turn, got %q: %s", got, string(output))
	}
}

func TestConvertGeminiResponseToClaudeNonStream_GroundedGoogleSearchChineseByteOffsets(t *testing.T) {
	originalRequest := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"请联网搜索后回答"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`)

	text := "第一段说明。第二段引用来源甲。第三段引用来源乙。第四段引用来源丙。第五段引用来源丁。总结。"
	segments := []string{
		"第二段引用来源甲。",
		"第三段引用来源乙。",
		"第四段引用来源丙。",
		"第五段引用来源丁。",
	}

	offsets := make([][2]int, 0, len(segments))
	for _, segment := range segments {
		start := strings.Index(text, segment)
		if start < 0 {
			t.Fatalf("Expected segment %q to exist in source text", segment)
		}
		offsets = append(offsets, [2]int{start, start + len(segment)})
	}
	if offsets[len(offsets)-1][0] <= len([]rune(text)) {
		t.Fatalf("Expected later byte offset to exceed rune length, got start=%d runeLen=%d", offsets[len(offsets)-1][0], len([]rune(text)))
	}

	rawJSON := []byte(fmt.Sprintf(`{
		"responseId":"resp_byte_offsets",
		"modelVersion":"gemini-3-flash-preview",
		"candidates":[{
			"content":{"role":"model","parts":[{"text":%q}]},
			"finishReason":"STOP",
			"groundingMetadata":{
				"webSearchQueries":["中文 grounded 引用测试"],
				"groundingChunks":[
					{"web":{"uri":"https://example.com/a","title":"Example A"}},
					{"web":{"uri":"https://example.com/b","title":"Example B"}},
					{"web":{"uri":"https://example.com/c","title":"Example C"}},
					{"web":{"uri":"https://example.com/d","title":"Example D"}}
				],
				"groundingSupports":[
					{"segment":{"startIndex":%d,"endIndex":%d,"text":%q},"groundingChunkIndices":[0]},
					{"segment":{"startIndex":%d,"endIndex":%d,"text":%q},"groundingChunkIndices":[1]},
					{"segment":{"startIndex":%d,"endIndex":%d,"text":%q},"groundingChunkIndices":[2]},
					{"segment":{"startIndex":%d,"endIndex":%d,"text":%q},"groundingChunkIndices":[3]}
				]
			}
		}],
		"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":8}
	}`,
		text,
		offsets[0][0], offsets[0][1], segments[0],
		offsets[1][0], offsets[1][1], segments[1],
		offsets[2][0], offsets[2][1], segments[2],
		offsets[3][0], offsets[3][1], segments[3],
	))

	output := ConvertGeminiResponseToClaudeNonStream(context.Background(), "gemini-3-flash-preview", originalRequest, nil, rawJSON, nil)
	content := gjson.GetBytes(output, "content").Array()
	citedTexts := make(map[string]int)
	for _, block := range content {
		if block.Get("type").String() != "text" {
			continue
		}
		for _, citation := range block.Get("citations").Array() {
			citedTexts[citation.Get("cited_text").String()]++
		}
	}

	for _, segment := range segments {
		if citedTexts[segment] != 1 {
			t.Fatalf("Expected cited segment %q to be preserved exactly once, got %d in output: %s", segment, citedTexts[segment], string(output))
		}
	}

	var groundedResultCount int64
	for _, block := range content {
		if block.Get("type").String() == "web_search_tool_result" {
			groundedResultCount = block.Get("content.#").Int()
			break
		}
	}
	if groundedResultCount != 4 {
		t.Fatalf("Expected 4 grounded search results, got %d: %s", groundedResultCount, string(output))
	}
}

func TestConvertGeminiResponseToClaude_StreamGroundedGoogleSearch(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"Who won Euro 2024?"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`)
	requestJSON := ConvertClaudeRequestToGemini("gemini-3-flash-preview", originalRequest, true)
	var param any

	chunks := [][]byte{
		[]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"Spain won Euro 2024, defeating England 2-1 in the final."}]}}],"modelVersion":"gemini-3-flash-preview","responseId":"resp_1"}`),
		[]byte(`{"candidates":[{"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["UEFA Euro 2024 winner"],"groundingChunks":[{"web":{"uri":"https://example.com/a","title":"Example A"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":57,"text":"Spain won Euro 2024, defeating England 2-1 in the final."},"groundingChunkIndices":[0]}]}}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":8},"modelVersion":"gemini-3-flash-preview","responseId":"resp_1"}`),
		[]byte(`[DONE]`),
	}

	var outputs [][]byte
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertGeminiResponseToClaude(ctx, "gemini-3-flash-preview", originalRequest, requestJSON, chunk, &param)...)
	}

	if len(outputs) != 1 {
		t.Fatalf("Expected grounded search stream to buffer until DONE, got %d outputs", len(outputs))
	}

	events := extractClaudeSSEEventData(outputs[0])
	if len(events) == 0 {
		t.Fatalf("Expected SSE events in final grounded search output: %s", string(outputs[0]))
	}

	messageStartIndex := -1
	textStartIndex := -1
	serverToolUseStartIndex := -1
	searchResultStartIndex := -1
	textDeltaIndex := -1
	citationDeltaIndex := -1
	textBlockStopIndex := -1
	messageDeltaIndex := -1
	messageStopIndex := -1
	foundEmptyEncryptedIndex := false
	foundEmptyEncryptedContent := false

	for i, event := range events {
		switch event.Get("type").String() {
		case "message_start":
			messageStartIndex = i
		case "content_block_start":
			switch event.Get("content_block.type").String() {
			case "text":
				if textStartIndex == -1 {
					textStartIndex = i
				}
			case "server_tool_use":
				if event.Get("content_block.name").String() == "web_search" {
					serverToolUseStartIndex = i
				}
			case "web_search_tool_result":
				searchResultStartIndex = i
				if event.Get("content_block.content.0.encrypted_content").Exists() && event.Get("content_block.content.0.encrypted_content").String() == "" {
					foundEmptyEncryptedContent = true
				}
			}
		case "content_block_delta":
			if event.Get("delta.type").String() == "citations_delta" && event.Get("delta.citation.type").String() == "web_search_result_location" {
				if citationDeltaIndex == -1 {
					citationDeltaIndex = i
				}
				if event.Get("delta.citation.encrypted_index").Exists() && event.Get("delta.citation.encrypted_index").String() == "" {
					foundEmptyEncryptedIndex = true
				}
			}
			if event.Get("delta.type").String() == "text_delta" && event.Get("index").Int() == 0 && strings.Contains(event.Get("delta.text").String(), "Spain won Euro 2024") {
				textDeltaIndex = i
			}
		case "content_block_stop":
			if event.Get("index").Int() == 0 {
				textBlockStopIndex = i
			}
		case "message_delta":
			messageDeltaIndex = i
		case "message_stop":
			messageStopIndex = i
		}
	}

	if messageStartIndex != 0 {
		t.Fatalf("Expected message_start first, got %s", string(outputs[0]))
	}
	if textStartIndex < 0 {
		t.Fatal("Expected grounded search output to open a text block before search blocks")
	}
	if serverToolUseStartIndex < 0 {
		t.Fatal("Expected server_tool_use block in buffered grounded stream output")
	}
	if searchResultStartIndex < 0 {
		t.Fatal("Expected web_search_tool_result block in buffered grounded stream output")
	}
	if !foundEmptyEncryptedContent {
		t.Fatal("Expected web_search_tool_result content to include string encrypted_content")
	}
	if citationDeltaIndex < 0 {
		t.Fatal("Expected citations_delta in buffered grounded stream output")
	}
	if !foundEmptyEncryptedIndex {
		t.Fatal("Expected citations_delta to include string encrypted_index")
	}
	if textDeltaIndex < 0 {
		t.Fatal("Expected text_delta in grounded stream output")
	}
	if textBlockStopIndex < 0 {
		t.Fatal("Expected grounded stream output to stop the text block")
	}
	if messageDeltaIndex < 0 {
		t.Fatal("Expected grounded stream output to include message_delta")
	}
	if messageStopIndex < 0 {
		t.Fatal("Expected grounded stream output to include message_stop")
	}
	if !(textStartIndex < serverToolUseStartIndex && serverToolUseStartIndex < searchResultStartIndex) {
		t.Fatalf("Expected text block start before web_search tool/result blocks, got %s", string(outputs[0]))
	}
	if !(searchResultStartIndex < textDeltaIndex && searchResultStartIndex < citationDeltaIndex) {
		t.Fatalf("Expected text/citation deltas after web_search_tool_result, got %s", string(outputs[0]))
	}
	if !(textDeltaIndex < textBlockStopIndex && textBlockStopIndex < messageDeltaIndex && messageDeltaIndex < messageStopIndex) {
		t.Fatalf("Expected text block to close before message_delta/message_stop, got %s", string(outputs[0]))
	}
}

func TestConvertGeminiResponseToClaude_StreamGroundedGoogleSearchWithoutUsagePreservesSearchUsage(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"Who won Euro 2024?"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`)
	requestJSON := ConvertClaudeRequestToGemini("gemini-3-flash-preview", originalRequest, true)
	var param any

	chunks := [][]byte{
		[]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"Spain won Euro 2024."}]}}],"modelVersion":"gemini-3-flash-preview","responseId":"resp_2"}`),
		[]byte(`{"candidates":[{"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["UEFA Euro 2024 winner"],"groundingChunks":[{"web":{"uri":"https://example.com/a","title":"Example A"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":20,"text":"Spain won Euro 2024."},"groundingChunkIndices":[0]}]}}],"modelVersion":"gemini-3-flash-preview","responseId":"resp_2"}`),
		[]byte(`[DONE]`),
	}

	var outputs [][]byte
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertGeminiResponseToClaude(ctx, "gemini-3-flash-preview", originalRequest, requestJSON, chunk, &param)...)
	}

	if len(outputs) != 1 {
		t.Fatalf("Expected grounded search stream to buffer until DONE, got %d outputs", len(outputs))
	}

	events := extractClaudeSSEEventData(outputs[0])
	if len(events) == 0 {
		t.Fatalf("Expected SSE events in final grounded search output: %s", string(outputs[0]))
	}

	var messageDelta gjson.Result
	for _, event := range events {
		if event.Get("type").String() == "message_delta" {
			messageDelta = event
			break
		}
	}
	if !messageDelta.Exists() {
		t.Fatalf("Expected message_delta event in final buffered output: %s", string(outputs[1]))
	}
	if got := messageDelta.Get("usage.input_tokens").Int(); got != 0 {
		t.Fatalf("Expected usage.input_tokens=0 when Gemini omits usage metadata, got %d", got)
	}
	if got := messageDelta.Get("usage.output_tokens").Int(); got != 0 {
		t.Fatalf("Expected usage.output_tokens=0 when Gemini omits usage metadata, got %d", got)
	}
	if got := messageDelta.Get("usage.server_tool_use.web_search_requests").Int(); got != 1 {
		t.Fatalf("Expected usage.server_tool_use.web_search_requests=1 when Gemini omits usage metadata, got %d", got)
	}
}

func TestConvertGeminiResponseToClaude_StreamGroundedGoogleSearchAfterThinkingKeepsBlockIndexes(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"latest headline"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`)
	requestJSON := ConvertClaudeRequestToGemini("gemini-3-flash-preview", originalRequest, true)
	var param any

	chunks := [][]byte{
		[]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"Thinking first.","thought":true}]}}],"modelVersion":"gemini-3-flash-preview","responseId":"resp_idx"}`),
		[]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"Answer body."}]}}],"modelVersion":"gemini-3-flash-preview","responseId":"resp_idx"}`),
		[]byte(`{"candidates":[{"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["latest headline"],"groundingChunks":[{"web":{"uri":"https://example.com/a","title":"Example A"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":12,"text":"Answer body."},"groundingChunkIndices":[0]}]}}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":8},"modelVersion":"gemini-3-flash-preview","responseId":"resp_idx"}`),
		[]byte(`[DONE]`),
	}

	var outputs [][]byte
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertGeminiResponseToClaude(ctx, "gemini-3-flash-preview", originalRequest, requestJSON, chunk, &param)...)
	}

	if len(outputs) != 1 {
		t.Fatalf("Expected grounded search stream with thinking to buffer until DONE, got %d outputs", len(outputs))
	}

	events := extractClaudeSSEEventData(outputs[0])
	if len(events) == 0 {
		t.Fatalf("Expected grounded finale events, got %s", string(outputs[0]))
	}

	thinkingStartIndex := -1
	textStartIndex := -1
	serverToolUseIndex := -1
	searchResultIndex := -1
	textDeltaIndex := -1
	citationIndex := -1
	textStopIndex := -1
	for i, event := range events {
		if event.Get("type").String() == "content_block_start" && event.Get("content_block.type").String() == "thinking" && event.Get("index").Int() == 0 {
			thinkingStartIndex = i
		}
		if event.Get("type").String() == "content_block_start" && event.Get("content_block.type").String() == "text" && event.Get("index").Int() == 1 {
			textStartIndex = i
		}
		if event.Get("type").String() == "content_block_delta" &&
			event.Get("delta.type").String() == "text_delta" &&
			event.Get("index").Int() == 1 {
			textDeltaIndex = i
		}
		if event.Get("type").String() == "content_block_delta" &&
			event.Get("delta.type").String() == "citations_delta" &&
			event.Get("index").Int() == 1 {
			citationIndex = i
		}
		if event.Get("type").String() == "content_block_stop" && event.Get("index").Int() == 1 {
			textStopIndex = i
		}
		if event.Get("type").String() == "content_block_start" &&
			event.Get("content_block.type").String() == "server_tool_use" &&
			event.Get("content_block.name").String() == "web_search" &&
			event.Get("index").Int() == 2 {
			serverToolUseIndex = i
		}
		if event.Get("type").String() == "content_block_start" &&
			event.Get("content_block.type").String() == "web_search_tool_result" &&
			event.Get("index").Int() == 3 {
			searchResultIndex = i
		}
	}

	if thinkingStartIndex < 0 {
		t.Fatalf("Expected thinking block to remain index 0, got %s", string(outputs[0]))
	}
	if textStartIndex < 0 {
		t.Fatalf("Expected answer text block to remain index 1, got %s", string(outputs[0]))
	}
	if serverToolUseIndex < 0 {
		t.Fatalf("Expected server_tool_use to start at index 2 after thinking+text blocks, got %s", string(outputs[0]))
	}
	if searchResultIndex < 0 {
		t.Fatalf("Expected web_search_tool_result to start at index 3 after server_tool_use, got %s", string(outputs[0]))
	}
	if textDeltaIndex < 0 || citationIndex < 0 || textStopIndex < 0 {
		t.Fatalf("Expected text_delta/citations_delta/content_block_stop on text block index 1, got %s", string(outputs[0]))
	}
	if !(thinkingStartIndex < textStartIndex && textStartIndex < serverToolUseIndex && serverToolUseIndex < searchResultIndex) {
		t.Fatalf("Expected thinking -> text start -> web_search tool/result order, got %s", string(outputs[0]))
	}
	if !(searchResultIndex < textDeltaIndex && searchResultIndex < citationIndex && citationIndex < textStopIndex) {
		t.Fatalf("Expected answer text/citations after web_search result, got %s", string(outputs[0]))
	}
}

func TestConvertGeminiResponseToClaude_StreamSearchCapableWithoutGroundingBuffersUntilDone(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"Who are you?"}],
		"tools":[{"type":"web_search_20250305","name":"web_search"}]
	}`)
	requestJSON := ConvertClaudeRequestToGemini("gemini-3-flash-preview", originalRequest, true)
	var param any

	chunks := [][]byte{
		[]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"I am Gemini 3 Flash Preview."}]}}],"modelVersion":"gemini-3-flash-preview","responseId":"resp_no_grounding"}`),
		[]byte(`{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":8},"modelVersion":"gemini-3-flash-preview","responseId":"resp_no_grounding"}`),
		[]byte(`[DONE]`),
	}

	var outputs [][]byte
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertGeminiResponseToClaude(ctx, "gemini-3-flash-preview", originalRequest, requestJSON, chunk, &param)...)
	}

	if len(outputs) != 1 {
		t.Fatalf("Expected search-capable request to buffer until DONE, got %d outputs", len(outputs))
	}

	finalEvents := extractClaudeSSEEventData(outputs[0])
	if len(finalEvents) == 0 {
		t.Fatalf("Expected buffered final output, got %s", string(outputs[0]))
	}
	foundTextDelta := false
	foundMessageDelta := false
	foundMessageStop := false
	for _, event := range finalEvents {
		if event.Get("type").String() == "content_block_delta" && event.Get("delta.type").String() == "text_delta" &&
			strings.Contains(event.Get("delta.text").String(), "Gemini 3 Flash Preview") {
			foundTextDelta = true
		}
		if event.Get("type").String() == "message_delta" {
			foundMessageDelta = true
		}
		if event.Get("type").String() == "message_stop" {
			foundMessageStop = true
		}
		if event.Get("type").String() == "content_block_start" && event.Get("content_block.type").String() == "server_tool_use" {
			t.Fatalf("Did not expect grounded search blocks in non-grounded stream output: %s", string(outputs[0]))
		}
	}
	if !foundTextDelta || !foundMessageDelta || !foundMessageStop {
		t.Fatalf("Expected buffered final output to include text_delta, message_delta, and message_stop, got %s", string(outputs[0]))
	}
}

func TestConvertGeminiResponseToClaudeNonStream_CodeExecution(t *testing.T) {
	originalRequest := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"Calculate fibonacci(20)."}],
		"tools":[{"type":"code_execution_20250825","name":"code_execution"}]
	}`)

	rawJSON := []byte(`{
		"responseId":"resp_code_exec",
		"modelVersion":"gemini-3-flash-preview",
		"candidates":[{
			"content":{"role":"model","parts":[
				{"text":"I'll calculate that."},
				{"executableCode":{"language":"PYTHON","code":"def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    print(a)\nfib(20)"}},
				{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"6765\n"}},
				{"text":"The 20th Fibonacci number is 6765."}
			]},
			"finishReason":"STOP"
		}],
		"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":18}
	}`)

	output := ConvertGeminiResponseToClaudeNonStream(context.Background(), "gemini-3-flash-preview", originalRequest, nil, rawJSON, nil)
	if got := gjson.GetBytes(output, "content.0.type").String(); got != "text" {
		t.Fatalf("Expected first block text, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.1.type").String(); got != "server_tool_use" {
		t.Fatalf("Expected second block server_tool_use, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.1.name").String(); got != "code_execution" {
		t.Fatalf("Expected server_tool_use name code_execution, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.1.input.language").String(); got != "PYTHON" {
		t.Fatalf("Expected code execution input language PYTHON, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.2.type").String(); got != "code_execution_tool_result" {
		t.Fatalf("Expected third block code_execution_tool_result, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.2.content.type").String(); got != "code_execution_result" {
		t.Fatalf("Expected nested code_execution_result, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.2.content.stdout").String(); got != "6765\n" {
		t.Fatalf("Expected stdout 6765, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.2.content.return_code").Int(); got != 0 {
		t.Fatalf("Expected return_code 0, got %d: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.3.type").String(); got != "text" {
		t.Fatalf("Expected fourth block text, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "usage.server_tool_use.code_execution_requests").Int(); got != 1 {
		t.Fatalf("Expected code_execution_requests=1, got %d: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "stop_reason").String(); got != "end_turn" {
		t.Fatalf("Expected stop_reason end_turn, got %q: %s", got, string(output))
	}
}

func TestConvertGeminiResponseToClaudeNonStream_CodeExecutionFailureMapsToStderr(t *testing.T) {
	originalRequest := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"Run broken code."}],
		"tools":[{"type":"code_execution_20250825","name":"code_execution"}]
	}`)

	rawJSON := []byte(`{
		"responseId":"resp_code_exec_failed",
		"modelVersion":"gemini-3-flash-preview",
		"candidates":[{
			"content":{"role":"model","parts":[
				{"executableCode":{"language":"PYTHON","code":"raise ValueError('boom')"}},
				{"codeExecutionResult":{"outcome":"OUTCOME_FAILED","output":"Traceback: boom\n"}}
			]},
			"finishReason":"STOP"
		}]
	}`)

	output := ConvertGeminiResponseToClaudeNonStream(context.Background(), "gemini-3-flash-preview", originalRequest, nil, rawJSON, nil)
	if got := gjson.GetBytes(output, "content.1.content.stderr").String(); got != "Traceback: boom\n" {
		t.Fatalf("Expected failed execution output in stderr, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.1.content.return_code").Int(); got != 1 {
		t.Fatalf("Expected failed execution return_code 1, got %d: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "usage.server_tool_use.code_execution_requests").Int(); got != 1 {
		t.Fatalf("Expected code_execution_requests=1, got %d: %s", got, string(output))
	}
}

func TestConvertGeminiResponseToClaude_StreamCodeExecution(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"Calculate fibonacci(20)."}],
		"tools":[{"type":"code_execution_20250825","name":"code_execution"}]
	}`)
	requestJSON := ConvertClaudeRequestToGemini("gemini-3-flash-preview", originalRequest, true)
	var param any

	chunks := [][]byte{
		[]byte(`{"candidates":[{"content":{"role":"model","parts":[{"executableCode":{"language":"PYTHON","code":"print(6765)"}}]}}],"modelVersion":"gemini-3-flash-preview","responseId":"resp_code_exec_stream"}`),
		[]byte(`{"candidates":[{"content":{"role":"model","parts":[{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"6765\n"}},{"text":"The 20th Fibonacci number is 6765."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":18},"modelVersion":"gemini-3-flash-preview","responseId":"resp_code_exec_stream"}`),
		[]byte(`[DONE]`),
	}

	var outputs [][]byte
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertGeminiResponseToClaude(ctx, "gemini-3-flash-preview", originalRequest, requestJSON, chunk, &param)...)
	}

	if len(outputs) != 3 {
		t.Fatalf("Expected three streamed outputs, got %d", len(outputs))
	}

	firstEvents := extractClaudeSSEEventData(outputs[0])
	if len(firstEvents) < 4 {
		t.Fatalf("Expected message_start and code_execution server tool events, got %s", string(outputs[0]))
	}
	if firstEvents[0].Get("type").String() != "message_start" {
		t.Fatalf("Expected first event message_start, got %s", firstEvents[0].Raw)
	}
	if firstEvents[1].Get("type").String() != "content_block_start" || firstEvents[1].Get("content_block.type").String() != "server_tool_use" {
		t.Fatalf("Expected server_tool_use start, got %s", firstEvents[1].Raw)
	}
	if firstEvents[1].Get("content_block.name").String() != "code_execution" {
		t.Fatalf("Expected server_tool_use name code_execution, got %s", firstEvents[1].Raw)
	}
	if firstEvents[2].Get("type").String() != "content_block_delta" || firstEvents[2].Get("delta.type").String() != "input_json_delta" {
		t.Fatalf("Expected input_json_delta for code execution input, got %s", firstEvents[2].Raw)
	}
	if !strings.Contains(firstEvents[2].Get("delta.partial_json").String(), "print(6765)") {
		t.Fatalf("Expected code payload in input_json_delta, got %s", firstEvents[2].Raw)
	}

	secondEvents := extractClaudeSSEEventData(outputs[1])
	foundResultBlock := false
	foundTextDelta := false
	foundUsage := false
	for _, event := range secondEvents {
		if event.Get("type").String() == "content_block_start" && event.Get("content_block.type").String() == "code_execution_tool_result" {
			foundResultBlock = true
			if got := event.Get("content_block.content.stdout").String(); got != "6765\n" {
				t.Fatalf("Expected streamed stdout 6765, got %q: %s", got, event.Raw)
			}
		}
		if event.Get("type").String() == "content_block_delta" && event.Get("delta.type").String() == "text_delta" &&
			strings.Contains(event.Get("delta.text").String(), "Fibonacci number is 6765") {
			foundTextDelta = true
		}
		if event.Get("type").String() == "message_delta" && event.Get("usage.server_tool_use.code_execution_requests").Int() == 1 {
			foundUsage = true
		}
	}
	if !foundResultBlock {
		t.Fatalf("Expected streamed code_execution_tool_result block, got %s", string(outputs[1]))
	}
	if !foundTextDelta {
		t.Fatalf("Expected streamed text delta after code execution result, got %s", string(outputs[1]))
	}
	if !foundUsage {
		t.Fatalf("Expected message_delta usage.server_tool_use.code_execution_requests=1, got %s", string(outputs[1]))
	}

	stopEvents := extractClaudeSSEEventData(outputs[2])
	if len(stopEvents) != 1 || stopEvents[0].Get("type").String() != "message_stop" {
		t.Fatalf("Expected DONE chunk to emit only message_stop, got %s", string(outputs[2]))
	}
}

func TestConvertGeminiResponseToClaudeNonStream_GroundedCodeExecution(t *testing.T) {
	originalRequest := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"Calculate and cite the source."}],
		"tools":[
			{"type":"code_execution_20250825","name":"code_execution"},
			{"type":"web_search_20250305","name":"web_search"}
		]
	}`)

	rawJSON := []byte(`{
		"responseId":"resp_grounded_code_exec",
		"modelVersion":"gemini-3-flash-preview",
		"candidates":[{
			"content":{"role":"model","parts":[
				{"executableCode":{"language":"PYTHON","code":"print(42)"}},
				{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"42\n"}},
				{"text":"The sourced answer is 42."}
			]},
			"finishReason":"STOP",
			"groundingMetadata":{
				"webSearchQueries":["answer source"],
				"groundingChunks":[
					{"web":{"uri":"https://example.com/source","title":"Example Source"}}
				],
				"groundingSupports":[
					{
						"segment":{"startIndex":0,"endIndex":24,"text":"The sourced answer is 42."},
						"groundingChunkIndices":[0]
					}
				]
			}
		}],
		"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":18}
	}`)

	output := ConvertGeminiResponseToClaudeNonStream(context.Background(), "gemini-3-flash-preview", originalRequest, nil, rawJSON, nil)
	if got := gjson.GetBytes(output, "content.0.type").String(); got != "server_tool_use" {
		t.Fatalf("Expected first block server_tool_use, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.0.name").String(); got != "code_execution" {
		t.Fatalf("Expected first server tool to be code_execution, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.1.type").String(); got != "code_execution_tool_result" {
		t.Fatalf("Expected second block code_execution_tool_result, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.2.type").String(); got != "text" {
		t.Fatalf("Expected third block text, got %q: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "content.2.citations.0.type").String(); got != "web_search_result_location" {
		t.Fatalf("Expected text block citation, got %q: %s", got, string(output))
	}

	foundWebSearchToolUse := false
	foundWebSearchToolResult := false
	for _, block := range gjson.GetBytes(output, "content").Array() {
		switch block.Get("type").String() {
		case "server_tool_use":
			if block.Get("name").String() == "web_search" {
				foundWebSearchToolUse = true
			}
		case "web_search_tool_result":
			foundWebSearchToolResult = true
		}
	}
	if !foundWebSearchToolUse {
		t.Fatalf("Expected web_search server tool after grounded text, got %s", string(output))
	}
	if !foundWebSearchToolResult {
		t.Fatalf("Expected web_search_tool_result in mixed grounded/code execution output, got %s", string(output))
	}
	if got := gjson.GetBytes(output, "usage.server_tool_use.code_execution_requests").Int(); got != 1 {
		t.Fatalf("Expected code_execution_requests=1, got %d: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "usage.server_tool_use.web_search_requests").Int(); got != 1 {
		t.Fatalf("Expected web_search_requests=1, got %d: %s", got, string(output))
	}
}

func TestConvertGeminiResponseToClaude_StreamGroundedCodeExecutionBuffersUntilDone(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{
		"model":"claude-sonnet-4-6",
		"messages":[{"role":"user","content":"Calculate and cite the source."}],
		"tools":[
			{"type":"code_execution_20250825","name":"code_execution"},
			{"type":"web_search_20250305","name":"web_search"}
		]
	}`)
	requestJSON := ConvertClaudeRequestToGemini("gemini-3-flash-preview", originalRequest, true)
	var param any

	chunks := [][]byte{
		[]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":""}]}}],"modelVersion":"gemini-3-flash-preview","responseId":"resp_buffered"}`),
		[]byte(`{"candidates":[{"content":{"role":"model","parts":[{"executableCode":{"language":"PYTHON","code":"print(42)"}}]}}],"modelVersion":"gemini-3-flash-preview","responseId":"resp_buffered"}`),
		[]byte(`{"candidates":[{"content":{"role":"model","parts":[{"codeExecutionResult":{"outcome":"OUTCOME_OK","output":"42\n"}},{"text":"The sourced answer is 42."}]},"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["answer source"],"groundingChunks":[{"web":{"uri":"https://example.com/source","title":"Example Source"}}],"groundingSupports":[{"segment":{"startIndex":0,"endIndex":24,"text":"The sourced answer is 42."},"groundingChunkIndices":[0]}]}}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":18},"modelVersion":"gemini-3-flash-preview","responseId":"resp_buffered"}`),
		[]byte(`[DONE]`),
	}

	var outputs [][]byte
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertGeminiResponseToClaude(ctx, "gemini-3-flash-preview", originalRequest, requestJSON, chunk, &param)...)
	}

	if len(outputs) != 1 {
		t.Fatalf("Expected only one buffered SSE output on DONE, got %d", len(outputs))
	}

	events := extractClaudeSSEEventData(outputs[0])
	if len(events) == 0 {
		t.Fatalf("Expected SSE events in buffered output, got %s", string(outputs[0]))
	}

	foundMessageStart := false
	foundCodeExecStart := false
	foundCodeExecResult := false
	foundCitation := false
	foundWebSearchStart := false
	foundWebSearchResult := false
	foundMessageDelta := false
	foundMessageStop := false
	codeExecStartIndex := -1
	textStartIndex := -1
	webSearchStartIndex := -1
	webSearchResultIndex := -1
	textDeltaIndex := -1
	citationIndex := -1
	for i, event := range events {
		switch event.Get("type").String() {
		case "message_start":
			foundMessageStart = true
		case "content_block_start":
			switch event.Get("content_block.type").String() {
			case "server_tool_use":
				switch event.Get("content_block.name").String() {
				case "code_execution":
					foundCodeExecStart = true
					if codeExecStartIndex == -1 {
						codeExecStartIndex = i
					}
				case "web_search":
					foundWebSearchStart = true
					if webSearchStartIndex == -1 {
						webSearchStartIndex = i
					}
				}
			case "code_execution_tool_result":
				foundCodeExecResult = true
			case "text":
				if textStartIndex == -1 {
					textStartIndex = i
				}
			case "web_search_tool_result":
				foundWebSearchResult = true
				if webSearchResultIndex == -1 {
					webSearchResultIndex = i
				}
			}
		case "content_block_delta":
			if event.Get("delta.type").String() == "citations_delta" {
				foundCitation = true
				if citationIndex == -1 {
					citationIndex = i
				}
			}
			if event.Get("delta.type").String() == "text_delta" &&
				strings.Contains(event.Get("delta.text").String(), "The sourced answer is 42") {
				textDeltaIndex = i
			}
		case "message_delta":
			foundMessageDelta = true
			if got := event.Get("usage.server_tool_use.code_execution_requests").Int(); got != 1 {
				t.Fatalf("Expected code_execution_requests=1 in message_delta, got %d: %s", got, event.Raw)
			}
			if got := event.Get("usage.server_tool_use.web_search_requests").Int(); got != 1 {
				t.Fatalf("Expected web_search_requests=1 in message_delta, got %d: %s", got, event.Raw)
			}
		case "message_stop":
			foundMessageStop = true
		}
	}

	if !foundMessageStart {
		t.Fatal("Expected buffered output to include message_start")
	}
	if !foundCodeExecStart {
		t.Fatal("Expected buffered output to include code_execution server_tool_use")
	}
	if !foundCodeExecResult {
		t.Fatal("Expected buffered output to include code_execution_tool_result")
	}
	if !foundCitation {
		t.Fatal("Expected buffered output to include citations_delta")
	}
	if !foundWebSearchStart {
		t.Fatal("Expected buffered output to include web_search server_tool_use")
	}
	if !foundWebSearchResult {
		t.Fatal("Expected buffered output to include web_search_tool_result")
	}
	if !foundMessageDelta {
		t.Fatal("Expected buffered output to include message_delta")
	}
	if !foundMessageStop {
		t.Fatal("Expected buffered output to include message_stop")
	}
	if codeExecStartIndex < 0 || textStartIndex < 0 || webSearchStartIndex < 0 || webSearchResultIndex < 0 || textDeltaIndex < 0 || citationIndex < 0 {
		t.Fatalf("Expected buffered output to include code_execution, text, and grounded search ordering markers, got %s", string(outputs[0]))
	}
	if !(codeExecStartIndex < textStartIndex && textStartIndex < webSearchStartIndex && webSearchStartIndex < webSearchResultIndex) {
		t.Fatalf("Expected code_execution blocks before text start, then web_search tool/result, got %s", string(outputs[0]))
	}
	if !(webSearchResultIndex < textDeltaIndex && webSearchResultIndex < citationIndex) {
		t.Fatalf("Expected answer text/citations after web_search_tool_result, got %s", string(outputs[0]))
	}
}

func extractClaudeSSEEventData(payload []byte) []gjson.Result {
	lines := strings.Split(string(payload), "\n")
	results := make([]gjson.Result, 0)
	for _, line := range lines {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		jsonLine := strings.TrimPrefix(line, "data: ")
		if jsonLine == "" {
			continue
		}
		results = append(results, gjson.Parse(jsonLine))
	}
	return results
}
