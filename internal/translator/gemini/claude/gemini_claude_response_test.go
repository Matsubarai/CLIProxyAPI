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

	if len(outputs) != 2 {
		t.Fatalf("Expected immediate text output plus grounded search finale, got %d outputs", len(outputs))
	}

	initialEvents := extractClaudeSSEEventData(outputs[0])
	if len(initialEvents) < 3 {
		t.Fatalf("Expected first output to contain streamed text events, got %s", string(outputs[0]))
	}
	if initialEvents[0].Get("type").String() != "message_start" {
		t.Fatalf("Expected first event to be message_start, got %s", initialEvents[0].Raw)
	}
	if initialEvents[1].Get("type").String() != "content_block_start" || initialEvents[1].Get("content_block.type").String() != "text" || initialEvents[1].Get("index").Int() != 0 {
		t.Fatalf("Expected second event to open text block 0, got %s", initialEvents[1].Raw)
	}
	if initialEvents[2].Get("type").String() != "content_block_delta" || initialEvents[2].Get("delta.type").String() != "text_delta" || !strings.Contains(initialEvents[2].Get("delta.text").String(), "Spain won Euro 2024") {
		t.Fatalf("Expected first output to stream text_delta, got %s", initialEvents[2].Raw)
	}

	events := extractClaudeSSEEventData(outputs[1])
	if len(events) == 0 {
		t.Fatalf("Expected SSE events in final grounded search output: %s", string(outputs[1]))
	}

	foundServerToolUse := false
	foundSearchResult := false
	foundCitationDelta := false
	foundMessageStop := false
	foundMessageStart := false
	foundTextDelta := false
	foundTextBlockStart := false
	foundTextBlockStop := false
	foundEmptyEncryptedIndex := false
	foundEmptyEncryptedContent := false

	for _, event := range events {
		switch event.Get("type").String() {
		case "message_start":
			foundMessageStart = true
		case "content_block_start":
			switch event.Get("content_block.type").String() {
			case "text":
				foundTextBlockStart = true
			case "server_tool_use":
				foundServerToolUse = true
			case "web_search_tool_result":
				foundSearchResult = true
				if event.Get("content_block.content.0.encrypted_content").Exists() && event.Get("content_block.content.0.encrypted_content").String() == "" {
					foundEmptyEncryptedContent = true
				}
			}
		case "content_block_delta":
			if event.Get("delta.type").String() == "citations_delta" && event.Get("delta.citation.type").String() == "web_search_result_location" {
				foundCitationDelta = true
				if event.Get("delta.citation.encrypted_index").Exists() && event.Get("delta.citation.encrypted_index").String() == "" {
					foundEmptyEncryptedIndex = true
				}
			}
			if event.Get("delta.type").String() == "text_delta" && event.Get("index").Int() == 0 && strings.Contains(event.Get("delta.text").String(), "Spain won Euro 2024") {
				foundTextDelta = true
			}
		case "content_block_stop":
			if event.Get("index").Int() == 0 {
				foundTextBlockStop = true
			}
		case "message_stop":
			foundMessageStop = true
		}
	}

	if !foundServerToolUse {
		t.Fatal("Expected server_tool_use block in buffered grounded stream output")
	}
	if !foundSearchResult {
		t.Fatal("Expected web_search_tool_result block in buffered grounded stream output")
	}
	if !foundEmptyEncryptedContent {
		t.Fatal("Expected web_search_tool_result content to include string encrypted_content")
	}
	if !foundCitationDelta {
		t.Fatal("Expected citations_delta in buffered grounded stream output")
	}
	if !foundEmptyEncryptedIndex {
		t.Fatal("Expected citations_delta to include string encrypted_index")
	}
	if !foundMessageStop {
		t.Fatal("Expected message_stop in grounded stream finale")
	}
	if foundMessageStart {
		t.Fatal("Expected final grounded stream output to omit duplicate message_start")
	}
	if foundTextBlockStart {
		t.Fatal("Expected final grounded stream output to reuse existing text block instead of reopening it")
	}
	if foundTextDelta {
		t.Fatal("Expected final grounded stream output to avoid duplicating streamed text")
	}
	if !foundTextBlockStop {
		t.Fatal("Expected final grounded stream output to stop the pre-opened text block")
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

	if len(outputs) != 2 {
		t.Fatalf("Expected immediate text output plus grounded search finale, got %d outputs", len(outputs))
	}

	events := extractClaudeSSEEventData(outputs[1])
	if len(events) == 0 {
		t.Fatalf("Expected SSE events in final grounded search output: %s", string(outputs[1]))
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

	if len(outputs) != 3 {
		t.Fatalf("Expected thinking output, text output, and grounded finale, got %d outputs", len(outputs))
	}

	events := extractClaudeSSEEventData(outputs[2])
	if len(events) == 0 {
		t.Fatalf("Expected grounded finale events, got %s", string(outputs[2]))
	}

	foundCitationOnText1 := false
	foundStopOnText1 := false
	foundServerToolUseAt2 := false
	foundSearchResultAt3 := false
	for _, event := range events {
		if event.Get("type").String() == "content_block_delta" &&
			event.Get("delta.type").String() == "citations_delta" &&
			event.Get("index").Int() == 1 {
			foundCitationOnText1 = true
		}
		if event.Get("type").String() == "content_block_stop" && event.Get("index").Int() == 1 {
			foundStopOnText1 = true
		}
		if event.Get("type").String() == "content_block_start" &&
			event.Get("content_block.type").String() == "server_tool_use" &&
			event.Get("index").Int() == 2 {
			foundServerToolUseAt2 = true
		}
		if event.Get("type").String() == "content_block_start" &&
			event.Get("content_block.type").String() == "web_search_tool_result" &&
			event.Get("index").Int() == 3 {
			foundSearchResultAt3 = true
		}
	}

	if !foundCitationOnText1 {
		t.Fatalf("Expected citations_delta to target the streamed text block index 1, got %s", string(outputs[2]))
	}
	if !foundStopOnText1 {
		t.Fatalf("Expected grounded finale to stop text block index 1, got %s", string(outputs[2]))
	}
	if !foundServerToolUseAt2 {
		t.Fatalf("Expected server_tool_use to start at index 2 after thinking+text blocks, got %s", string(outputs[2]))
	}
	if !foundSearchResultAt3 {
		t.Fatalf("Expected web_search_tool_result to start at index 3 after server_tool_use, got %s", string(outputs[2]))
	}
}

func TestConvertGeminiResponseToClaude_StreamSearchCapableWithoutGroundingRemainsImmediate(t *testing.T) {
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

	if len(outputs) != 3 {
		t.Fatalf("Expected normal immediate streaming for non-grounded search-capable request, got %d outputs", len(outputs))
	}

	initialEvents := extractClaudeSSEEventData(outputs[0])
	if len(initialEvents) < 3 || initialEvents[2].Get("delta.type").String() != "text_delta" {
		t.Fatalf("Expected first output to contain text_delta, got %s", string(outputs[0]))
	}

	finalEvents := extractClaudeSSEEventData(outputs[1])
	if len(finalEvents) == 0 || finalEvents[len(finalEvents)-1].Get("type").String() != "message_delta" {
		t.Fatalf("Expected second output to contain message_delta, got %s", string(outputs[1]))
	}
	for _, event := range finalEvents {
		if event.Get("type").String() == "content_block_start" && event.Get("content_block.type").String() == "server_tool_use" {
			t.Fatalf("Did not expect grounded search blocks in non-grounded stream output: %s", string(outputs[1]))
		}
	}

	stopEvents := extractClaudeSSEEventData(outputs[2])
	if len(stopEvents) != 1 || stopEvents[0].Get("type").String() != "message_stop" {
		t.Fatalf("Expected DONE chunk to emit only message_stop, got %s", string(outputs[2]))
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
