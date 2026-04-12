// Package claude provides response translation functionality for Claude API.
// This package handles the conversion of backend client responses into Claude-compatible
// Server-Sent Events (SSE) format, implementing a sophisticated state machine that manages
// different response types including text content, thinking processes, and function calls.
// The translation ensures proper sequencing of SSE events and maintains state across
// multiple response chunks to provide a seamless streaming experience.
package claude

import (
	"bytes"
	"context"
	"fmt"
	"sort"
	"strings"
	"sync/atomic"
	"unicode/utf8"

	translatorcommon "github.com/router-for-me/CLIProxyAPI/v6/internal/translator/common"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// Params holds parameters for response conversion.
type Params struct {
	IsGlAPIKey       bool
	HasFirstResponse bool
	ResponseType     int
	ResponseIndex    int
	HasContent       bool // Tracks whether any content (text, thinking, or tool use) has been output
	ToolNameMap      map[string]string
	SanitizedNameMap map[string]string
	SawToolCall      bool

	// Search responses need the full Gemini stream so grounded metadata can be converted into
	// Claude citations and synthetic web_search blocks once the stream is complete.
	BufferedGeminiChunks [][]byte
}

type aggregatedGeminiPart struct {
	Kind    string
	Text    strings.Builder
	Thought bool
	Name    string
	Args    strings.Builder
}

type groundedSupport struct {
	Start     int
	End       int
	CitedText string
	Citations [][]byte
}

// toolUseIDCounter provides a process-wide unique counter for tool use identifiers.
var toolUseIDCounter uint64

// serverToolUseIDCounter provides a process-wide unique counter for synthetic server tool use identifiers.
var serverToolUseIDCounter uint64

// ConvertGeminiResponseToClaude performs sophisticated streaming response format conversion.
// This function implements a complex state machine that translates backend client responses
// into Claude-compatible Server-Sent Events (SSE) format. It manages different response types
// and handles state transitions between content blocks, thinking processes, and function calls.
//
// Response type states: 0=none, 1=content, 2=thinking, 3=function
// The function maintains state across multiple calls to ensure proper SSE event sequencing.
//
// Parameters:
//   - ctx: The context for the request.
//   - modelName: The name of the model.
//   - rawJSON: The raw JSON response from the Gemini API.
//   - param: A pointer to a parameter object for the conversion.
//
// Returns:
//   - [][]byte: A slice of bytes, each containing a Claude-compatible SSE payload.
func ConvertGeminiResponseToClaude(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) [][]byte {
	if *param == nil {
		*param = &Params{
			IsGlAPIKey:           false,
			HasFirstResponse:     false,
			ResponseType:         0,
			ResponseIndex:        0,
			ToolNameMap:          util.ToolNameMapFromClaudeRequest(originalRequestRawJSON),
			SanitizedNameMap:     util.SanitizedToolNameMap(originalRequestRawJSON),
			SawToolCall:          false,
			BufferedGeminiChunks: nil,
		}
	}

	searchEnabled := searchEnabledForGeminiResponse(originalRequestRawJSON, requestRawJSON)
	if searchEnabled && !bytes.Equal(rawJSON, []byte("[DONE]")) {
		(*param).(*Params).BufferedGeminiChunks = append((*param).(*Params).BufferedGeminiChunks, append([]byte(nil), rawJSON...))
	}
	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		// If the completed stream resolves to a grounded search answer, replace the normal
		// message_stop path with a Claude-style grounded finale.
		if outputs := finalizeGroundedSearchStreamOnDone(originalRequestRawJSON, (*param).(*Params)); len(outputs) > 0 {
			return outputs
		}
	}
	if chunkItems := geminiChunkItems(rawJSON); len(chunkItems) > 1 {
		outputs := make([][]byte, 0, len(chunkItems))
		for _, item := range chunkItems {
			outputs = append(outputs, convertGeminiResponseToClaudeImmediate([]byte(item.Raw), param, searchEnabled)...)
		}
		return outputs
	}
	if chunkItems := geminiChunkItems(rawJSON); len(chunkItems) == 1 && bytes.TrimSpace(rawJSON)[0] == '[' {
		return convertGeminiResponseToClaudeImmediate([]byte(chunkItems[0].Raw), param, searchEnabled)
	}
	return convertGeminiResponseToClaudeImmediate(rawJSON, param, searchEnabled)
}

func convertGeminiResponseToClaudeImmediate(rawJSON []byte, param *any, searchEnabled bool) [][]byte {
	if bytes.Equal(rawJSON, []byte("[DONE]")) {
		// Only send message_stop if we have actually output content
		if (*param).(*Params).HasContent {
			return [][]byte{translatorcommon.AppendSSEEventString(nil, "message_stop", `{"type":"message_stop"}`, 3)}
		}
		return [][]byte{}
	}

	output := make([]byte, 0, 1024)
	appendEvent := func(event, payload string) {
		output = translatorcommon.AppendSSEEventString(output, event, payload, 3)
	}

	// Initialize the streaming session with a message_start event
	// This is only sent for the very first response chunk
	if !(*param).(*Params).HasFirstResponse {
		// Create the initial message structure with default values
		// This follows the Claude API specification for streaming message initialization
		messageStartTemplate := []byte(`{"type":"message_start","message":{"id":"msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY","type":"message","role":"assistant","content":[],"model":"claude-3-5-sonnet-20241022","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}`)

		// Override default values with actual response metadata if available
		if modelVersionResult := gjson.GetBytes(rawJSON, "modelVersion"); modelVersionResult.Exists() {
			messageStartTemplate, _ = sjson.SetBytes(messageStartTemplate, "message.model", modelVersionResult.String())
		}
		if responseIDResult := gjson.GetBytes(rawJSON, "responseId"); responseIDResult.Exists() {
			messageStartTemplate, _ = sjson.SetBytes(messageStartTemplate, "message.id", responseIDResult.String())
		}
		appendEvent("message_start", string(messageStartTemplate))

		(*param).(*Params).HasFirstResponse = true
	}

	// Process the response parts array from the backend client
	// Each part can contain text content, thinking content, or function calls
	partsResult := gjson.GetBytes(rawJSON, "candidates.0.content.parts")
	if partsResult.IsArray() {
		partResults := partsResult.Array()
		for i := 0; i < len(partResults); i++ {
			partResult := partResults[i]

			// Extract the different types of content from each part
			partTextResult := partResult.Get("text")
			functionCallResult := partResult.Get("functionCall")

			// Handle text content (both regular content and thinking)
			if partTextResult.Exists() {
				// Process thinking content (internal reasoning)
				if partResult.Get("thought").Bool() {
					// Continue existing thinking block
					if (*param).(*Params).ResponseType == 2 {
						data, _ := sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"thinking_delta","thinking":""}}`, (*param).(*Params).ResponseIndex)), "delta.thinking", partTextResult.String())
						appendEvent("content_block_delta", string(data))
						(*param).(*Params).HasContent = true
					} else {
						// Transition from another state to thinking
						// First, close any existing content block
						if (*param).(*Params).ResponseType != 0 {
							if (*param).(*Params).ResponseType == 2 {
								// output = output + "event: content_block_delta\n"
								// output = output + fmt.Sprintf(`data: {"type":"content_block_delta","index":%d,"delta":{"type":"signature_delta","signature":null}}`, (*param).(*Params).ResponseIndex)
								// output = output + "\n\n\n"
							}
							appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))
							(*param).(*Params).ResponseIndex++
						}

						// Start a new thinking content block
						appendEvent("content_block_start", fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"thinking","thinking":""}}`, (*param).(*Params).ResponseIndex))
						data, _ := sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"thinking_delta","thinking":""}}`, (*param).(*Params).ResponseIndex)), "delta.thinking", partTextResult.String())
						appendEvent("content_block_delta", string(data))
						(*param).(*Params).ResponseType = 2 // Set state to thinking
						(*param).(*Params).HasContent = true
					}
				} else {
					// Process regular text content (user-visible output)
					// Continue existing text block
					if (*param).(*Params).ResponseType == 1 {
						data, _ := sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":""}}`, (*param).(*Params).ResponseIndex)), "delta.text", partTextResult.String())
						appendEvent("content_block_delta", string(data))
						(*param).(*Params).HasContent = true
					} else {
						// Transition from another state to text content
						// First, close any existing content block
						if (*param).(*Params).ResponseType != 0 {
							if (*param).(*Params).ResponseType == 2 {
								// output = output + "event: content_block_delta\n"
								// output = output + fmt.Sprintf(`data: {"type":"content_block_delta","index":%d,"delta":{"type":"signature_delta","signature":null}}`, (*param).(*Params).ResponseIndex)
								// output = output + "\n\n\n"
							}
							appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))
							(*param).(*Params).ResponseIndex++
						}

						// Start a new text content block
						appendEvent("content_block_start", fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"text","text":""}}`, (*param).(*Params).ResponseIndex))
						data, _ := sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":""}}`, (*param).(*Params).ResponseIndex)), "delta.text", partTextResult.String())
						appendEvent("content_block_delta", string(data))
						(*param).(*Params).ResponseType = 1 // Set state to content
						(*param).(*Params).HasContent = true
					}
				}
			} else if functionCallResult.Exists() {
				// Handle function/tool calls from the AI model
				// This processes tool usage requests and formats them for Claude API compatibility
				(*param).(*Params).SawToolCall = true
				upstreamToolName := functionCallResult.Get("name").String()
				upstreamToolName = util.RestoreSanitizedToolName((*param).(*Params).SanitizedNameMap, upstreamToolName)
				clientToolName := util.MapToolName((*param).(*Params).ToolNameMap, upstreamToolName)

				// FIX: Handle streaming split/delta where name might be empty in subsequent chunks.
				// If we are already in tool use mode and name is empty, treat as continuation (delta).
				if (*param).(*Params).ResponseType == 3 && upstreamToolName == "" {
					if fcArgsResult := functionCallResult.Get("args"); fcArgsResult.Exists() {
						data, _ := sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":""}}`, (*param).(*Params).ResponseIndex)), "delta.partial_json", fcArgsResult.Raw)
						appendEvent("content_block_delta", string(data))
					}
					// Continue to next part without closing/opening logic
					continue
				}

				// Handle state transitions when switching to function calls
				// Close any existing function call block first
				if (*param).(*Params).ResponseType == 3 {
					appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))
					(*param).(*Params).ResponseIndex++
					(*param).(*Params).ResponseType = 0
				}

				// Special handling for thinking state transition
				if (*param).(*Params).ResponseType == 2 {
					// output = output + "event: content_block_delta\n"
					// output = output + fmt.Sprintf(`data: {"type":"content_block_delta","index":%d,"delta":{"type":"signature_delta","signature":null}}`, (*param).(*Params).ResponseIndex)
					// output = output + "\n\n\n"
				}

				// Close any other existing content block
				if (*param).(*Params).ResponseType != 0 {
					appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))
					(*param).(*Params).ResponseIndex++
				}

				// Start a new tool use content block
				// This creates the structure for a function call in Claude format
				// Create the tool use block with unique ID and function details
				data := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"tool_use","id":"","name":"","input":{}}}`, (*param).(*Params).ResponseIndex))
				data, _ = sjson.SetBytes(data, "content_block.id", util.SanitizeClaudeToolID(fmt.Sprintf("%s-%d", upstreamToolName, atomic.AddUint64(&toolUseIDCounter, 1))))
				data, _ = sjson.SetBytes(data, "content_block.name", clientToolName)
				appendEvent("content_block_start", string(data))

				if fcArgsResult := functionCallResult.Get("args"); fcArgsResult.Exists() {
					data, _ = sjson.SetBytes([]byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":""}}`, (*param).(*Params).ResponseIndex)), "delta.partial_json", fcArgsResult.Raw)
					appendEvent("content_block_delta", string(data))
				}
				(*param).(*Params).ResponseType = 3
				(*param).(*Params).HasContent = true
			}
		}
	}

	usageResult := gjson.GetBytes(rawJSON, "usageMetadata")
	if usageResult.Exists() && bytes.Contains(rawJSON, []byte(`"finishReason"`)) {
		if searchEnabled && hasGeminiGroundingMetadata(rawJSON) {
			if len(output) == 0 {
				return nil
			}
			return [][]byte{output}
		}
		if candidatesTokenCountResult := usageResult.Get("candidatesTokenCount"); candidatesTokenCountResult.Exists() {
			// Only send final events if we have actually output content
			if (*param).(*Params).HasContent {
				appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))

				template := []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}`)
				if (*param).(*Params).SawToolCall {
					template = []byte(`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}`)
				} else if finish := gjson.GetBytes(rawJSON, "candidates.0.finishReason"); finish.Exists() && finish.String() == "MAX_TOKENS" {
					template = []byte(`{"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}`)
				}

				thoughtsTokenCount := usageResult.Get("thoughtsTokenCount").Int()
				template, _ = sjson.SetBytes(template, "usage.output_tokens", candidatesTokenCountResult.Int()+thoughtsTokenCount)
				template, _ = sjson.SetBytes(template, "usage.input_tokens", usageResult.Get("promptTokenCount").Int())

				appendEvent("message_delta", string(template))
			}
		}
	}

	if len(output) == 0 {
		return nil
	}
	return [][]byte{output}
}

// ConvertGeminiResponseToClaudeNonStream converts a non-streaming Gemini response to a non-streaming Claude response.
//
// Parameters:
//   - ctx: The context for the request.
//   - modelName: The name of the model.
//   - rawJSON: The raw JSON response from the Gemini API.
//   - param: A pointer to a parameter object for the conversion.
//
// Returns:
//   - []byte: A Claude-compatible JSON response.
func ConvertGeminiResponseToClaudeNonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) []byte {
	_ = requestRawJSON
	return buildClaudeMessageFromGeminiResponse(originalRequestRawJSON, rawJSON)
}

func buildClaudeMessageFromGeminiResponse(originalRequestRawJSON, rawJSON []byte) []byte {
	root := gjson.ParseBytes(rawJSON)
	toolNameMap := util.ToolNameMapFromClaudeRequest(originalRequestRawJSON)
	sanitizedNameMap := util.SanitizedToolNameMap(originalRequestRawJSON)

	out := []byte(`{"id":"","type":"message","role":"assistant","model":"","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}`)
	out, _ = sjson.SetBytes(out, "id", root.Get("responseId").String())
	out, _ = sjson.SetBytes(out, "model", root.Get("modelVersion").String())

	inputTokens := root.Get("usageMetadata.promptTokenCount").Int()
	outputTokens := root.Get("usageMetadata.candidatesTokenCount").Int() + root.Get("usageMetadata.thoughtsTokenCount").Int()
	out, _ = sjson.SetBytes(out, "usage.input_tokens", inputTokens)
	out, _ = sjson.SetBytes(out, "usage.output_tokens", outputTokens)

	parts := root.Get("candidates.0.content.parts")
	groundingMetadata := root.Get("candidates.0.groundingMetadata")
	hasToolCall := false

	if groundingMetadata.Exists() && !hasGeminiFunctionCall(parts) {
		// Gemini grounding metadata is translated into Claude's built-in web search response
		// shape: cited text, synthetic server_tool_use, and web_search_tool_result.
		contentBlocks := buildClaudeGroundedContent(parts, groundingMetadata)
		for _, block := range contentBlocks {
			out, _ = sjson.SetRawBytes(out, "content.-1", block)
		}
		if count := groundedWebSearchRequestCount(groundingMetadata); count > 0 {
			out, _ = sjson.SetBytes(out, "usage.server_tool_use.web_search_requests", count)
		}
	} else {
		contentBlocks, sawToolCall := buildClaudeContentFromGeminiParts(parts, toolNameMap, sanitizedNameMap)
		hasToolCall = sawToolCall
		for _, block := range contentBlocks {
			out, _ = sjson.SetRawBytes(out, "content.-1", block)
		}
	}

	stopReason := "end_turn"
	if hasToolCall {
		stopReason = "tool_use"
	} else if groundingMetadata.Exists() {
		stopReason = "end_turn"
	} else if finish := root.Get("candidates.0.finishReason"); finish.Exists() {
		switch finish.String() {
		case "MAX_TOKENS":
			stopReason = "max_tokens"
		case "STOP", "FINISH_REASON_UNSPECIFIED", "UNKNOWN":
			stopReason = "end_turn"
		default:
			stopReason = "end_turn"
		}
	}
	out, _ = sjson.SetBytes(out, "stop_reason", stopReason)

	if inputTokens == int64(0) && outputTokens == int64(0) && !root.Get("usageMetadata").Exists() && !gjson.GetBytes(out, "usage.server_tool_use").Exists() {
		out, _ = sjson.DeleteBytes(out, "usage")
	}

	return out
}

func buildClaudeContentFromGeminiParts(parts gjson.Result, toolNameMap, sanitizedNameMap map[string]string) ([][]byte, bool) {
	contentBlocks := make([][]byte, 0)
	textBuilder := strings.Builder{}
	thinkingBuilder := strings.Builder{}
	toolIDCounter := 0
	hasToolCall := false

	flushText := func() {
		if textBuilder.Len() == 0 {
			return
		}
		block := []byte(`{"type":"text","text":""}`)
		block, _ = sjson.SetBytes(block, "text", textBuilder.String())
		contentBlocks = append(contentBlocks, block)
		textBuilder.Reset()
	}

	flushThinking := func() {
		if thinkingBuilder.Len() == 0 {
			return
		}
		block := []byte(`{"type":"thinking","thinking":""}`)
		block, _ = sjson.SetBytes(block, "thinking", thinkingBuilder.String())
		contentBlocks = append(contentBlocks, block)
		thinkingBuilder.Reset()
	}

	if parts.IsArray() {
		for _, part := range parts.Array() {
			if text := part.Get("text"); text.Exists() && text.String() != "" {
				if part.Get("thought").Bool() {
					flushText()
					thinkingBuilder.WriteString(text.String())
					continue
				}
				flushThinking()
				textBuilder.WriteString(text.String())
				continue
			}

			if functionCall := part.Get("functionCall"); functionCall.Exists() {
				flushThinking()
				flushText()
				hasToolCall = true

				upstreamToolName := functionCall.Get("name").String()
				upstreamToolName = util.RestoreSanitizedToolName(sanitizedNameMap, upstreamToolName)
				clientToolName := util.MapToolName(toolNameMap, upstreamToolName)
				toolIDCounter++
				toolBlock := []byte(`{"type":"tool_use","id":"","name":"","input":{}}`)
				toolBlock, _ = sjson.SetBytes(toolBlock, "id", util.SanitizeClaudeToolID(fmt.Sprintf("%s-%d", upstreamToolName, toolIDCounter)))
				toolBlock, _ = sjson.SetBytes(toolBlock, "name", clientToolName)
				inputRaw := "{}"
				if args := functionCall.Get("args"); args.Exists() && gjson.Valid(args.Raw) && args.IsObject() {
					inputRaw = args.Raw
				}
				toolBlock, _ = sjson.SetRawBytes(toolBlock, "input", []byte(inputRaw))
				contentBlocks = append(contentBlocks, toolBlock)
				continue
			}
		}
	}

	flushThinking()
	flushText()

	return contentBlocks, hasToolCall
}

func buildClaudeGroundedContent(parts, groundingMetadata gjson.Result) [][]byte {
	contentBlocks := make([][]byte, 0)
	contentBlocks = append(contentBlocks, buildClaudeThinkingBlocks(parts)...)

	visibleText := concatenateVisibleGeminiText(parts)
	for _, block := range buildClaudeGroundedTextBlocks(visibleText, groundingMetadata) {
		contentBlocks = append(contentBlocks, block)
	}

	toolUseID := util.SanitizeClaudeToolID(fmt.Sprintf("srvtoolu_%d", atomic.AddUint64(&serverToolUseIDCounter, 1)))
	serverToolUse := []byte(`{"type":"server_tool_use","id":"","name":"web_search","input":{}}`)
	serverToolUse, _ = sjson.SetBytes(serverToolUse, "id", toolUseID)
	if query := firstGroundedQuery(groundingMetadata); query != "" {
		serverToolUse, _ = sjson.SetBytes(serverToolUse, "input.query", query)
	}
	contentBlocks = append(contentBlocks, serverToolUse)

	searchResults := buildClaudeWebSearchResults(groundingMetadata)
	toolResult := []byte(`{"type":"web_search_tool_result","tool_use_id":"","content":[]}`)
	toolResult, _ = sjson.SetBytes(toolResult, "tool_use_id", toolUseID)
	for _, result := range searchResults {
		toolResult, _ = sjson.SetRawBytes(toolResult, "content.-1", result)
	}
	contentBlocks = append(contentBlocks, toolResult)

	return contentBlocks
}

func buildClaudeThinkingBlocks(parts gjson.Result) [][]byte {
	blocks := make([][]byte, 0)
	if !parts.IsArray() {
		return blocks
	}

	var thinking strings.Builder
	flush := func() {
		if thinking.Len() == 0 {
			return
		}
		block := []byte(`{"type":"thinking","thinking":""}`)
		block, _ = sjson.SetBytes(block, "thinking", thinking.String())
		blocks = append(blocks, block)
		thinking.Reset()
	}

	for _, part := range parts.Array() {
		if part.Get("thought").Bool() && part.Get("text").String() != "" {
			thinking.WriteString(part.Get("text").String())
			continue
		}
		flush()
	}
	flush()

	return blocks
}

func concatenateVisibleGeminiText(parts gjson.Result) string {
	if !parts.IsArray() {
		return ""
	}
	var text strings.Builder
	for _, part := range parts.Array() {
		if part.Get("thought").Bool() {
			continue
		}
		if v := part.Get("text"); v.Exists() {
			text.WriteString(v.String())
		}
	}
	return text.String()
}

func buildClaudeWebSearchResults(groundingMetadata gjson.Result) [][]byte {
	results := make([][]byte, 0)
	chunks := groundingMetadata.Get("groundingChunks")
	if !chunks.IsArray() {
		return results
	}

	for _, chunk := range chunks.Array() {
		web := chunk.Get("web")
		if !web.Exists() {
			continue
		}
		uri := web.Get("uri").String()
		title := web.Get("title").String()
		if uri == "" && title == "" {
			continue
		}

		result := []byte(`{"type":"web_search_result"}`)
		if uri != "" {
			result, _ = sjson.SetBytes(result, "url", uri)
		}
		if title != "" {
			result, _ = sjson.SetBytes(result, "title", title)
		}
		result = ensureClaudeWebSearchResultEncryptedContent(result)
		results = append(results, result)
	}

	return results
}

func buildClaudeGroundedTextBlocks(text string, groundingMetadata gjson.Result) [][]byte {
	if text == "" {
		return nil
	}

	supports := resolvedGroundedSupports(text, groundedSupportsFromMetadata(groundingMetadata))
	if len(supports) == 0 {
		block := []byte(`{"type":"text","text":""}`)
		block, _ = sjson.SetBytes(block, "text", text)
		return [][]byte{block}
	}

	blocks := make([][]byte, 0, len(supports)+1)
	cursor := 0
	runes := []rune(text)
	textLen := len(runes)
	for _, support := range supports {
		if support.Start > cursor {
			block := []byte(`{"type":"text","text":""}`)
			block, _ = sjson.SetBytes(block, "text", string(runes[cursor:support.Start]))
			blocks = append(blocks, block)
		}

		if support.End <= support.Start || support.Start >= textLen {
			cursor = maxInt(cursor, support.End)
			continue
		}

		end := minInt(support.End, textLen)
		citedText := string(runes[support.Start:end])
		if len(support.Citations) == 0 {
			block := []byte(`{"type":"text","text":""}`)
			block, _ = sjson.SetBytes(block, "text", citedText)
			blocks = append(blocks, block)
			cursor = end
			continue
		}

		block := []byte(`{"type":"text","text":"","citations":[]}`)
		block, _ = sjson.SetBytes(block, "text", citedText)
		for _, citation := range support.Citations {
			block, _ = sjson.SetRawBytes(block, "citations.-1", citation)
		}
		blocks = append(blocks, block)
		cursor = end
	}

	if cursor < textLen {
		block := []byte(`{"type":"text","text":""}`)
		block, _ = sjson.SetBytes(block, "text", string(runes[cursor:]))
		blocks = append(blocks, block)
	}

	return blocks
}

func resolvedGroundedSupports(text string, supports []groundedSupport) []groundedSupport {
	if len(supports) == 0 {
		return nil
	}

	textLen := len([]rune(text))
	resolved := make([]groundedSupport, 0, len(supports))
	for _, support := range supports {
		start, end := resolveGroundedSupportRange(text, support.Start, support.End, support.CitedText)
		if start > textLen {
			continue
		}
		if end < start {
			end = start
		}
		if end > textLen {
			end = textLen
		}
		support.Start = start
		support.End = end
		resolved = append(resolved, support)
	}

	sort.Slice(resolved, func(i, j int) bool {
		if resolved[i].Start == resolved[j].Start {
			return resolved[i].End < resolved[j].End
		}
		return resolved[i].Start < resolved[j].Start
	})

	cursor := 0
	for i := range resolved {
		if resolved[i].Start < cursor {
			resolved[i].Start = cursor
		}
		if resolved[i].End < resolved[i].Start {
			resolved[i].End = resolved[i].Start
		}
		cursor = maxInt(cursor, resolved[i].End)
	}

	return resolved
}

func resolveGroundedSupportRange(text string, start, end int, citedText string) (int, int) {
	runes := []rune(text)
	runeLen := len(runes)

	runeStart := clampInt(start, 0, runeLen)
	runeEnd := clampInt(end, runeStart, runeLen)
	runeSlice := string(runes[runeStart:runeEnd])

	byteStart, byteEnd := normalizeGroundingByteOffsets([]byte(text), start, end)
	byteSlice := text[byteStart:byteEnd]
	byteRuneStart := utf8.RuneCountInString(text[:byteStart])
	byteRuneEnd := byteRuneStart + utf8.RuneCountInString(byteSlice)

	// Gemini grounding offsets are usually character-based, but some multilingual responses
	// arrive effectively as UTF-8 byte offsets. Prefer the interpretation that matches the
	// cited segment text and fall back to byte offsets when rune offsets are clearly invalid.
	if citedText != "" {
		byteExact := byteSlice == citedText
		runeExact := runeSlice == citedText
		switch {
		case byteExact && !runeExact:
			return byteRuneStart, byteRuneEnd
		case runeExact && !byteExact:
			return runeStart, runeEnd
		case byteExact && runeExact:
			if end > runeLen {
				return byteRuneStart, byteRuneEnd
			}
			return runeStart, runeEnd
		}
	}

	if end > runeLen {
		return byteRuneStart, byteRuneEnd
	}
	return runeStart, runeEnd
}

func normalizeGroundingByteOffsets(text []byte, start, end int) (int, int) {
	start = clampInt(start, 0, len(text))
	end = clampInt(end, start, len(text))

	for start > 0 && start < len(text) && !utf8.RuneStart(text[start]) {
		start--
	}
	for end > start && end < len(text) && !utf8.RuneStart(text[end]) {
		end++
	}
	return start, end
}

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func groundedSupportsFromMetadata(groundingMetadata gjson.Result) []groundedSupport {
	supportsResult := groundingMetadata.Get("groundingSupports")
	if !supportsResult.IsArray() {
		return nil
	}

	chunks := groundingMetadata.Get("groundingChunks").Array()
	supports := make([]groundedSupport, 0, len(supportsResult.Array()))
	for _, support := range supportsResult.Array() {
		start := int(support.Get("segment.startIndex").Int())
		end := int(support.Get("segment.endIndex").Int())
		citedText := support.Get("segment.text").String()
		if end < start {
			continue
		}

		citations := make([][]byte, 0)
		for _, idxResult := range support.Get("groundingChunkIndices").Array() {
			idx := int(idxResult.Int())
			if idx < 0 || idx >= len(chunks) {
				continue
			}
			web := chunks[idx].Get("web")
			if !web.Exists() {
				continue
			}
			uri := web.Get("uri").String()
			title := web.Get("title").String()
			if uri == "" && title == "" {
				continue
			}

			citation := []byte(`{"type":"web_search_result_location"}`)
			if uri != "" {
				citation, _ = sjson.SetBytes(citation, "url", uri)
			}
			if title != "" {
				citation, _ = sjson.SetBytes(citation, "title", title)
			}
			if citedText != "" {
				citation, _ = sjson.SetBytes(citation, "cited_text", citedText)
			}
			citation = ensureClaudeCitationEncryptedIndex(citation)
			citations = append(citations, citation)
		}

		supports = append(supports, groundedSupport{
			Start:     start,
			End:       end,
			CitedText: citedText,
			Citations: citations,
		})
	}

	sort.Slice(supports, func(i, j int) bool {
		if supports[i].Start == supports[j].Start {
			return supports[i].End < supports[j].End
		}
		return supports[i].Start < supports[j].Start
	})

	cursor := 0
	for i := range supports {
		if supports[i].Start < cursor {
			supports[i].Start = cursor
		}
		if supports[i].End < supports[i].Start {
			supports[i].End = supports[i].Start
		}
		cursor = maxInt(cursor, supports[i].End)
	}

	return supports
}

func groundedWebSearchRequestCount(groundingMetadata gjson.Result) int64 {
	queries := groundingMetadata.Get("webSearchQueries")
	if queries.IsArray() && len(queries.Array()) > 0 {
		return int64(len(queries.Array()))
	}
	if len(buildClaudeWebSearchResults(groundingMetadata)) > 0 {
		return 1
	}
	return 0
}

func firstGroundedQuery(groundingMetadata gjson.Result) string {
	queries := groundingMetadata.Get("webSearchQueries")
	if !queries.IsArray() {
		return ""
	}
	arr := queries.Array()
	if len(arr) == 0 {
		return ""
	}
	return arr[0].String()
}

func ensureClaudeCitationEncryptedIndex(citation []byte) []byte {
	if !gjson.GetBytes(citation, "encrypted_index").Exists() {
		citation, _ = sjson.SetBytes(citation, "encrypted_index", "")
		return citation
	}
	if gjson.GetBytes(citation, "encrypted_index").Type != gjson.String {
		citation, _ = sjson.SetBytes(citation, "encrypted_index", gjson.GetBytes(citation, "encrypted_index").String())
	}
	return citation
}

func ensureClaudeWebSearchResultEncryptedContent(result []byte) []byte {
	if !gjson.GetBytes(result, "encrypted_content").Exists() {
		result, _ = sjson.SetBytes(result, "encrypted_content", "")
		return result
	}
	if gjson.GetBytes(result, "encrypted_content").Type != gjson.String {
		result, _ = sjson.SetBytes(result, "encrypted_content", gjson.GetBytes(result, "encrypted_content").String())
	}
	return result
}

func finalizeGroundedSearchStreamOnDone(originalRequestRawJSON []byte, param *Params) [][]byte {
	if len(param.BufferedGeminiChunks) == 0 {
		return nil
	}
	// Reconstruct a single Gemini response from the streamed chunks so non-stream grounding
	// translation can be reused for the final Claude search-shaped payload.
	finalResponse := aggregateGeminiBufferedChunks(param.BufferedGeminiChunks)
	param.BufferedGeminiChunks = nil
	if len(finalResponse) == 0 {
		return nil
	}
	parts := gjson.GetBytes(finalResponse, "candidates.0.content.parts")
	if hasGeminiFunctionCall(parts) || !gjson.GetBytes(finalResponse, "candidates.0.groundingMetadata").Exists() {
		return nil
	}
	message := buildClaudeMessageFromGeminiResponse(originalRequestRawJSON, finalResponse)
	return [][]byte{renderClaudeGroundedSearchFinaleAsSSE(message, param.ResponseType, param.ResponseIndex)}
}

func hasGeminiFunctionCall(parts gjson.Result) bool {
	if !parts.IsArray() {
		return false
	}
	for _, part := range parts.Array() {
		if part.Get("functionCall").Exists() {
			return true
		}
	}
	return false
}

func hasGeminiGroundingMetadata(rawJSON []byte) bool {
	for _, item := range geminiChunkItems(rawJSON) {
		if item.Get("candidates.0.groundingMetadata").Exists() {
			return true
		}
	}
	return false
}

func renderClaudeGroundedSearchFinaleAsSSE(message []byte, responseType, responseIndex int) []byte {
	output := make([]byte, 0, len(message)+512)
	textIndex := -1
	nextIndex := responseIndex
	if responseType != 0 {
		nextIndex = responseIndex + 1
	}
	if responseType == 1 {
		textIndex = responseIndex
	}

	emitStop := func(index int) {
		stop := []byte(fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, index))
		output = translatorcommon.AppendSSEEventBytes(output, "content_block_stop", stop, 3)
	}

	if len(message) > 0 {
		content := gjson.GetBytes(message, "content")
		hasTextBlock := false
		for _, block := range content.Array() {
			if block.Get("type").String() != "text" {
				continue
			}
			hasTextBlock = true
			if textIndex < 0 {
				continue
			}
			for _, citation := range block.Get("citations").Array() {
				citationDelta := []byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"citations_delta","citation":{}}}`, textIndex))
				citationDelta, _ = sjson.SetRawBytes(citationDelta, "delta.citation", []byte(citation.Raw))
				output = translatorcommon.AppendSSEEventBytes(output, "content_block_delta", citationDelta, 3)
			}
		}
		if hasTextBlock && textIndex >= 0 {
			emitStop(textIndex)
		}

		for _, block := range content.Array() {
			switch block.Get("type").String() {
			case "server_tool_use":
				start := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"server_tool_use","id":"","name":""}}`, nextIndex))
				start, _ = sjson.SetBytes(start, "content_block.id", block.Get("id").String())
				start, _ = sjson.SetBytes(start, "content_block.name", block.Get("name").String())
				output = translatorcommon.AppendSSEEventBytes(output, "content_block_start", start, 3)
				if input := block.Get("input"); input.Exists() {
					delta := []byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":""}}`, nextIndex))
					delta, _ = sjson.SetBytes(delta, "delta.partial_json", input.Raw)
					output = translatorcommon.AppendSSEEventBytes(output, "content_block_delta", delta, 3)
				}
				emitStop(nextIndex)
				nextIndex++
			case "web_search_tool_result":
				start := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"web_search_tool_result","tool_use_id":"","content":[]}}`, nextIndex))
				start, _ = sjson.SetBytes(start, "content_block.tool_use_id", block.Get("tool_use_id").String())
				if contentResult := block.Get("content"); contentResult.Exists() {
					start, _ = sjson.SetRawBytes(start, "content_block.content", []byte(contentResult.Raw))
				}
				output = translatorcommon.AppendSSEEventBytes(output, "content_block_start", start, 3)
				emitStop(nextIndex)
				nextIndex++
			}
		}
	}

	messageDelta := []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}`)
	if stopReason := gjson.GetBytes(message, "stop_reason").String(); stopReason != "" {
		messageDelta, _ = sjson.SetBytes(messageDelta, "delta.stop_reason", stopReason)
	}
	if usage := gjson.GetBytes(message, "usage"); usage.Exists() {
		messageDelta, _ = sjson.SetRawBytes(messageDelta, "usage", []byte(usage.Raw))
	}
	output = translatorcommon.AppendSSEEventBytes(output, "message_delta", messageDelta, 3)
	output = translatorcommon.AppendSSEEventString(output, "message_stop", `{"type":"message_stop"}`, 3)

	return output
}

func aggregateGeminiBufferedChunks(chunks [][]byte) []byte {
	if len(chunks) == 0 {
		return nil
	}

	out := []byte(`{"candidates":[{"content":{"role":"model","parts":[]}}]}`)
	aggregatedParts := make([]*aggregatedGeminiPart, 0)
	var modelVersion string
	var responseID string
	var finishReason string
	var groundingMetadataRaw string
	var usageMetadataRaw string

	for _, chunk := range chunks {
		for _, root := range geminiChunkItems(chunk) {
			if mv := root.Get("modelVersion"); mv.Exists() && mv.String() != "" {
				modelVersion = mv.String()
			}
			if rid := root.Get("responseId"); rid.Exists() && rid.String() != "" {
				responseID = rid.String()
			}
			if fr := root.Get("candidates.0.finishReason"); fr.Exists() && fr.String() != "" {
				finishReason = fr.String()
			}
			if gm := root.Get("candidates.0.groundingMetadata"); gm.Exists() && gm.Raw != "" {
				groundingMetadataRaw = gm.Raw
			}
			if usage := root.Get("usageMetadata"); usage.Exists() && usage.Raw != "" {
				usageMetadataRaw = usage.Raw
			}

			parts := root.Get("candidates.0.content.parts")
			if !parts.IsArray() {
				continue
			}
			// Gemini streams text/thoughts/functionCall args incrementally. Merge adjacent deltas
			// back into complete parts before translating them as a final Claude message.
			for _, part := range parts.Array() {
				if text := part.Get("text"); text.Exists() {
					thought := part.Get("thought").Bool()
					if len(aggregatedParts) > 0 && aggregatedParts[len(aggregatedParts)-1].Kind == "text" && aggregatedParts[len(aggregatedParts)-1].Thought == thought {
						aggregatedParts[len(aggregatedParts)-1].Text.WriteString(text.String())
					} else {
						p := &aggregatedGeminiPart{Kind: "text", Thought: thought}
						p.Text.WriteString(text.String())
						aggregatedParts = append(aggregatedParts, p)
					}
					continue
				}
				if fc := part.Get("functionCall"); fc.Exists() {
					name := fc.Get("name").String()
					argsRaw := ""
					if args := fc.Get("args"); args.Exists() {
						argsRaw = args.Raw
					}
					if len(aggregatedParts) > 0 && aggregatedParts[len(aggregatedParts)-1].Kind == "functionCall" && name == "" {
						aggregatedParts[len(aggregatedParts)-1].Args.WriteString(argsRaw)
					} else {
						p := &aggregatedGeminiPart{Kind: "functionCall", Name: name}
						p.Args.WriteString(argsRaw)
						aggregatedParts = append(aggregatedParts, p)
					}
				}
			}
		}
	}

	if modelVersion != "" {
		out, _ = sjson.SetBytes(out, "modelVersion", modelVersion)
	}
	if responseID != "" {
		out, _ = sjson.SetBytes(out, "responseId", responseID)
	}
	if finishReason != "" {
		out, _ = sjson.SetBytes(out, "candidates.0.finishReason", finishReason)
	}
	if groundingMetadataRaw != "" {
		out, _ = sjson.SetRawBytes(out, "candidates.0.groundingMetadata", []byte(groundingMetadataRaw))
	}
	if usageMetadataRaw != "" {
		out, _ = sjson.SetRawBytes(out, "usageMetadata", []byte(usageMetadataRaw))
	}

	for _, part := range aggregatedParts {
		switch part.Kind {
		case "text":
			partJSON := []byte(`{"text":""}`)
			partJSON, _ = sjson.SetBytes(partJSON, "text", part.Text.String())
			if part.Thought {
				partJSON, _ = sjson.SetBytes(partJSON, "thought", true)
			}
			out, _ = sjson.SetRawBytes(out, "candidates.0.content.parts.-1", partJSON)
		case "functionCall":
			partJSON := []byte(`{"functionCall":{"name":"","args":{}}}`)
			if part.Name != "" {
				partJSON, _ = sjson.SetBytes(partJSON, "functionCall.name", part.Name)
			}
			partJSON, _ = sjson.SetRawBytes(partJSON, "functionCall.args", normalizeGeminiFunctionArgs(part.Args.String()))
			out, _ = sjson.SetRawBytes(out, "candidates.0.content.parts.-1", partJSON)
		}
	}

	return out
}

func geminiChunkItems(rawJSON []byte) []gjson.Result {
	root := gjson.ParseBytes(rawJSON)
	if root.IsArray() {
		return root.Array()
	}
	if root.Exists() {
		return []gjson.Result{root}
	}
	return nil
}

func normalizeGeminiFunctionArgs(raw string) []byte {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return []byte(`{}`)
	}
	if gjson.Valid(trimmed) && gjson.Parse(trimmed).IsObject() {
		return []byte(trimmed)
	}
	fixed := util.FixJSON(trimmed)
	if gjson.Valid(fixed) && gjson.Parse(fixed).IsObject() {
		return []byte(fixed)
	}
	return []byte(`{}`)
}

func hasClaudeWebSearchRequest(rawJSON []byte) bool {
	tools := gjson.GetBytes(rawJSON, "tools")
	if !tools.IsArray() {
		return false
	}
	for _, tool := range tools.Array() {
		if strings.HasPrefix(tool.Get("type").String(), "web_search_") {
			return true
		}
	}
	return false
}

func searchEnabledForGeminiResponse(originalRequestRawJSON, requestRawJSON []byte) bool {
	// Prefer the translated Gemini request when available because it reflects the final tool
	// set after Claude tool_choice filtering. Fall back to the original Claude request only
	// when the translated payload is unavailable.
	if hasGeminiGoogleSearchRequest(requestRawJSON) {
		return true
	}
	if len(requestRawJSON) == 0 {
		return hasClaudeWebSearchRequest(originalRequestRawJSON)
	}
	return false
}

func hasGeminiGoogleSearchRequest(rawJSON []byte) bool {
	tools := gjson.GetBytes(rawJSON, "tools")
	if !tools.IsArray() {
		return false
	}
	for _, tool := range tools.Array() {
		if tool.Get("googleSearch").Exists() || tool.Get("google_search").Exists() {
			return true
		}
	}
	return false
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func ClaudeTokenCount(ctx context.Context, count int64) []byte {
	return translatorcommon.ClaudeInputTokensJSON(count)
}
