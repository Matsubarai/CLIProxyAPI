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
	// Gemini emits code execution as separate executableCode/codeExecutionResult parts, so the
	// stream translator keeps the last synthetic server tool ID to pair the result correctly.
	LastServerToolUseID   string
	CodeExecutionRequests int64

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
	// Raw preserves opaque Gemini native-tool parts that should not be merged like text/function args.
	Raw string
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
			IsGlAPIKey:            false,
			HasFirstResponse:      false,
			ResponseType:          0,
			ResponseIndex:         0,
			ToolNameMap:           util.ToolNameMapFromClaudeRequest(originalRequestRawJSON),
			SanitizedNameMap:      util.SanitizedToolNameMap(originalRequestRawJSON),
			SawToolCall:           false,
			LastServerToolUseID:   "",
			CodeExecutionRequests: 0,
			BufferedGeminiChunks:  nil,
		}
	}

	if shouldBufferEntireGeminiResponse(originalRequestRawJSON, requestRawJSON) {
		if !bytes.Equal(rawJSON, []byte("[DONE]")) {
			(*param).(*Params).BufferedGeminiChunks = append((*param).(*Params).BufferedGeminiChunks, append([]byte(nil), rawJSON...))
			return nil
		}

		// Grounded search metadata typically arrives only on the terminal Gemini chunk. Buffering
		// lets the final Claude SSE follow the official web_search ordering instead of patching it
		// in after text has already been streamed.
		finalResponse := aggregateGeminiBufferedChunks((*param).(*Params).BufferedGeminiChunks)
		(*param).(*Params).BufferedGeminiChunks = nil
		if len(finalResponse) == 0 {
			return nil
		}

		message := buildClaudeMessageFromGeminiResponse(originalRequestRawJSON, finalResponse)
		return [][]byte{renderClaudeMessageAsSSE(message)}
	}

	if chunkItems := geminiChunkItems(rawJSON); len(chunkItems) > 1 {
		outputs := make([][]byte, 0, len(chunkItems))
		for _, item := range chunkItems {
			outputs = append(outputs, convertGeminiResponseToClaudeImmediate([]byte(item.Raw), param)...)
		}
		return outputs
	}
	if chunkItems := geminiChunkItems(rawJSON); len(chunkItems) == 1 && bytes.TrimSpace(rawJSON)[0] == '[' {
		return convertGeminiResponseToClaudeImmediate([]byte(chunkItems[0].Raw), param)
	}
	return convertGeminiResponseToClaudeImmediate(rawJSON, param)
}

func convertGeminiResponseToClaudeImmediate(rawJSON []byte, param *any) [][]byte {
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
			executableCodeResult := geminiExecutableCodePart(partResult)
			codeExecutionResult := geminiCodeExecutionResultPart(partResult)

			// Handle text content (both regular content and thinking)
			if partTextResult.Exists() && partTextResult.String() != "" {
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
			} else if executableCodeResult.Exists() {
				if (*param).(*Params).ResponseType == 2 {
					// output = output + "event: content_block_delta\n"
					// output = output + fmt.Sprintf(`data: {"type":"content_block_delta","index":%d,"delta":{"type":"signature_delta","signature":null}}`, (*param).(*Params).ResponseIndex)
					// output = output + "\n\n\n"
				}

				if (*param).(*Params).ResponseType != 0 {
					appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))
					(*param).(*Params).ResponseIndex++
					(*param).(*Params).ResponseType = 0
				}

				serverToolUseID := util.SanitizeClaudeToolID(fmt.Sprintf("srvtoolu_%d", atomic.AddUint64(&serverToolUseIDCounter, 1)))
				(*param).(*Params).LastServerToolUseID = serverToolUseID

				start := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"server_tool_use","id":"","name":""}}`, (*param).(*Params).ResponseIndex))
				start, _ = sjson.SetBytes(start, "content_block.id", serverToolUseID)
				start, _ = sjson.SetBytes(start, "content_block.name", "code_execution")
				appendEvent("content_block_start", string(start))

				input := buildClaudeCodeExecutionInput(executableCodeResult)
				delta := []byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":""}}`, (*param).(*Params).ResponseIndex))
				delta, _ = sjson.SetBytes(delta, "delta.partial_json", string(input))
				appendEvent("content_block_delta", string(delta))
				appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))

				(*param).(*Params).ResponseIndex++
				(*param).(*Params).HasContent = true
			} else if codeExecutionResult.Exists() {
				if (*param).(*Params).ResponseType == 2 {
					// output = output + "event: content_block_delta\n"
					// output = output + fmt.Sprintf(`data: {"type":"content_block_delta","index":%d,"delta":{"type":"signature_delta","signature":null}}`, (*param).(*Params).ResponseIndex)
					// output = output + "\n\n\n"
				}

				if (*param).(*Params).ResponseType != 0 {
					appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))
					(*param).(*Params).ResponseIndex++
					(*param).(*Params).ResponseType = 0
				}

				serverToolUseID := (*param).(*Params).LastServerToolUseID
				if serverToolUseID == "" {
					serverToolUseID = util.SanitizeClaudeToolID(fmt.Sprintf("srvtoolu_%d", atomic.AddUint64(&serverToolUseIDCounter, 1)))
					(*param).(*Params).LastServerToolUseID = serverToolUseID
				}

				result := buildClaudeCodeExecutionResultContent(codeExecutionResult)
				start := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"code_execution_tool_result","tool_use_id":"","content":{}}}`, (*param).(*Params).ResponseIndex))
				start, _ = sjson.SetBytes(start, "content_block.tool_use_id", serverToolUseID)
				start, _ = sjson.SetRawBytes(start, "content_block.content", result)
				appendEvent("content_block_start", string(start))
				appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))

				(*param).(*Params).ResponseIndex++
				(*param).(*Params).CodeExecutionRequests++
				(*param).(*Params).HasContent = true
			}
		}
	}

	usageResult := gjson.GetBytes(rawJSON, "usageMetadata")
	if usageResult.Exists() && bytes.Contains(rawJSON, []byte(`"finishReason"`)) {
		if candidatesTokenCountResult := usageResult.Get("candidatesTokenCount"); candidatesTokenCountResult.Exists() {
			// Only send final events if we have actually output content
			if (*param).(*Params).HasContent {
				if (*param).(*Params).ResponseType != 0 {
					appendEvent("content_block_stop", fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, (*param).(*Params).ResponseIndex))
				}

				template := []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}`)
				if (*param).(*Params).SawToolCall {
					template = []byte(`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}`)
				} else if finish := gjson.GetBytes(rawJSON, "candidates.0.finishReason"); finish.Exists() && finish.String() == "MAX_TOKENS" {
					template = []byte(`{"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}`)
				}

				thoughtsTokenCount := usageResult.Get("thoughtsTokenCount").Int()
				template, _ = sjson.SetBytes(template, "usage.output_tokens", candidatesTokenCountResult.Int()+thoughtsTokenCount)
				template, _ = sjson.SetBytes(template, "usage.input_tokens", usageResult.Get("promptTokenCount").Int())
				if (*param).(*Params).CodeExecutionRequests > 0 {
					template, _ = sjson.SetBytes(template, "usage.server_tool_use.code_execution_requests", (*param).(*Params).CodeExecutionRequests)
				}

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
	contentBlocks, hasToolCall, codeExecutionRequests := buildClaudeContentFromGeminiParts(parts, groundingMetadata, toolNameMap, sanitizedNameMap)
	for _, block := range contentBlocks {
		out, _ = sjson.SetRawBytes(out, "content.-1", block)
	}
	if count := groundedWebSearchRequestCount(groundingMetadata); count > 0 {
		out, _ = sjson.SetBytes(out, "usage.server_tool_use.web_search_requests", count)
	}
	if codeExecutionRequests > 0 {
		out, _ = sjson.SetBytes(out, "usage.server_tool_use.code_execution_requests", codeExecutionRequests)
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

func buildClaudeContentFromGeminiParts(parts, groundingMetadata gjson.Result, toolNameMap, sanitizedNameMap map[string]string) ([][]byte, bool, int64) {
	contentBlocks := make([][]byte, 0)
	textBuilder := strings.Builder{}
	thinkingBuilder := strings.Builder{}
	toolIDCounter := 0
	hasToolCall := false
	codeExecutionRequests := int64(0)
	lastServerToolUseID := ""
	visibleText := concatenateVisibleGeminiText(parts)
	supports := resolvedGroundedSupports(visibleText, groundedSupportsFromMetadata(groundingMetadata))
	supportIdx := 0
	visibleOffset := 0

	flushText := func() {
		if textBuilder.Len() == 0 {
			return
		}
		for _, block := range buildClaudeTextBlocksWithGrounding(textBuilder.String(), supports, &supportIdx, &visibleOffset) {
			contentBlocks = append(contentBlocks, block)
		}
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

			if executableCode := geminiExecutableCodePart(part); executableCode.Exists() {
				flushThinking()
				flushText()

				lastServerToolUseID = util.SanitizeClaudeToolID(fmt.Sprintf("srvtoolu_%d", atomic.AddUint64(&serverToolUseIDCounter, 1)))
				serverToolUse := []byte(`{"type":"server_tool_use","id":"","name":"code_execution","input":{}}`)
				serverToolUse, _ = sjson.SetBytes(serverToolUse, "id", lastServerToolUseID)
				serverToolUse, _ = sjson.SetRawBytes(serverToolUse, "input", buildClaudeCodeExecutionInput(executableCode))
				contentBlocks = append(contentBlocks, serverToolUse)
				continue
			}

			if codeExecutionResult := geminiCodeExecutionResultPart(part); codeExecutionResult.Exists() {
				flushThinking()
				flushText()

				if lastServerToolUseID == "" {
					lastServerToolUseID = util.SanitizeClaudeToolID(fmt.Sprintf("srvtoolu_%d", atomic.AddUint64(&serverToolUseIDCounter, 1)))
				}
				toolResult := []byte(`{"type":"code_execution_tool_result","tool_use_id":"","content":{}}`)
				toolResult, _ = sjson.SetBytes(toolResult, "tool_use_id", lastServerToolUseID)
				toolResult, _ = sjson.SetRawBytes(toolResult, "content", buildClaudeCodeExecutionResultContent(codeExecutionResult))
				contentBlocks = append(contentBlocks, toolResult)
				codeExecutionRequests++
				continue
			}
		}
	}

	flushThinking()
	flushText()

	if groundingMetadata.Exists() {
		// Keep synthetic web_search blocks after the grounded text in the logical Claude message.
		// The streaming renderer can then reorder them into Claude's wire-level event sequence.
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
	}

	return contentBlocks, hasToolCall, codeExecutionRequests
}

func buildClaudeTextBlocksWithGrounding(text string, supports []groundedSupport, supportIdx, visibleOffset *int) [][]byte {
	if text == "" {
		return nil
	}

	runes := []rune(text)
	partStart := *visibleOffset
	partEnd := partStart + len(runes)
	blocks := make([][]byte, 0)
	appendPlain := func(s string) {
		if s == "" {
			return
		}
		block := []byte(`{"type":"text","text":""}`)
		block, _ = sjson.SetBytes(block, "text", s)
		blocks = append(blocks, block)
	}

	for *supportIdx < len(supports) && supports[*supportIdx].End <= partStart {
		(*supportIdx)++
	}

	if len(supports) == 0 || *supportIdx >= len(supports) || supports[*supportIdx].Start >= partEnd {
		appendPlain(text)
		*visibleOffset = partEnd
		return blocks
	}

	cursor := 0
	scanIdx := *supportIdx
	for scanIdx < len(supports) && supports[scanIdx].Start < partEnd {
		support := supports[scanIdx]
		localStart := maxInt(support.Start, partStart) - partStart
		localEnd := minInt(support.End, partEnd) - partStart
		if localStart > cursor {
			appendPlain(string(runes[cursor:localStart]))
		}
		if localEnd > localStart {
			block := []byte(`{"type":"text","text":"","citations":[]}`)
			block, _ = sjson.SetBytes(block, "text", string(runes[localStart:localEnd]))
			if len(support.Citations) == 0 {
				block, _ = sjson.DeleteBytes(block, "citations")
			} else {
				for _, citation := range support.Citations {
					block, _ = sjson.SetRawBytes(block, "citations.-1", citation)
				}
			}
			blocks = append(blocks, block)
		}
		cursor = maxInt(cursor, localEnd)
		if support.End <= partEnd {
			*supportIdx = scanIdx + 1
		} else {
			*supportIdx = scanIdx
		}
		scanIdx++
	}

	if cursor < len(runes) {
		appendPlain(string(runes[cursor:]))
	}
	*visibleOffset = partEnd
	return blocks
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

func renderClaudeMessageAsSSE(message []byte) []byte {
	if output := renderClaudeGroundedWebSearchMessageAsSSE(message); len(output) > 0 {
		return output
	}
	return renderClaudeMessageAsSSEGeneric(message)
}

func renderClaudeGroundedWebSearchMessageAsSSE(message []byte) []byte {
	blocks := collectClaudeRenderableBlocks(message)
	searchToolUsePos := -1
	searchToolResultPos := -1
	firstDeferredTextPos := -1

	for i, block := range blocks {
		blockType := block.Get("type").String()
		if searchToolUsePos < 0 && blockType == "server_tool_use" && block.Get("name").String() == "web_search" {
			searchToolUsePos = i
			break
		}
	}
	if searchToolUsePos <= 0 {
		return nil
	}
	for i := searchToolUsePos + 1; i < len(blocks); i++ {
		if blocks[i].Get("type").String() == "web_search_tool_result" {
			searchToolResultPos = i
			break
		}
	}
	if searchToolResultPos < 0 {
		return nil
	}
	for i := 0; i < searchToolUsePos; i++ {
		if blocks[i].Get("type").String() == "text" {
			firstDeferredTextPos = i
			break
		}
	}
	if firstDeferredTextPos < 0 {
		return nil
	}
	for i := firstDeferredTextPos; i < searchToolUsePos; i++ {
		if blocks[i].Get("type").String() != "text" {
			return nil
		}
	}

	// Claude's documented wire format opens the answer text block first, then emits the
	// web_search tool use/result, and only then streams the grounded text/citations.
	output := make([]byte, 0, len(message)+512)
	output = appendClaudeMessageStartAsSSE(output, message)

	nextIndex := 0
	for i := 0; i < firstDeferredTextPos; i++ {
		output = appendClaudeRenderableBlockAsSSE(output, nextIndex, blocks[i])
		nextIndex++
	}

	deferredTextIndex := nextIndex
	output = appendClaudeTextBlockStartAsSSE(output, deferredTextIndex)
	nextIndex++

	for i := searchToolUsePos; i <= searchToolResultPos; i++ {
		output = appendClaudeRenderableBlockAsSSE(output, nextIndex, blocks[i])
		nextIndex++
	}

	output = appendClaudeTextBlockDeltaAndStopAsSSE(output, deferredTextIndex, blocks[firstDeferredTextPos])

	for i := firstDeferredTextPos + 1; i < searchToolUsePos; i++ {
		output = appendClaudeRenderableBlockAsSSE(output, nextIndex, blocks[i])
		nextIndex++
	}
	for i := searchToolResultPos + 1; i < len(blocks); i++ {
		output = appendClaudeRenderableBlockAsSSE(output, nextIndex, blocks[i])
		nextIndex++
	}

	output = appendClaudeMessageEndAsSSE(output, message)
	return output
}

func renderClaudeMessageAsSSEGeneric(message []byte) []byte {
	output := make([]byte, 0, len(message)+512)
	output = appendClaudeMessageStartAsSSE(output, message)
	nextIndex := 0
	for _, block := range collectClaudeRenderableBlocks(message) {
		output = appendClaudeRenderableBlockAsSSE(output, nextIndex, block)
		nextIndex++
	}
	return appendClaudeMessageEndAsSSE(output, message)
}

func collectClaudeRenderableBlocks(message []byte) []gjson.Result {
	content := gjson.GetBytes(message, "content")
	if !content.IsArray() {
		return nil
	}
	blocks := make([]gjson.Result, 0, len(content.Array()))
	for _, block := range content.Array() {
		switch block.Get("type").String() {
		case "text":
			if block.Get("text").String() == "" && len(block.Get("citations").Array()) == 0 {
				continue
			}
		case "thinking":
			if block.Get("thinking").String() == "" {
				continue
			}
		}
		blocks = append(blocks, block)
	}
	return blocks
}

func appendClaudeMessageStartAsSSE(output, message []byte) []byte {
	messageStart := []byte(`{"type":"message_start","message":{"id":"","type":"message","role":"assistant","content":[],"model":"","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}`)
	messageStart, _ = sjson.SetBytes(messageStart, "message.id", gjson.GetBytes(message, "id").String())
	messageStart, _ = sjson.SetBytes(messageStart, "message.model", gjson.GetBytes(message, "model").String())
	return translatorcommon.AppendSSEEventBytes(output, "message_start", messageStart, 3)
}

func appendClaudeMessageEndAsSSE(output, message []byte) []byte {
	messageDelta := []byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}`)
	if stopReason := gjson.GetBytes(message, "stop_reason").String(); stopReason != "" {
		messageDelta, _ = sjson.SetBytes(messageDelta, "delta.stop_reason", stopReason)
	}
	if usage := gjson.GetBytes(message, "usage"); usage.Exists() {
		messageDelta, _ = sjson.SetRawBytes(messageDelta, "usage", []byte(usage.Raw))
	}
	output = translatorcommon.AppendSSEEventBytes(output, "message_delta", messageDelta, 3)
	return translatorcommon.AppendSSEEventString(output, "message_stop", `{"type":"message_stop"}`, 3)
}

func appendClaudeRenderableBlockAsSSE(output []byte, index int, block gjson.Result) []byte {
	switch block.Get("type").String() {
	case "text":
		output = appendClaudeTextBlockStartAsSSE(output, index)
		return appendClaudeTextBlockDeltaAndStopAsSSE(output, index, block)
	case "thinking":
		start := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"thinking","thinking":""}}`, index))
		output = translatorcommon.AppendSSEEventBytes(output, "content_block_start", start, 3)
		delta := []byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"thinking_delta","thinking":""}}`, index))
		delta, _ = sjson.SetBytes(delta, "delta.thinking", block.Get("thinking").String())
		output = translatorcommon.AppendSSEEventBytes(output, "content_block_delta", delta, 3)
	case "tool_use":
		start := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"tool_use","id":"","name":"","input":{}}}`, index))
		start, _ = sjson.SetBytes(start, "content_block.id", block.Get("id").String())
		start, _ = sjson.SetBytes(start, "content_block.name", block.Get("name").String())
		output = translatorcommon.AppendSSEEventBytes(output, "content_block_start", start, 3)
		if input := block.Get("input"); input.Exists() {
			delta := []byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":""}}`, index))
			delta, _ = sjson.SetBytes(delta, "delta.partial_json", input.Raw)
			output = translatorcommon.AppendSSEEventBytes(output, "content_block_delta", delta, 3)
		}
	case "server_tool_use":
		start := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"server_tool_use","id":"","name":""}}`, index))
		start, _ = sjson.SetBytes(start, "content_block.id", block.Get("id").String())
		start, _ = sjson.SetBytes(start, "content_block.name", block.Get("name").String())
		output = translatorcommon.AppendSSEEventBytes(output, "content_block_start", start, 3)
		if input := block.Get("input"); input.Exists() {
			delta := []byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"input_json_delta","partial_json":""}}`, index))
			delta, _ = sjson.SetBytes(delta, "delta.partial_json", input.Raw)
			output = translatorcommon.AppendSSEEventBytes(output, "content_block_delta", delta, 3)
		}
	case "web_search_tool_result":
		start := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"web_search_tool_result","tool_use_id":"","content":[]}}`, index))
		start, _ = sjson.SetBytes(start, "content_block.tool_use_id", block.Get("tool_use_id").String())
		start, _ = sjson.SetRawBytes(start, "content_block.content", []byte(block.Get("content").Raw))
		output = translatorcommon.AppendSSEEventBytes(output, "content_block_start", start, 3)
	case "code_execution_tool_result":
		start := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"code_execution_tool_result","tool_use_id":"","content":{}}}`, index))
		start, _ = sjson.SetBytes(start, "content_block.tool_use_id", block.Get("tool_use_id").String())
		start, _ = sjson.SetRawBytes(start, "content_block.content", []byte(block.Get("content").Raw))
		output = translatorcommon.AppendSSEEventBytes(output, "content_block_start", start, 3)
	default:
		return output
	}
	stop := []byte(fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, index))
	return translatorcommon.AppendSSEEventBytes(output, "content_block_stop", stop, 3)
}

func appendClaudeTextBlockStartAsSSE(output []byte, index int) []byte {
	start := []byte(fmt.Sprintf(`{"type":"content_block_start","index":%d,"content_block":{"type":"text","text":""}}`, index))
	return translatorcommon.AppendSSEEventBytes(output, "content_block_start", start, 3)
}

func appendClaudeTextBlockDeltaAndStopAsSSE(output []byte, index int, block gjson.Result) []byte {
	if text := block.Get("text").String(); text != "" {
		delta := []byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":""}}`, index))
		delta, _ = sjson.SetBytes(delta, "delta.text", text)
		output = translatorcommon.AppendSSEEventBytes(output, "content_block_delta", delta, 3)
	}
	for _, citation := range block.Get("citations").Array() {
		delta := []byte(fmt.Sprintf(`{"type":"content_block_delta","index":%d,"delta":{"type":"citations_delta","citation":{}}}`, index))
		delta, _ = sjson.SetRawBytes(delta, "delta.citation", []byte(citation.Raw))
		output = translatorcommon.AppendSSEEventBytes(output, "content_block_delta", delta, 3)
	}
	stop := []byte(fmt.Sprintf(`{"type":"content_block_stop","index":%d}`, index))
	return translatorcommon.AppendSSEEventBytes(output, "content_block_stop", stop, 3)
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
				if text := part.Get("text"); text.Exists() && text.String() != "" {
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
					continue
				}
				if geminiExecutableCodePart(part).Exists() {
					aggregatedParts = append(aggregatedParts, &aggregatedGeminiPart{Kind: "executableCode", Raw: part.Raw})
					continue
				}
				if geminiCodeExecutionResultPart(part).Exists() {
					aggregatedParts = append(aggregatedParts, &aggregatedGeminiPart{Kind: "codeExecutionResult", Raw: part.Raw})
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
		case "executableCode", "codeExecutionResult":
			out, _ = sjson.SetRawBytes(out, "candidates.0.content.parts.-1", []byte(part.Raw))
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

func geminiExecutableCodePart(part gjson.Result) gjson.Result {
	if executableCode := part.Get("executableCode"); executableCode.Exists() {
		return executableCode
	}
	return part.Get("executable_code")
}

func geminiCodeExecutionResultPart(part gjson.Result) gjson.Result {
	if codeExecutionResult := part.Get("codeExecutionResult"); codeExecutionResult.Exists() {
		return codeExecutionResult
	}
	return part.Get("code_execution_result")
}

func buildClaudeCodeExecutionInput(executableCode gjson.Result) []byte {
	input := []byte(`{}`)
	if code := executableCode.Get("code").String(); code != "" {
		input, _ = sjson.SetBytes(input, "code", code)
	}
	if language := executableCode.Get("language").String(); language != "" {
		input, _ = sjson.SetBytes(input, "language", language)
	}
	return input
}

func buildClaudeCodeExecutionResultContent(codeExecutionResult gjson.Result) []byte {
	outcome := codeExecutionResult.Get("outcome").String()
	output := codeExecutionResult.Get("output").String()

	content := []byte(`{"type":"code_execution_result","stdout":"","stderr":"","return_code":0}`)
	switch outcome {
	case "OUTCOME_OK":
		content, _ = sjson.SetBytes(content, "stdout", output)
		content, _ = sjson.SetBytes(content, "return_code", 0)
	case "OUTCOME_FAILED":
		content, _ = sjson.SetBytes(content, "stderr", output)
		content, _ = sjson.SetBytes(content, "return_code", 1)
	case "OUTCOME_DEADLINE_EXCEEDED":
		content, _ = sjson.SetBytes(content, "stderr", output)
		content, _ = sjson.SetBytes(content, "return_code", 124)
	default:
		content, _ = sjson.SetBytes(content, "stderr", output)
		content, _ = sjson.SetBytes(content, "return_code", 1)
	}
	return content
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

func shouldBufferEntireGeminiResponse(originalRequestRawJSON, requestRawJSON []byte) bool {
	// Any request that declares web_search needs the final grounded metadata before we can emit
	// a Claude-compatible stream with stable block ordering and citations.
	if len(requestRawJSON) > 0 {
		return hasGeminiGoogleSearchRequest(requestRawJSON)
	}
	return hasClaudeWebSearchRequest(originalRequestRawJSON)
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
