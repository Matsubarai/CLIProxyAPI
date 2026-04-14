package claude

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertClaudeRequestToGemini_ToolChoice_SpecificTool(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "hi"}
				]
			}
		],
		"tools": [
			{
				"name": "json",
				"description": "A JSON tool",
				"input_schema": {
					"type": "object",
					"properties": {}
				}
			}
		],
		"tool_choice": {"type": "tool", "name": "json"}
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)

	if got := gjson.GetBytes(output, "toolConfig.functionCallingConfig.mode").String(); got != "ANY" {
		t.Fatalf("Expected toolConfig.functionCallingConfig.mode 'ANY', got '%s'", got)
	}
	allowed := gjson.GetBytes(output, "toolConfig.functionCallingConfig.allowedFunctionNames").Array()
	if len(allowed) != 1 || allowed[0].String() != "json" {
		t.Fatalf("Expected allowedFunctionNames ['json'], got %s", gjson.GetBytes(output, "toolConfig.functionCallingConfig.allowedFunctionNames").Raw)
	}
}

func TestConvertClaudeRequestToGemini_ImageContent(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "describe this image"},
					{
						"type": "image",
						"source": {
							"type": "base64",
							"media_type": "image/png",
							"data": "aGVsbG8="
						}
					}
				]
			}
		]
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)

	parts := gjson.GetBytes(output, "contents.0.parts").Array()
	if len(parts) != 2 {
		t.Fatalf("Expected 2 parts, got %d", len(parts))
	}
	if got := parts[0].Get("text").String(); got != "describe this image" {
		t.Fatalf("Expected first part text 'describe this image', got '%s'", got)
	}
	if got := parts[1].Get("inline_data.mime_type").String(); got != "image/png" {
		t.Fatalf("Expected image mime type 'image/png', got '%s'", got)
	}
	if got := parts[1].Get("inline_data.data").String(); got != "aGVsbG8=" {
		t.Fatalf("Expected image data 'aGVsbG8=', got '%s'", got)
	}
}

func TestConvertClaudeRequestToGemini_WebSearchToolMappedToGoogleSearch(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "What is the weather in NYC?"}
				]
			}
		],
		"tools": [
			{
				"type": "web_search_20250305",
				"name": "web_search",
				"max_uses": 5,
				"allowed_domains": ["weather.com"]
			},
			{
				"name": "json",
				"description": "A JSON tool",
				"input_schema": {
					"type": "object",
					"properties": {}
				}
			}
		],
		"tool_choice": {"type": "auto"}
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)
	tools := gjson.GetBytes(output, "tools").Array()
	if len(tools) != 2 {
		t.Fatalf("Expected 2 Gemini tools, got %d: %s", len(tools), string(output))
	}

	foundGoogleSearch := false
	foundFunctionDecl := false
	for _, tool := range tools {
		if tool.Get("googleSearch").Exists() {
			foundGoogleSearch = true
		}
		if tool.Get("functionDeclarations.0.name").String() == "json" {
			foundFunctionDecl = true
		}
	}
	if !foundGoogleSearch {
		t.Fatalf("Expected googleSearch tool in output: %s", string(output))
	}
	if !foundFunctionDecl {
		t.Fatalf("Expected function declaration for custom tool in output: %s", string(output))
	}
	if got := gjson.GetBytes(output, "toolConfig.functionCallingConfig.mode").String(); got != "AUTO" {
		t.Fatalf("Expected function calling mode AUTO, got %q: %s", got, string(output))
	}
}

func TestConvertClaudeRequestToGemini_WebSearchBlockedDomainsMappedToExcludeDomains(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "Find recent AI news."}
				]
			}
		],
		"tools": [
			{
				"type": "web_search_20250305",
				"name": "web_search",
				"blocked_domains": ["example.com", "news.example.com"]
			}
		],
		"tool_choice": {"type": "auto"}
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)
	tools := gjson.GetBytes(output, "tools").Array()
	if len(tools) != 1 {
		t.Fatalf("Expected only googleSearch tool, got %d: %s", len(tools), string(output))
	}
	if !gjson.GetBytes(output, "tools.0.googleSearch").Exists() {
		t.Fatalf("Expected googleSearch tool in output: %s", string(output))
	}

	excludeDomains := gjson.GetBytes(output, "tools.0.googleSearch.exclude_domains").Array()
	if len(excludeDomains) != 2 || excludeDomains[0].String() != "example.com" || excludeDomains[1].String() != "news.example.com" {
		t.Fatalf("Expected exclude_domains to mirror blocked_domains, got %s", gjson.GetBytes(output, "tools.0.googleSearch.exclude_domains").Raw)
	}
	if gjson.GetBytes(output, "tools.0.googleSearch.blocked_domains").Exists() {
		t.Fatalf("Did not expect blocked_domains to remain in googleSearch: %s", string(output))
	}
}

func TestConvertClaudeRequestToGemini_CodeExecutionToolMappedToGeminiCodeExecution(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "Calculate fibonacci(20)."}
				]
			}
		],
		"tools": [
			{
				"type": "code_execution_20250825",
				"name": "code_execution"
			},
			{
				"name": "json",
				"description": "A JSON tool",
				"input_schema": {
					"type": "object",
					"properties": {}
				}
			}
		],
		"tool_choice": {"type": "auto"}
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)
	tools := gjson.GetBytes(output, "tools").Array()
	if len(tools) != 2 {
		t.Fatalf("Expected 2 Gemini tools, got %d: %s", len(tools), string(output))
	}

	foundCodeExecution := false
	foundFunctionDecl := false
	for _, tool := range tools {
		if tool.Get("codeExecution").Exists() {
			foundCodeExecution = true
		}
		if tool.Get("functionDeclarations.0.name").String() == "json" {
			foundFunctionDecl = true
		}
	}
	if !foundCodeExecution {
		t.Fatalf("Expected codeExecution tool in output: %s", string(output))
	}
	if !foundFunctionDecl {
		t.Fatalf("Expected function declaration for custom tool in output: %s", string(output))
	}
	if got := gjson.GetBytes(output, "toolConfig.functionCallingConfig.mode").String(); got != "AUTO" {
		t.Fatalf("Expected function calling mode AUTO, got %q: %s", got, string(output))
	}
}

func TestConvertClaudeRequestToGemini_ToolChoiceSpecificFunctionOmitsGoogleSearch(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "hi"}
				]
			}
		],
		"tools": [
			{
				"type": "web_search_20260209",
				"name": "web_search"
			},
			{
				"name": "json",
				"description": "A JSON tool",
				"input_schema": {
					"type": "object",
					"properties": {}
				}
			}
		],
		"tool_choice": {"type": "tool", "name": "json"}
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)
	tools := gjson.GetBytes(output, "tools").Array()
	if len(tools) != 1 {
		t.Fatalf("Expected only function declarations tool, got %d: %s", len(tools), string(output))
	}
	if gjson.GetBytes(output, "tools.0.googleSearch").Exists() {
		t.Fatalf("Did not expect googleSearch tool when a custom function is forced: %s", string(output))
	}
	if got := gjson.GetBytes(output, "toolConfig.functionCallingConfig.mode").String(); got != "ANY" {
		t.Fatalf("Expected function calling mode ANY, got %q: %s", got, string(output))
	}
	allowed := gjson.GetBytes(output, "toolConfig.functionCallingConfig.allowedFunctionNames").Array()
	if len(allowed) != 1 || allowed[0].String() != "json" {
		t.Fatalf("Expected allowedFunctionNames ['json'], got %s", gjson.GetBytes(output, "toolConfig.functionCallingConfig.allowedFunctionNames").Raw)
	}
}

func TestConvertClaudeRequestToGemini_ToolChoiceSpecificFunctionOmitsCodeExecution(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "hi"}
				]
			}
		],
		"tools": [
			{
				"type": "code_execution_20250825",
				"name": "code_execution"
			},
			{
				"name": "json",
				"description": "A JSON tool",
				"input_schema": {
					"type": "object",
					"properties": {}
				}
			}
		],
		"tool_choice": {"type": "tool", "name": "json"}
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)
	tools := gjson.GetBytes(output, "tools").Array()
	if len(tools) != 1 {
		t.Fatalf("Expected only function declarations tool, got %d: %s", len(tools), string(output))
	}
	if gjson.GetBytes(output, "tools.0.codeExecution").Exists() {
		t.Fatalf("Did not expect codeExecution tool when a custom function is forced: %s", string(output))
	}
	if got := gjson.GetBytes(output, "toolConfig.functionCallingConfig.mode").String(); got != "ANY" {
		t.Fatalf("Expected function calling mode ANY, got %q: %s", got, string(output))
	}
	allowed := gjson.GetBytes(output, "toolConfig.functionCallingConfig.allowedFunctionNames").Array()
	if len(allowed) != 1 || allowed[0].String() != "json" {
		t.Fatalf("Expected allowedFunctionNames ['json'], got %s", gjson.GetBytes(output, "toolConfig.functionCallingConfig.allowedFunctionNames").Raw)
	}
}

func TestConvertClaudeRequestToGemini_ToolChoiceSpecificWebSearchDisablesFunctionCalling(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "search this"}
				]
			}
		],
		"tools": [
			{
				"type": "web_search_20250305",
				"name": "web_search"
			},
			{
				"name": "json",
				"description": "A JSON tool",
				"input_schema": {
					"type": "object",
					"properties": {}
				}
			}
		],
		"tool_choice": {"type": "tool", "name": "web_search"}
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)
	tools := gjson.GetBytes(output, "tools").Array()
	if len(tools) != 2 {
		t.Fatalf("Expected googleSearch plus function declarations, got %d: %s", len(tools), string(output))
	}
	foundGoogleSearch := false
	foundFunctionDecl := false
	for _, tool := range tools {
		if tool.Get("googleSearch").Exists() {
			foundGoogleSearch = true
		}
		if tool.Get("functionDeclarations.0.name").String() == "json" {
			foundFunctionDecl = true
		}
	}
	if !foundGoogleSearch {
		t.Fatalf("Expected googleSearch tool when web_search is forced: %s", string(output))
	}
	if !foundFunctionDecl {
		t.Fatalf("Expected function declarations to remain when web_search is forced: %s", string(output))
	}
	if got := gjson.GetBytes(output, "toolConfig.functionCallingConfig.mode").String(); got != "NONE" {
		t.Fatalf("Expected function calling mode NONE when web_search is forced, got %q: %s", got, string(output))
	}
	if gjson.GetBytes(output, "toolConfig.functionCallingConfig.allowedFunctionNames").Exists() {
		t.Fatalf("Did not expect allowedFunctionNames when web_search is forced: %s", string(output))
	}
}

func TestConvertClaudeRequestToGemini_ToolChoiceSpecificCodeExecutionDisablesFunctionCalling(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "calculate this"}
				]
			}
		],
		"tools": [
			{
				"type": "code_execution_20250825",
				"name": "code_execution"
			},
			{
				"name": "json",
				"description": "A JSON tool",
				"input_schema": {
					"type": "object",
					"properties": {}
				}
			}
		],
		"tool_choice": {"type": "tool", "name": "code_execution"}
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)
	tools := gjson.GetBytes(output, "tools").Array()
	if len(tools) != 2 {
		t.Fatalf("Expected codeExecution plus function declarations, got %d: %s", len(tools), string(output))
	}
	foundCodeExecution := false
	foundFunctionDecl := false
	for _, tool := range tools {
		if tool.Get("codeExecution").Exists() {
			foundCodeExecution = true
		}
		if tool.Get("functionDeclarations.0.name").String() == "json" {
			foundFunctionDecl = true
		}
	}
	if !foundCodeExecution {
		t.Fatalf("Expected codeExecution tool when code_execution is forced: %s", string(output))
	}
	if !foundFunctionDecl {
		t.Fatalf("Expected function declarations to remain when code_execution is forced: %s", string(output))
	}
	if got := gjson.GetBytes(output, "toolConfig.functionCallingConfig.mode").String(); got != "NONE" {
		t.Fatalf("Expected function calling mode NONE when code_execution is forced, got %q: %s", got, string(output))
	}
	if gjson.GetBytes(output, "toolConfig.functionCallingConfig.allowedFunctionNames").Exists() {
		t.Fatalf("Did not expect allowedFunctionNames when code_execution is forced: %s", string(output))
	}
}

func TestConvertClaudeRequestToGemini_ToolChoiceNoneOmitsGoogleSearch(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "hi"}
				]
			}
		],
		"tools": [
			{
				"type": "web_search_20250305",
				"name": "web_search"
			},
			{
				"name": "json",
				"description": "A JSON tool",
				"input_schema": {
					"type": "object",
					"properties": {}
				}
			}
		],
		"tool_choice": {"type": "none"}
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)
	tools := gjson.GetBytes(output, "tools").Array()
	if len(tools) != 1 {
		t.Fatalf("Expected only function declarations tool, got %d: %s", len(tools), string(output))
	}
	if gjson.GetBytes(output, "tools.0.googleSearch").Exists() {
		t.Fatalf("Did not expect googleSearch tool when tool_choice is none: %s", string(output))
	}
	if got := gjson.GetBytes(output, "toolConfig.functionCallingConfig.mode").String(); got != "NONE" {
		t.Fatalf("Expected function calling mode NONE, got %q: %s", got, string(output))
	}
}

func TestConvertClaudeRequestToGemini_ToolChoiceNoneOmitsCodeExecution(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "hi"}
				]
			}
		],
		"tools": [
			{
				"type": "code_execution_20250825",
				"name": "code_execution"
			},
			{
				"name": "json",
				"description": "A JSON tool",
				"input_schema": {
					"type": "object",
					"properties": {}
				}
			}
		],
		"tool_choice": {"type": "none"}
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)
	tools := gjson.GetBytes(output, "tools").Array()
	if len(tools) != 1 {
		t.Fatalf("Expected only function declarations tool, got %d: %s", len(tools), string(output))
	}
	if gjson.GetBytes(output, "tools.0.codeExecution").Exists() {
		t.Fatalf("Did not expect codeExecution tool when tool_choice is none: %s", string(output))
	}
	if got := gjson.GetBytes(output, "toolConfig.functionCallingConfig.mode").String(); got != "NONE" {
		t.Fatalf("Expected function calling mode NONE, got %q: %s", got, string(output))
	}
}

func TestConvertClaudeRequestToGemini_DropsReplayOnlyWebSearchBlocks(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3-flash-preview",
		"messages": [
			{
				"role": "assistant",
				"content": [
					{"type": "text", "text": "I'll search for that."},
					{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "latest weather nyc"}},
					{"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [{"type": "web_search_result", "url": "https://example.com", "title": "Example", "encrypted_content": "abc"}]},
					{"type": "text", "text": "It is sunny.", "citations": [{"type": "web_search_result_location", "url": "https://example.com", "title": "Example", "encrypted_index": "enc"}]}
				]
			}
		],
		"tools": [
			{
				"type": "web_search_20250305",
				"name": "web_search"
			}
		]
	}`)

	output := ConvertClaudeRequestToGemini("gemini-3-flash-preview", inputJSON, false)
	parts := gjson.GetBytes(output, "contents.0.parts").Array()
	if len(parts) != 2 {
		t.Fatalf("Expected only text parts to survive replay filtering, got %d: %s", len(parts), string(output))
	}
	if got := parts[0].Get("text").String(); got != "I'll search for that." {
		t.Fatalf("Unexpected first text part %q", got)
	}
	if got := parts[1].Get("text").String(); got != "It is sunny." {
		t.Fatalf("Unexpected second text part %q", got)
	}
	if strings.Contains(string(output), "srvtoolu_1") {
		t.Fatalf("Replay-only server tool blocks should not appear in Gemini request: %s", string(output))
	}
}
