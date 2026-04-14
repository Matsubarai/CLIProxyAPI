package responses

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertOpenAIResponsesRequestToGemini_WebSearchAddsGoogleSearchTool(t *testing.T) {
	input := []byte(`{
		"model":"gpt-5",
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"latest headline"}]}],
		"tools":[
			{"type":"function","name":"lookup_headline","parameters":{"type":"object","properties":{}}},
			{"type":"web_search_preview_2025_03_11","search_context_size":"high"}
		]
	}`)

	output := ConvertOpenAIResponsesRequestToGemini("gemini-2.5-flash", input, false)

	if got := gjson.GetBytes(output, "tools.#").Int(); got != 2 {
		t.Fatalf("tools count = %d, want 2: %s", got, string(output))
	}
	if !gjson.GetBytes(output, "tools.1.googleSearch").Exists() {
		t.Fatalf("expected googleSearch tool: %s", string(output))
	}
	if gjson.GetBytes(output, "tools.1.googleSearch.search_context_size").Exists() {
		t.Fatalf("did not expect search_context_size passthrough: %s", string(output))
	}
}

func TestConvertOpenAIResponsesRequestToGemini_ToolChoiceNoneSkipsGoogleSearch(t *testing.T) {
	input := []byte(`{
		"model":"gpt-5",
		"input":"latest headline",
		"tools":[
			{"type":"function","name":"lookup_headline","parameters":{"type":"object","properties":{}}},
			{"type":"web_search"}
		],
		"tool_choice":"none"
	}`)

	output := ConvertOpenAIResponsesRequestToGemini("gemini-2.5-flash", input, false)

	if got := gjson.GetBytes(output, "tools.#").Int(); got != 1 {
		t.Fatalf("tools count = %d, want 1: %s", got, string(output))
	}
	if gjson.GetBytes(output, "tools.0.googleSearch").Exists() {
		t.Fatalf("did not expect googleSearch tool when tool_choice=none: %s", string(output))
	}
	if got := gjson.GetBytes(output, "toolConfig.functionCallingConfig.mode").String(); got != "NONE" {
		t.Fatalf("functionCallingConfig.mode = %q, want %q: %s", got, "NONE", string(output))
	}
}

func TestConvertOpenAIResponsesRequestToGemini_ForcedWebSearchDisablesFunctionCalling(t *testing.T) {
	input := []byte(`{
		"model":"gpt-5",
		"input":"latest headline",
		"tools":[
			{"type":"function","name":"lookup_headline","parameters":{"type":"object","properties":{}}},
			{"type":"web_search_preview"}
		],
		"tool_choice":{"type":"web_search_preview"}
	}`)

	output := ConvertOpenAIResponsesRequestToGemini("gemini-2.5-flash", input, false)

	if !gjson.GetBytes(output, "tools.1.googleSearch").Exists() {
		t.Fatalf("expected googleSearch tool when web_search is forced: %s", string(output))
	}
	if got := gjson.GetBytes(output, "toolConfig.functionCallingConfig.mode").String(); got != "NONE" {
		t.Fatalf("functionCallingConfig.mode = %q, want %q: %s", got, "NONE", string(output))
	}
}

func TestConvertOpenAIResponsesRequestToGemini_MapsDeveloperMessagesToSystemInstruction(t *testing.T) {
	input := []byte(`{
		"model":"gpt-5",
		"instructions":"baseline policy",
		"input":[
			{"type":"message","role":"developer","content":[{"type":"input_text","text":"reply tersely"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}
		]
	}`)

	output := ConvertOpenAIResponsesRequestToGemini("gemini-2.5-flash", input, false)

	if got := gjson.GetBytes(output, "systemInstruction.parts.0.text").String(); got != "baseline policy" {
		t.Fatalf("systemInstruction.parts[0].text = %q, want %q: %s", got, "baseline policy", string(output))
	}
	if got := gjson.GetBytes(output, "systemInstruction.parts.1.text").String(); got != "reply tersely" {
		t.Fatalf("systemInstruction.parts[1].text = %q, want %q: %s", got, "reply tersely", string(output))
	}
	if got := gjson.GetBytes(output, "contents.#").Int(); got != 1 {
		t.Fatalf("contents count = %d, want 1: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "contents.0.role").String(); got != "user" {
		t.Fatalf("contents[0].role = %q, want %q: %s", got, "user", string(output))
	}
	if got := gjson.GetBytes(output, "contents.0.parts.0.text").String(); got != "hi" {
		t.Fatalf("contents[0].parts[0].text = %q, want %q: %s", got, "hi", string(output))
	}
	for _, content := range gjson.GetBytes(output, "contents").Array() {
		if got := content.Get("role").String(); got == "developer" {
			t.Fatalf("did not expect Gemini contents role=developer: %s", string(output))
		}
	}
}
