package chat_completions

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertOpenAIRequestToGemini_WebSearchOptionsAddsGoogleSearchTool(t *testing.T) {
	input := []byte(`{
		"model":"gpt-4.1",
		"messages":[{"role":"user","content":"latest weather"}],
		"web_search_options":{
			"search_context_size":"high",
			"user_location":{"type":"approximate","city":"Shanghai"}
		}
	}`)

	output := ConvertOpenAIRequestToGemini("gemini-2.5-flash", input, false)

	if got := gjson.GetBytes(output, "tools.#").Int(); got != 1 {
		t.Fatalf("tools count = %d, want 1: %s", got, string(output))
	}
	if !gjson.GetBytes(output, "tools.0.googleSearch").Exists() {
		t.Fatalf("expected synthesized googleSearch tool: %s", string(output))
	}
	if gjson.GetBytes(output, "tools.0.googleSearch.search_context_size").Exists() {
		t.Fatalf("did not expect search_context_size to be forwarded: %s", string(output))
	}
	if gjson.GetBytes(output, "tools.0.googleSearch.user_location").Exists() {
		t.Fatalf("did not expect user_location to be forwarded: %s", string(output))
	}
}

func TestConvertOpenAIRequestToGemini_WebSearchOptionsDoesNotDuplicateGoogleSearchTool(t *testing.T) {
	input := []byte(`{
		"model":"gpt-4.1",
		"messages":[{"role":"user","content":"latest weather"}],
		"web_search_options":{"search_context_size":"low"},
		"tools":[
			{"type":"function","function":{"name":"lookup_weather","parameters":{"type":"object","properties":{}}}},
			{"googleSearch":{"exclude_domains":["example.com"]}}
		]
	}`)

	output := ConvertOpenAIRequestToGemini("gemini-2.5-flash", input, false)
	tools := gjson.GetBytes(output, "tools").Array()
	if len(tools) != 2 {
		t.Fatalf("tools count = %d, want 2: %s", len(tools), string(output))
	}

	var googleSearchCount int
	for _, tool := range tools {
		if tool.Get("googleSearch").Exists() {
			googleSearchCount++
		}
	}
	if googleSearchCount != 1 {
		t.Fatalf("googleSearch tool count = %d, want 1: %s", googleSearchCount, string(output))
	}
	if got := gjson.GetBytes(output, "tools.1.googleSearch.exclude_domains.0").String(); got != "example.com" {
		t.Fatalf("exclude_domains[0] = %q, want %q: %s", got, "example.com", string(output))
	}
}
