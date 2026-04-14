package chat_completions

import (
	"bytes"
	"context"
	"fmt"
	"testing"
	"unicode/utf8"

	"github.com/tidwall/gjson"
)

func TestConvertGeminiResponseToOpenAINonStream_GroundingMetadataAnnotations(t *testing.T) {
	rawJSON := []byte(`{
		"responseId":"resp_grounded",
		"modelVersion":"gemini-2.5-flash",
		"candidates":[{
			"index":0,
			"content":{"role":"model","parts":[{"text":"Alpha beta"}]},
			"finishReason":"STOP",
			"groundingMetadata":{
				"groundingChunks":[
					{"web":{"uri":"https://example.com/a","title":"Example A"}},
					{"web":{"uri":"https://example.com/b","title":"Example B"}}
				],
				"groundingSupports":[
					{"segment":{"startIndex":0,"endIndex":10,"text":"Alpha beta"},"groundingChunkIndices":[0,1]}
				]
			}
		}],
		"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":8,"totalTokenCount":20}
	}`)

	output := ConvertGeminiResponseToOpenAINonStream(context.Background(), "gemini-2.5-flash", nil, nil, rawJSON, nil)

	if got := gjson.GetBytes(output, "choices.0.message.content").String(); got != "Alpha beta" {
		t.Fatalf("message.content = %q, want %q: %s", got, "Alpha beta", string(output))
	}
	if got := gjson.GetBytes(output, "choices.0.message.annotations.#").Int(); got != 2 {
		t.Fatalf("annotations count = %d, want 2: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "choices.0.message.annotations.0.type").String(); got != "url_citation" {
		t.Fatalf("annotations.0.type = %q, want %q: %s", got, "url_citation", string(output))
	}
	if got := gjson.GetBytes(output, "choices.0.message.annotations.0.url_citation.start_index").Int(); got != 0 {
		t.Fatalf("annotations.0.start_index = %d, want 0: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "choices.0.message.annotations.0.url_citation.end_index").Int(); got != 10 {
		t.Fatalf("annotations.0.end_index = %d, want 10: %s", got, string(output))
	}
	if got := gjson.GetBytes(output, "choices.0.message.annotations.1.url_citation.url").String(); got != "https://example.com/b" {
		t.Fatalf("annotations.1.url = %q, want %q: %s", got, "https://example.com/b", string(output))
	}
}

func TestConvertGeminiResponseToOpenAINonStream_GroundingByteOffsets(t *testing.T) {
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
			"index":0,
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

	output := ConvertGeminiResponseToOpenAINonStream(context.Background(), "gemini-2.5-flash", nil, nil, rawJSON, nil)

	if got := gjson.GetBytes(output, "choices.0.message.annotations.0.url_citation.start_index").Int(); got != int64(wantStart) {
		t.Fatalf("start_index = %d, want %d: %s", got, wantStart, string(output))
	}
	if got := gjson.GetBytes(output, "choices.0.message.annotations.0.url_citation.end_index").Int(); got != int64(wantEnd) {
		t.Fatalf("end_index = %d, want %d: %s", got, wantEnd, string(output))
	}
}

func TestConvertGeminiResponseToOpenAI_StreamGroundingMetadataEmitsDeltaAnnotations(t *testing.T) {
	ctx := context.Background()
	var param any

	firstChunk := []byte(`{
		"responseId":"resp_stream_grounded",
		"modelVersion":"gemini-2.5-flash",
		"candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"Alpha beta"}]}}]
	}`)
	secondChunk := []byte(`{
		"responseId":"resp_stream_grounded",
		"modelVersion":"gemini-2.5-flash",
		"candidates":[{
			"index":0,
			"finishReason":"STOP",
			"groundingMetadata":{
				"groundingChunks":[
					{"web":{"uri":"https://example.com/source","title":"Example Source"}}
				],
				"groundingSupports":[
					{"segment":{"startIndex":0,"endIndex":10,"text":"Alpha beta"},"groundingChunkIndices":[0]}
				]
			}
		}]
	}`)

	firstOutput := ConvertGeminiResponseToOpenAI(ctx, "gemini-2.5-flash", nil, nil, firstChunk, &param)
	if len(firstOutput) != 1 {
		t.Fatalf("first output chunks = %d, want 1", len(firstOutput))
	}
	if got := gjson.GetBytes(firstOutput[0], "choices.0.delta.content").String(); got != "Alpha beta" {
		t.Fatalf("first delta.content = %q, want %q: %s", got, "Alpha beta", string(firstOutput[0]))
	}
	if gjson.GetBytes(firstOutput[0], "choices.0.delta.annotations").Exists() {
		t.Fatalf("did not expect annotations on first chunk: %s", string(firstOutput[0]))
	}

	secondOutput := ConvertGeminiResponseToOpenAI(ctx, "gemini-2.5-flash", nil, nil, secondChunk, &param)
	if len(secondOutput) != 1 {
		t.Fatalf("second output chunks = %d, want 1", len(secondOutput))
	}
	if got := gjson.GetBytes(secondOutput[0], "choices.0.delta.annotations.0.type").String(); got != "url_citation" {
		t.Fatalf("delta.annotations.0.type = %q, want %q: %s", got, "url_citation", string(secondOutput[0]))
	}
	if !gjson.GetBytes(secondOutput[0], "choices.0.delta").IsObject() {
		t.Fatalf("delta should be an object: %s", string(secondOutput[0]))
	}
	if gjson.GetBytes(secondOutput[0], "choices.0.delta.content").Exists() {
		t.Fatalf("did not expect delta.content on annotation-only chunk: %s", string(secondOutput[0]))
	}
	if got := gjson.GetBytes(secondOutput[0], "choices.0.delta.annotations.0.url_citation.start_index").Int(); got != 0 {
		t.Fatalf("delta.annotations.0.start_index = %d, want 0: %s", got, string(secondOutput[0]))
	}
	if got := gjson.GetBytes(secondOutput[0], "choices.0.delta.annotations.0.url_citation.end_index").Int(); got != 10 {
		t.Fatalf("delta.annotations.0.end_index = %d, want 10: %s", got, string(secondOutput[0]))
	}
	if got := gjson.GetBytes(secondOutput[0], "choices.0.finish_reason").String(); got != "stop" {
		t.Fatalf("finish_reason = %q, want %q: %s", got, "stop", string(secondOutput[0]))
	}
}

func TestConvertGeminiResponseToOpenAI_StreamRoleOnlyChunkKeepsEmptyDelta(t *testing.T) {
	ctx := context.Background()
	var param any

	rawJSON := []byte(`{
		"responseId":"resp_stream_role_only",
		"modelVersion":"gemini-2.5-flash",
		"candidates":[{"index":0,"content":{"role":"model"}}]
	}`)

	output := ConvertGeminiResponseToOpenAI(ctx, "gemini-2.5-flash", nil, nil, rawJSON, &param)
	if len(output) != 1 {
		t.Fatalf("output chunks = %d, want 1", len(output))
	}
	if !gjson.GetBytes(output[0], "choices.0.delta").IsObject() {
		t.Fatalf("delta should be an object: %s", string(output[0]))
	}
	if got := gjson.GetBytes(output[0], "choices.0.delta").Raw; got != "{}" {
		t.Fatalf("delta = %s, want {}: %s", got, string(output[0]))
	}
	if gjson.GetBytes(output[0], "choices.0.delta.content").Exists() {
		t.Fatalf("did not expect delta.content: %s", string(output[0]))
	}
	if gjson.GetBytes(output[0], "choices.0.finish_reason").Exists() {
		t.Fatalf("did not expect finish_reason: %s", string(output[0]))
	}
}
