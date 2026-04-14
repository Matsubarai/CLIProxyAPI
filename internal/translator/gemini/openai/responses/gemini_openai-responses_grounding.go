package responses

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type responsesGroundedSupport struct {
	Start     int
	End       int
	CitedText string
	Citations [][]byte
}

func buildResponsesAnnotationsFromGroundingMetadata(text string, groundingMetadata gjson.Result) [][]byte {
	supports := resolveResponsesGroundedSupports(text, responsesGroundedSupportsFromMetadata(groundingMetadata))
	if len(supports) == 0 {
		return nil
	}

	annotations := make([][]byte, 0)
	for _, support := range supports {
		if support.End <= support.Start {
			continue
		}
		for _, citation := range support.Citations {
			annotation := []byte(`{"type":"url_citation","start_index":0,"end_index":0,"title":"","url":""}`)
			annotation, _ = sjson.SetBytes(annotation, "start_index", support.Start)
			annotation, _ = sjson.SetBytes(annotation, "end_index", support.End)
			annotation, _ = sjson.SetBytes(annotation, "title", gjson.GetBytes(citation, "title").String())
			annotation, _ = sjson.SetBytes(annotation, "url", gjson.GetBytes(citation, "url").String())
			annotations = append(annotations, annotation)
		}
	}

	return annotations
}

func buildResponsesWebSearchCallItem(responseID string, originalRequestRawJSON, requestRawJSON []byte, groundingMetadata gjson.Result, includeSources bool) []byte {
	if !groundingMetadata.Exists() && !requestDeclaresResponsesWebSearch(originalRequestRawJSON, requestRawJSON) {
		return nil
	}
	if responseID == "" {
		return nil
	}

	item := []byte(`{"id":"","type":"web_search_call","status":"completed","action":{"type":"search"}}`)
	item, _ = sjson.SetBytes(item, "id", responsesWebSearchCallID(responseID))

	queries := resolveResponsesWebSearchQueries(originalRequestRawJSON, requestRawJSON, groundingMetadata)
	if len(queries) > 0 {
		item, _ = sjson.SetRawBytes(item, "action.queries", []byte(`[]`))
		for _, query := range queries {
			item, _ = sjson.SetBytes(item, "action.queries.-1", query)
		}
	}
	if query := firstResponsesWebSearchQuery(queries); query != "" {
		item, _ = sjson.SetBytes(item, "action.query", query)
	}

	if includeSources {
		sources := buildResponsesWebSearchSources(groundingMetadata)
		if len(sources) > 0 {
			item, _ = sjson.SetRawBytes(item, "action.sources", []byte(`[]`))
			for _, source := range sources {
				item, _ = sjson.SetRawBytes(item, "action.sources.-1", source)
			}
		}
	}

	return item
}

func buildResponsesWebSearchSources(groundingMetadata gjson.Result) [][]byte {
	chunks := groundingMetadata.Get("groundingChunks")
	if !chunks.IsArray() {
		return nil
	}

	sources := make([][]byte, 0)
	for _, chunk := range chunks.Array() {
		web := chunk.Get("web")
		if !web.Exists() {
			continue
		}
		url := strings.TrimSpace(web.Get("uri").String())
		if url == "" {
			continue
		}

		source := []byte(`{"type":"url","url":""}`)
		source, _ = sjson.SetBytes(source, "url", url)
		sources = append(sources, source)
	}

	return sources
}

func responsesWebSearchCallID(responseID string) string {
	return fmt.Sprintf("ws_%s_0", strings.TrimPrefix(responseID, "resp_"))
}

func resolveResponsesWebSearchQueries(originalRequestRawJSON, requestRawJSON []byte, groundingMetadata gjson.Result) []string {
	queries := groundingMetadata.Get("webSearchQueries")
	if queries.IsArray() {
		values := make([]string, 0, len(queries.Array()))
		for _, item := range queries.Array() {
			if query := strings.TrimSpace(item.String()); query != "" {
				values = append(values, query)
			}
		}
		if len(values) > 0 {
			return values
		}
	}

	reqJSON := pickRequestJSON(originalRequestRawJSON, requestRawJSON)
	if len(reqJSON) == 0 {
		return nil
	}

	if query := extractResponsesWebSearchQueryFromInput(unwrapRequestRoot(gjson.ParseBytes(reqJSON)).Get("input")); query != "" {
		return []string{query}
	}

	return nil
}

func firstResponsesWebSearchQuery(queries []string) string {
	if len(queries) == 0 {
		return ""
	}
	return queries[0]
}

func extractResponsesWebSearchQueryFromInput(input gjson.Result) string {
	if !input.Exists() {
		return ""
	}

	if input.Type == gjson.String {
		return strings.TrimSpace(input.String())
	}

	if input.IsArray() {
		fallback := ""
		for _, item := range input.Array() {
			if item.Type == gjson.String {
				if text := strings.TrimSpace(item.String()); text != "" {
					return text
				}
				continue
			}

			text := extractResponsesWebSearchQueryFromInputItem(item)
			if text == "" {
				continue
			}

			if item.Get("role").String() == "user" {
				return text
			}
			if fallback == "" {
				fallback = text
			}
		}
		return fallback
	}

	return extractResponsesWebSearchQueryFromInputItem(input)
}

func extractResponsesWebSearchQueryFromInputItem(item gjson.Result) string {
	if !item.Exists() {
		return ""
	}

	if content := item.Get("content"); content.Exists() {
		return extractResponsesWebSearchQueryFromContent(content)
	}

	if text := item.Get("text"); text.Exists() {
		return strings.TrimSpace(text.String())
	}

	return ""
}

func extractResponsesWebSearchQueryFromContent(content gjson.Result) string {
	if !content.Exists() {
		return ""
	}

	if content.Type == gjson.String {
		return strings.TrimSpace(content.String())
	}

	if content.IsArray() {
		parts := make([]string, 0, len(content.Array()))
		for _, part := range content.Array() {
			if part.Type == gjson.String {
				if text := strings.TrimSpace(part.String()); text != "" {
					parts = append(parts, text)
				}
				continue
			}

			partType := part.Get("type").String()
			if partType != "" && !strings.HasSuffix(partType, "_text") && partType != "text" {
				continue
			}

			if text := strings.TrimSpace(part.Get("text").String()); text != "" {
				parts = append(parts, text)
			}
		}
		return strings.TrimSpace(strings.Join(parts, " "))
	}

	if text := content.Get("text"); text.Exists() {
		return strings.TrimSpace(text.String())
	}

	return ""
}

func responsesGroundedSupportsFromMetadata(groundingMetadata gjson.Result) []responsesGroundedSupport {
	supportsResult := groundingMetadata.Get("groundingSupports")
	if !supportsResult.IsArray() {
		return nil
	}

	chunks := groundingMetadata.Get("groundingChunks").Array()
	supports := make([]responsesGroundedSupport, 0, len(supportsResult.Array()))
	for _, support := range supportsResult.Array() {
		start := int(support.Get("segment.startIndex").Int())
		end := int(support.Get("segment.endIndex").Int())
		if end < start {
			continue
		}

		citedText := support.Get("segment.text").String()
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

			url := strings.TrimSpace(web.Get("uri").String())
			if url == "" {
				continue
			}
			title := strings.TrimSpace(web.Get("title").String())
			if title == "" {
				title = url
			}

			citation := []byte(`{"title":"","url":""}`)
			citation, _ = sjson.SetBytes(citation, "title", title)
			citation, _ = sjson.SetBytes(citation, "url", url)
			citations = append(citations, citation)
		}

		if len(citations) == 0 {
			continue
		}

		supports = append(supports, responsesGroundedSupport{
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
		cursor = maxResponsesGroundingInt(cursor, supports[i].End)
	}

	return supports
}

func resolveResponsesGroundedSupports(text string, supports []responsesGroundedSupport) []responsesGroundedSupport {
	if len(supports) == 0 {
		return nil
	}

	textLen := len([]rune(text))
	resolved := make([]responsesGroundedSupport, 0, len(supports))
	for _, support := range supports {
		start, end := resolveResponsesGroundedSupportRange(text, support.Start, support.End, support.CitedText)
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
		cursor = maxResponsesGroundingInt(cursor, resolved[i].End)
	}

	return resolved
}

func resolveResponsesGroundedSupportRange(text string, start, end int, citedText string) (int, int) {
	runes := []rune(text)
	runeLen := len(runes)

	runeStart := clampResponsesGroundingInt(start, 0, runeLen)
	runeEnd := clampResponsesGroundingInt(end, runeStart, runeLen)
	runeSlice := string(runes[runeStart:runeEnd])

	byteStart, byteEnd := normalizeResponsesGroundingByteOffsets([]byte(text), start, end)
	byteSlice := text[byteStart:byteEnd]
	byteRuneStart := utf8.RuneCountInString(text[:byteStart])
	byteRuneEnd := byteRuneStart + utf8.RuneCountInString(byteSlice)

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

func normalizeResponsesGroundingByteOffsets(text []byte, start, end int) (int, int) {
	start = clampResponsesGroundingInt(start, 0, len(text))
	end = clampResponsesGroundingInt(end, start, len(text))

	for start > 0 && start < len(text) && !utf8.RuneStart(text[start]) {
		start--
	}
	for end > start && end < len(text) && !utf8.RuneStart(text[end]) {
		end++
	}
	return start, end
}

func clampResponsesGroundingInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func maxResponsesGroundingInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
