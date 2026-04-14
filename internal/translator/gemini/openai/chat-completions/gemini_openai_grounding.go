package chat_completions

import (
	"sort"
	"unicode/utf8"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type openAIGroundedSupport struct {
	Start     int
	End       int
	CitedText string
	Citations [][]byte
}

func buildOpenAIAnnotationsFromGroundingMetadata(text string, groundingMetadata gjson.Result) [][]byte {
	supports := resolveOpenAIGroundedSupports(text, openAIGroundedSupportsFromMetadata(groundingMetadata))
	if len(supports) == 0 {
		return nil
	}

	annotations := make([][]byte, 0)
	for _, support := range supports {
		if support.End <= support.Start {
			continue
		}
		for _, citation := range support.Citations {
			annotation := []byte(`{"type":"url_citation","url_citation":{"start_index":0,"end_index":0}}`)
			annotation, _ = sjson.SetBytes(annotation, "url_citation.start_index", support.Start)
			annotation, _ = sjson.SetBytes(annotation, "url_citation.end_index", support.End)
			if url := gjson.GetBytes(citation, "url"); url.Exists() {
				annotation, _ = sjson.SetBytes(annotation, "url_citation.url", url.String())
			}
			if title := gjson.GetBytes(citation, "title"); title.Exists() {
				annotation, _ = sjson.SetBytes(annotation, "url_citation.title", title.String())
			}
			annotations = append(annotations, annotation)
		}
	}

	return annotations
}

func openAIGroundedSupportsFromMetadata(groundingMetadata gjson.Result) []openAIGroundedSupport {
	supportsResult := groundingMetadata.Get("groundingSupports")
	if !supportsResult.IsArray() {
		return nil
	}

	chunks := groundingMetadata.Get("groundingChunks").Array()
	supports := make([]openAIGroundedSupport, 0, len(supportsResult.Array()))
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

			uri := web.Get("uri").String()
			title := web.Get("title").String()
			if uri == "" && title == "" {
				continue
			}

			citation := []byte(`{}`)
			if uri != "" {
				citation, _ = sjson.SetBytes(citation, "url", uri)
			}
			if title != "" {
				citation, _ = sjson.SetBytes(citation, "title", title)
			}
			citations = append(citations, citation)
		}

		if len(citations) == 0 {
			continue
		}

		supports = append(supports, openAIGroundedSupport{
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
		cursor = maxGroundingInt(cursor, supports[i].End)
	}

	return supports
}

func resolveOpenAIGroundedSupports(text string, supports []openAIGroundedSupport) []openAIGroundedSupport {
	if len(supports) == 0 {
		return nil
	}

	textLen := len([]rune(text))
	resolved := make([]openAIGroundedSupport, 0, len(supports))
	for _, support := range supports {
		start, end := resolveOpenAIGroundedSupportRange(text, support.Start, support.End, support.CitedText)
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
		cursor = maxGroundingInt(cursor, resolved[i].End)
	}

	return resolved
}

func resolveOpenAIGroundedSupportRange(text string, start, end int, citedText string) (int, int) {
	runes := []rune(text)
	runeLen := len(runes)

	runeStart := clampGroundingInt(start, 0, runeLen)
	runeEnd := clampGroundingInt(end, runeStart, runeLen)
	runeSlice := string(runes[runeStart:runeEnd])

	byteStart, byteEnd := normalizeGroundingByteOffsets([]byte(text), start, end)
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

func normalizeGroundingByteOffsets(text []byte, start, end int) (int, int) {
	start = clampGroundingInt(start, 0, len(text))
	end = clampGroundingInt(end, start, len(text))

	for start > 0 && start < len(text) && !utf8.RuneStart(text[start]) {
		start--
	}
	for end > start && end < len(text) && !utf8.RuneStart(text[end]) {
		end++
	}
	return start, end
}

func clampGroundingInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func maxGroundingInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
