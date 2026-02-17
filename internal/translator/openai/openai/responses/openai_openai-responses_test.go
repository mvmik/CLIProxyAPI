package responses

import (
	"context"
	"encoding/base64"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

// parseSSEEvent is a helper to parse SSE event chunks
func parseSSEEvent(t *testing.T, chunk string) (string, gjson.Result) {
	t.Helper()

	lines := strings.Split(chunk, "\n")
	if len(lines) < 2 {
		t.Fatalf("unexpected SSE chunk: %q", chunk)
	}

	event := strings.TrimSpace(strings.TrimPrefix(lines[0], "event:"))
	dataLine := strings.TrimSpace(strings.TrimPrefix(lines[1], "data:"))
	if !gjson.Valid(dataLine) {
		t.Fatalf("invalid SSE data JSON: %q", dataLine)
	}
	return event, gjson.Parse(dataLine)
}

// TestRequest_EncryptedContentToReasoningContent tests decoding encrypted_content to reasoning_content
func TestRequest_EncryptedContentToReasoningContent(t *testing.T) {
	reasoningText := "Let me think about this..."
	encryptedContent := base64.StdEncoding.EncodeToString([]byte(reasoningText))

	inputJSON := []byte(`{
		"model": "gpt-5",
		"input": [
			{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is 2+2?"}]},
			{"type": "reasoning", "encrypted_content": "` + encryptedContent + `"},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "The answer is 4."}]},
			{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "And 3+3?"}]}
		]
	}`)

	output := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("gpt-5", inputJSON, false)
	outputStr := string(output)

	// The assistant message is at index 1 (user -> assistant -> user, reasoning attaches to assistant)
	// Check that the assistant message has reasoning_content
	assistantReasoning := gjson.Get(outputStr, "messages.1.reasoning_content")
	if !assistantReasoning.Exists() {
		t.Fatalf("expected reasoning_content to exist on assistant message")
	}
	if assistantReasoning.String() != reasoningText {
		t.Errorf("expected reasoning_content %q, got %q", reasoningText, assistantReasoning.String())
	}

	// Verify the assistant content is preserved
	assistantContent := gjson.Get(outputStr, "messages.1.content.0.text")
	if assistantContent.String() != "The answer is 4." {
		t.Errorf("expected assistant content 'The answer is 4.', got %q", assistantContent.String())
	}
}

// TestRequest_MultipleReasoningItems tests accumulating multiple reasoning items
func TestRequest_MultipleReasoningItems(t *testing.T) {
	reasoning1 := "First thought..."
	reasoning2 := "Second thought..."
	encrypted1 := base64.StdEncoding.EncodeToString([]byte(reasoning1))
	encrypted2 := base64.StdEncoding.EncodeToString([]byte(reasoning2))

	inputJSON := []byte(`{
		"model": "gpt-5",
		"input": [
			{"type": "message", "role": "user", "content": "Hello"},
			{"type": "reasoning", "encrypted_content": "` + encrypted1 + `"},
			{"type": "reasoning", "encrypted_content": "` + encrypted2 + `"},
			{"type": "message", "role": "assistant", "content": "Hi!"}
		]
	}`)

	output := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("gpt-5", inputJSON, false)
	outputStr := string(output)

	// The assistant message is at index 1 (user -> assistant)
	// Multiple reasoning items should be concatenated
	assistantReasoning := gjson.Get(outputStr, "messages.1.reasoning_content")
	if !assistantReasoning.Exists() {
		t.Fatalf("expected reasoning_content to exist")
	}
	expected := reasoning1 + reasoning2
	if assistantReasoning.String() != expected {
		t.Errorf("expected reasoning_content %q, got %q", expected, assistantReasoning.String())
	}
}

// TestRequest_InvalidBase64 tests graceful handling of invalid base64
func TestRequest_InvalidBase64(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gpt-5",
		"input": [
			{"type": "message", "role": "user", "content": "Hello"},
			{"type": "reasoning", "encrypted_content": "!!!invalid-base64!!!"},
			{"type": "message", "role": "assistant", "content": "Hi!"}
		]
	}`)

	output := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("gpt-5", inputJSON, false)
	outputStr := string(output)

	// Invalid base64 should be silently skipped
	assistantReasoning := gjson.Get(outputStr, "messages.1.reasoning_content")
	if assistantReasoning.Exists() && assistantReasoning.String() != "" {
		t.Errorf("expected no reasoning_content for invalid base64, got %q", assistantReasoning.String())
	}
}

// TestRequest_ReasoningWithFunctionCall tests reasoning attached to function_call items
func TestRequest_ReasoningWithFunctionCall(t *testing.T) {
	reasoningText := "I need to call a function..."
	encryptedContent := base64.StdEncoding.EncodeToString([]byte(reasoningText))

	inputJSON := []byte(`{
		"model": "gpt-5",
		"input": [
			{"type": "message", "role": "user", "content": "What's the weather?"},
			{"type": "reasoning", "encrypted_content": "` + encryptedContent + `"},
			{"type": "function_call", "call_id": "call_123", "name": "get_weather", "arguments": "{\"location\":\"NYC\"}"}
		]
	}`)

	output := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("gpt-5", inputJSON, false)
	outputStr := string(output)

	// The function_call message is at index 1 (user -> assistant with tool_calls)
	// Function call message should have reasoning_content
	assistantReasoning := gjson.Get(outputStr, "messages.1.reasoning_content")
	if !assistantReasoning.Exists() {
		t.Fatalf("expected reasoning_content to exist on function_call message")
	}
	if assistantReasoning.String() != reasoningText {
		t.Errorf("expected reasoning_content %q, got %q", reasoningText, assistantReasoning.String())
	}

	// Verify tool_calls is preserved
	toolCallName := gjson.Get(outputStr, "messages.1.tool_calls.0.function.name")
	if toolCallName.String() != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %q", toolCallName.String())
	}
}

// TestRequest_AssistantMessageAndFunctionCallMerged ensures assistant content + tool_call are unified in one message.
func TestRequest_AssistantMessageAndFunctionCallMerged(t *testing.T) {
	reasoningText := "Need a tool call."
	encryptedContent := base64.StdEncoding.EncodeToString([]byte(reasoningText))

	inputJSON := []byte(`{
		"model": "gpt-5",
		"input": [
			{"type": "message", "role": "user", "content": "weather?"},
			{"type": "reasoning", "encrypted_content": "` + encryptedContent + `"},
			{"type": "message", "role": "assistant", "content": [{"type":"output_text","text":"Let me check."}]},
			{"type": "function_call", "call_id": "call_999", "name": "get_weather", "arguments": "{\"location\":\"NYC\"}"}
		]
	}`)

	output := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("gpt-5", inputJSON, false)
	outputStr := string(output)

	if got := gjson.Get(outputStr, "messages.#").Int(); got != 2 {
		t.Fatalf("expected exactly 2 messages (user + merged assistant), got %d", got)
	}

	if got := gjson.Get(outputStr, "messages.1.role").String(); got != "assistant" {
		t.Fatalf("expected assistant at messages.1, got %q", got)
	}

	if got := gjson.Get(outputStr, "messages.1.reasoning_content").String(); got != reasoningText {
		t.Fatalf("expected reasoning_content %q, got %q", reasoningText, got)
	}

	if got := gjson.Get(outputStr, "messages.1.content.0.text").String(); got != "Let me check." {
		t.Fatalf("expected assistant content 'Let me check.', got %q", got)
	}

	if got := gjson.Get(outputStr, "messages.1.tool_calls.0.function.name").String(); got != "get_weather" {
		t.Fatalf("expected merged tool call name 'get_weather', got %q", got)
	}
}

// TestRequest_PendingAssistantFlushedBeforeToolOutput ensures tool output follows merged assistant tool_calls.
func TestRequest_PendingAssistantFlushedBeforeToolOutput(t *testing.T) {
	reasoningText := "Call tool now."
	encryptedContent := base64.StdEncoding.EncodeToString([]byte(reasoningText))

	inputJSON := []byte(`{
		"model": "gpt-5",
		"input": [
			{"type":"message","role":"user","content":"start"},
			{"type":"reasoning","encrypted_content":"` + encryptedContent + `"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Calling tool"}]},
			{"type":"function_call","call_id":"call_1","name":"do_work","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_1","output":"done"},
			{"type":"message","role":"user","content":"next"}
		]
	}`)

	output := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("gpt-5", inputJSON, false)
	outputStr := string(output)

	if got := gjson.Get(outputStr, "messages.#").Int(); got != 4 {
		t.Fatalf("expected 4 messages (user, assistant, tool, user), got %d", got)
	}

	if got := gjson.Get(outputStr, "messages.1.role").String(); got != "assistant" {
		t.Fatalf("expected assistant at messages.1, got %q", got)
	}
	if got := gjson.Get(outputStr, "messages.1.reasoning_content").String(); got != reasoningText {
		t.Fatalf("expected reasoning_content %q, got %q", reasoningText, got)
	}
	if got := gjson.Get(outputStr, "messages.1.tool_calls.0.id").String(); got != "call_1" {
		t.Fatalf("expected tool_call id 'call_1', got %q", got)
	}

	if got := gjson.Get(outputStr, "messages.2.role").String(); got != "tool" {
		t.Fatalf("expected tool message at messages.2, got %q", got)
	}
	if got := gjson.Get(outputStr, "messages.2.tool_call_id").String(); got != "call_1" {
		t.Fatalf("expected tool_call_id 'call_1', got %q", got)
	}
}

// TestRequest_NoReasoningForNonAssistant tests reasoning is not attached to user messages
func TestRequest_NoReasoningForNonAssistant(t *testing.T) {
	reasoningText := "Some reasoning..."
	encryptedContent := base64.StdEncoding.EncodeToString([]byte(reasoningText))

	inputJSON := []byte(`{
		"model": "gpt-5",
		"input": [
			{"type": "reasoning", "encrypted_content": "` + encryptedContent + `"},
			{"type": "message", "role": "user", "content": "Hello"}
		]
	}`)

	output := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("gpt-5", inputJSON, false)
	outputStr := string(output)

	// User message should NOT have reasoning_content
	userReasoning := gjson.Get(outputStr, "messages.0.reasoning_content")
	if userReasoning.Exists() && userReasoning.String() != "" {
		t.Errorf("expected no reasoning_content on user message, got %q", userReasoning.String())
	}
}

// TestResponse_Streaming_EncryptedContent tests encoding reasoning_content to encrypted_content in streaming
func TestResponse_Streaming_EncryptedContent(t *testing.T) {
	reasoningText := "Let me think about this step by step..."

	// Simulate streaming chunks with reasoning_content
	in := []string{
		`data: {"id":"test-123","object":"chat.completion.chunk","created":1234567890,"choices":[{"index":0,"delta":{"reasoning_content":"Let me "},"finish_reason":null}]}`,
		`data: {"id":"test-123","object":"chat.completion.chunk","created":1234567890,"choices":[{"index":0,"delta":{"reasoning_content":"think about "},"finish_reason":null}]}`,
		`data: {"id":"test-123","object":"chat.completion.chunk","created":1234567890,"choices":[{"index":0,"delta":{"reasoning_content":"this step by step..."},"finish_reason":null}]}`,
		`data: {"id":"test-123","object":"chat.completion.chunk","created":1234567890,"choices":[{"index":0,"delta":{"content":"The answer"},"finish_reason":null}]}`,
		`data: {"id":"test-123","object":"chat.completion.chunk","created":1234567890,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}

	var param any
	var out []string
	for _, line := range in {
		out = append(out, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "gpt-5", nil, nil, []byte(line), &param)...)
	}

	var gotEncryptedContent string

	for _, chunk := range out {
		ev, data := parseSSEEvent(t, chunk)
		switch ev {
		case "response.output_item.done":
			if data.Get("item.type").String() == "reasoning" {
				gotEncryptedContent = data.Get("item.encrypted_content").String()
			}
		}
	}

	// Verify encrypted_content is valid base64 and decodes to original text
	if gotEncryptedContent == "" {
		t.Fatalf("expected encrypted_content to be present")
	}
	decoded, err := base64.StdEncoding.DecodeString(gotEncryptedContent)
	if err != nil {
		t.Fatalf("failed to decode encrypted_content: %v", err)
	}
	if string(decoded) != reasoningText {
		t.Errorf("expected decoded encrypted_content %q, got %q", reasoningText, string(decoded))
	}
}

// TestResponse_Streaming_CompletedOutput tests encrypted_content in response.completed output array
func TestResponse_Streaming_CompletedOutput(t *testing.T) {
	reasoningText := "Thinking..."

	in := []string{
		`data: {"id":"test-456","object":"chat.completion.chunk","created":1234567890,"choices":[{"index":0,"delta":{"reasoning_content":"Thinking..."},"finish_reason":null}]}`,
		`data: {"id":"test-456","object":"chat.completion.chunk","created":1234567890,"choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}`,
		`data: {"id":"test-456","object":"chat.completion.chunk","created":1234567890,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}

	var param any
	var out []string
	for _, line := range in {
		out = append(out, ConvertOpenAIChatCompletionsResponseToOpenAIResponses(context.Background(), "gpt-5", nil, nil, []byte(line), &param)...)
	}

	var (
		outputEncrypted string
		outputSummary   string
	)

	for _, chunk := range out {
		ev, data := parseSSEEvent(t, chunk)
		if ev == "response.completed" {
			// Check reasoning item in response.output array
			output := data.Get("response.output")
			if output.Exists() && output.IsArray() {
				for _, item := range output.Array() {
					if item.Get("type").String() == "reasoning" {
						outputEncrypted = item.Get("encrypted_content").String()
						outputSummary = item.Get("summary.0.text").String()
					}
				}
			}
		}
	}

	if outputSummary != reasoningText {
		t.Errorf("expected output summary %q, got %q", reasoningText, outputSummary)
	}

	if outputEncrypted == "" {
		t.Fatalf("expected encrypted_content in response.output")
	}
	decoded, err := base64.StdEncoding.DecodeString(outputEncrypted)
	if err != nil {
		t.Fatalf("failed to decode output encrypted_content: %v", err)
	}
	if string(decoded) != reasoningText {
		t.Errorf("expected decoded output encrypted_content %q, got %q", reasoningText, string(decoded))
	}
}

// TestResponse_NonStreaming_EncryptedContent tests encoding reasoning_content to encrypted_content in non-streaming
func TestResponse_NonStreaming_EncryptedContent(t *testing.T) {
	reasoningText := "I am reasoning about this..."

	inputJSON := []byte(`{
		"id": "test-789",
		"object": "chat.completion",
		"created": 1234567890,
		"model": "gpt-5",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"reasoning_content": "` + reasoningText + `",
				"content": "The answer is 42."
			},
			"finish_reason": "stop"
		}],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 5,
			"total_tokens": 15
		}
	}`)

	output := ConvertOpenAIChatCompletionsResponseToOpenAIResponsesNonStream(context.Background(), "gpt-5", nil, nil, inputJSON, nil)
	outputStr := string(output)

	// Check reasoning item exists with encrypted_content
	outputArray := gjson.Get(outputStr, "output")
	if !outputArray.Exists() || !outputArray.IsArray() {
		t.Fatalf("expected output array to exist")
	}

	var reasoningItem gjson.Result
	for _, item := range outputArray.Array() {
		if item.Get("type").String() == "reasoning" {
			reasoningItem = item
			break
		}
	}

	if !reasoningItem.Exists() {
		t.Fatalf("expected reasoning item in output")
	}

	// Check summary text
	summaryText := reasoningItem.Get("summary.0.text").String()
	if summaryText != reasoningText {
		t.Errorf("expected summary text %q, got %q", reasoningText, summaryText)
	}

	// Check encrypted_content
	encryptedContent := reasoningItem.Get("encrypted_content").String()
	if encryptedContent == "" {
		t.Fatalf("expected encrypted_content to be present")
	}

	decoded, err := base64.StdEncoding.DecodeString(encryptedContent)
	if err != nil {
		t.Fatalf("failed to decode encrypted_content: %v", err)
	}
	if string(decoded) != reasoningText {
		t.Errorf("expected decoded encrypted_content %q, got %q", reasoningText, string(decoded))
	}

	// Verify message item also exists
	var messageItem gjson.Result
	for _, item := range outputArray.Array() {
		if item.Get("type").String() == "message" {
			messageItem = item
			break
		}
	}
	if !messageItem.Exists() {
		t.Fatalf("expected message item in output")
	}
	if messageItem.Get("content.0.text").String() != "The answer is 42." {
		t.Errorf("expected message content 'The answer is 42.', got %q", messageItem.Get("content.0.text").String())
	}
}

// TestRoundTrip tests the full round-trip: reasoning_content -> encrypted_content -> reasoning_content
func TestRoundTrip(t *testing.T) {
	originalReasoning := "This is my original chain of thought that needs to be preserved."

	// Step 1: Non-streaming response converts reasoning_content to encrypted_content
	responseJSON := []byte(`{
		"id": "roundtrip-test",
		"object": "chat.completion",
		"created": 1234567890,
		"model": "gpt-5",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"reasoning_content": "` + originalReasoning + `",
				"content": "Done."
			},
			"finish_reason": "stop"
		}]
	}`)

	responseOutput := ConvertOpenAIChatCompletionsResponseToOpenAIResponsesNonStream(context.Background(), "gpt-5", nil, nil, responseJSON, nil)

	// Extract encrypted_content from response
	encryptedFromResponse := gjson.Get(responseOutput, "output.0.encrypted_content").String()
	if encryptedFromResponse == "" {
		t.Fatalf("expected encrypted_content in response")
	}

	// Step 2: Client sends back the encrypted_content in a new request
	requestJSON := []byte(`{
		"model": "gpt-5",
		"input": [
			{"type": "message", "role": "user", "content": "First question?"},
			{"type": "reasoning", "encrypted_content": "` + encryptedFromResponse + `"},
			{"type": "message", "role": "assistant", "content": "First answer."},
			{"type": "message", "role": "user", "content": "Follow-up?"}
		]
	}`)

	requestOutput := ConvertOpenAIResponsesRequestToOpenAIChatCompletions("gpt-5", requestJSON, false)
	requestOutputStr := string(requestOutput)

	// Step 3: Verify reasoning_content is restored
	// The assistant message is at index 1 (user -> assistant -> user)
	restoredReasoning := gjson.Get(requestOutputStr, "messages.1.reasoning_content").String()
	if restoredReasoning != originalReasoning {
		t.Errorf("round-trip failed: expected %q, got %q", originalReasoning, restoredReasoning)
	}
}

// TestResponse_NonStreaming_NoReasoning tests response without reasoning_content
func TestResponse_NonStreaming_NoReasoning(t *testing.T) {
	inputJSON := []byte(`{
		"id": "test-no-reasoning",
		"object": "chat.completion",
		"created": 1234567890,
		"model": "gpt-5",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "Just a normal response."
			},
			"finish_reason": "stop"
		}]
	}`)

	output := ConvertOpenAIChatCompletionsResponseToOpenAIResponsesNonStream(context.Background(), "gpt-5", nil, nil, inputJSON, nil)
	outputStr := string(output)

	// Should have only message item, no reasoning item
	outputArray := gjson.Get(outputStr, "output")
	if !outputArray.Exists() || !outputArray.IsArray() {
		t.Fatalf("expected output array to exist")
	}

	for _, item := range outputArray.Array() {
		if item.Get("type").String() == "reasoning" {
			t.Errorf("did not expect reasoning item when no reasoning_content present")
		}
	}
}
