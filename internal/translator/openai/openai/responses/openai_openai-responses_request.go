package responses

import (
	"encoding/base64"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ConvertOpenAIResponsesRequestToOpenAIChatCompletions converts OpenAI responses format to OpenAI chat completions format.
// It transforms the OpenAI responses API format (with instructions and input array) into the standard
// OpenAI chat completions format (with messages array and system content).
//
// The conversion handles:
// 1. Model name and streaming configuration
// 2. Instructions to system message conversion
// 3. Input array to messages array transformation
// 4. Tool definitions and tool choice conversion
// 5. Function calls and function results handling
// 6. Generation parameters mapping (max_tokens, reasoning, etc.)
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - rawJSON: The raw JSON request data in OpenAI responses format
//   - stream: A boolean indicating if the request is for a streaming response
//
// Returns:
//   - []byte: The transformed request data in OpenAI chat completions format
func ConvertOpenAIResponsesRequestToOpenAIChatCompletions(modelName string, inputRawJSON []byte, stream bool) []byte {
	return ConvertOpenAIResponsesRequestToOpenAIChatCompletionsWithDefault(modelName, inputRawJSON, stream, "")
}

// ConvertOpenAIResponsesRequestToOpenAIChatCompletionsWithDefault converts OpenAI responses format to OpenAI chat completions format
// with support for a default reasoning effort value.
// It transforms the OpenAI responses API format (with instructions and input array) into the standard
// OpenAI chat completions format (with messages array and system content).
//
// The conversion handles:
// 1. Model name and streaming configuration
// 2. Instructions to system message conversion
// 3. Input array to messages array transformation
// 4. Tool definitions and tool choice conversion
// 5. Function calls and function results handling
// 6. Generation parameters mapping (max_tokens, reasoning, etc.)
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - inputRawJSON: The raw JSON request data in OpenAI responses format
//   - stream: A boolean indicating if the request is for a streaming response
//   - defaultReasoningEffort: The default reasoning effort to use if not present in the request (may be empty)
//
// Returns:
//   - []byte: The transformed request data in OpenAI chat completions format
func ConvertOpenAIResponsesRequestToOpenAIChatCompletionsWithDefault(modelName string, inputRawJSON []byte, stream bool, defaultReasoningEffort string) []byte {
	rawJSON := inputRawJSON
	// Base OpenAI chat completions template with default values
	out := `{"model":"","messages":[],"stream":false}`

	root := gjson.ParseBytes(rawJSON)

	// Set model name
	out, _ = sjson.Set(out, "model", modelName)

	// Set stream configuration
	out, _ = sjson.Set(out, "stream", stream)

	// Map generation parameters from responses format to chat completions format
	if maxTokens := root.Get("max_output_tokens"); maxTokens.Exists() {
		out, _ = sjson.Set(out, "max_tokens", maxTokens.Int())
	}

	if parallelToolCalls := root.Get("parallel_tool_calls"); parallelToolCalls.Exists() {
		out, _ = sjson.Set(out, "parallel_tool_calls", parallelToolCalls.Bool())
	}

	// Convert instructions to system message
	if instructions := root.Get("instructions"); instructions.Exists() {
		systemMessage := `{"role":"system","content":""}`
		systemMessage, _ = sjson.Set(systemMessage, "content", instructions.String())
		out, _ = sjson.SetRaw(out, "messages.-1", systemMessage)
	}

	// Convert input array to messages.
	// Keep assistant message/tool-calls merged in a single turn.
	var pendingReasoningContent string
	var pendingAssistantMessage string

	flushPendingAssistant := func() {
		if pendingAssistantMessage == "" {
			return
		}
		out, _ = sjson.SetRaw(out, "messages.-1", pendingAssistantMessage)
		pendingAssistantMessage = ""
	}

	attachPendingReasoningToAssistant := func() {
		if pendingReasoningContent == "" || pendingAssistantMessage == "" {
			return
		}
		if existing := gjson.Get(pendingAssistantMessage, "reasoning_content").String(); existing != "" {
			pendingAssistantMessage, _ = sjson.Set(pendingAssistantMessage, "reasoning_content", existing+pendingReasoningContent)
			pendingAssistantMessage, _ = sjson.Set(pendingAssistantMessage, "reasoning", existing+pendingReasoningContent)
		} else {
			pendingAssistantMessage, _ = sjson.Set(pendingAssistantMessage, "reasoning_content", pendingReasoningContent)
			pendingAssistantMessage, _ = sjson.Set(pendingAssistantMessage, "reasoning", pendingReasoningContent)
		}
		pendingReasoningContent = ""
	}

	buildMessage := func(item gjson.Result, role string) string {
		message := `{"role":"","content":[]}`
		message, _ = sjson.Set(message, "role", role)

		if content := item.Get("content"); content.Exists() && content.IsArray() {
			content.ForEach(func(_, contentItem gjson.Result) bool {
				contentType := contentItem.Get("type").String()
				if contentType == "" {
					contentType = "input_text"
				}

				switch contentType {
				case "input_text", "output_text":
					text := contentItem.Get("text").String()
					contentPart := `{"type":"text","text":""}`
					contentPart, _ = sjson.Set(contentPart, "text", text)
					message, _ = sjson.SetRaw(message, "content.-1", contentPart)
				case "input_image":
					imageURL := contentItem.Get("image_url").String()
					contentPart := `{"type":"image_url","image_url":{"url":""}}`
					contentPart, _ = sjson.Set(contentPart, "image_url.url", imageURL)
					message, _ = sjson.SetRaw(message, "content.-1", contentPart)
				}
				return true
			})
		} else if content.Type == gjson.String {
			message, _ = sjson.Set(message, "content", content.String())
		}

		return message
	}

	if input := root.Get("input"); input.Exists() && input.IsArray() {
		input.ForEach(func(_, item gjson.Result) bool {
			itemType := item.Get("type").String()
			if itemType == "" && item.Get("role").String() != "" {
				itemType = "message"
			}

			switch itemType {
			case "reasoning":
				// Handle reasoning input items with encrypted_content
				// Base64 decode and defer until we encounter the next assistant message
				if enc := item.Get("encrypted_content"); enc.Exists() && enc.String() != "" {
					decoded, err := base64.StdEncoding.DecodeString(enc.String())
					if err == nil {
						// Accumulate pending reasoning content (concatenate if multiple)
						pendingReasoningContent += string(decoded)
					}
					// If base64 decode fails, silently skip (invalid data)
				}

			case "message", "":
				// Handle regular message conversion.
				role := item.Get("role").String()
				if role == "developer" {
					role = "user"
				}
				message := buildMessage(item, role)
				if role == "assistant" {
					// Hold assistant message to merge with following function_call items.
					flushPendingAssistant()
					pendingAssistantMessage = message
					attachPendingReasoningToAssistant()
				} else {
					flushPendingAssistant()
					out, _ = sjson.SetRaw(out, "messages.-1", message)
				}

			case "function_call":
				// Handle function call conversion to assistant message with tool_calls.
				// Merge into pending assistant turn if one exists.
				if pendingAssistantMessage == "" {
					pendingAssistantMessage = `{"role":"assistant","tool_calls":[]}`
				}
				attachPendingReasoningToAssistant()
				toolCall := `{"id":"","type":"function","function":{"name":"","arguments":""}}`

				if callId := item.Get("call_id"); callId.Exists() {
					toolCall, _ = sjson.Set(toolCall, "id", callId.String())
				}

				if name := item.Get("name"); name.Exists() {
					toolCall, _ = sjson.Set(toolCall, "function.name", name.String())
				}

				if arguments := item.Get("arguments"); arguments.Exists() {
					toolCall, _ = sjson.Set(toolCall, "function.arguments", arguments.String())
				}

				pendingAssistantMessage, _ = sjson.SetRaw(pendingAssistantMessage, "tool_calls.-1", toolCall)

			case "function_call_output":
				// Handle function call output conversion to tool message.
				flushPendingAssistant()
				toolMessage := `{"role":"tool","tool_call_id":"","content":""}`

				if callId := item.Get("call_id"); callId.Exists() {
					toolMessage, _ = sjson.Set(toolMessage, "tool_call_id", callId.String())
				}

				if output := item.Get("output"); output.Exists() {
					toolMessage, _ = sjson.Set(toolMessage, "content", output.String())
				}

				out, _ = sjson.SetRaw(out, "messages.-1", toolMessage)
			}

			return true
		})
		flushPendingAssistant()
	} else if input.Type == gjson.String {
		msg := "{}"
		msg, _ = sjson.Set(msg, "role", "user")
		msg, _ = sjson.Set(msg, "content", input.String())
		out, _ = sjson.SetRaw(out, "messages.-1", msg)
	}

	// Convert tools from responses format to chat completions format
	if tools := root.Get("tools"); tools.Exists() && tools.IsArray() {
		var chatCompletionsTools []interface{}

		tools.ForEach(func(_, tool gjson.Result) bool {
			// Built-in tools (e.g. {"type":"web_search"}) are already compatible with the Chat Completions schema.
			// Only function tools need structural conversion because Chat Completions nests details under "function".
			toolType := tool.Get("type").String()
			if toolType != "" && toolType != "function" && tool.IsObject() {
				// Almost all providers lack built-in tools, so we just ignore them.
				// chatCompletionsTools = append(chatCompletionsTools, tool.Value())
				return true
			}

			chatTool := `{"type":"function","function":{}}`

			// Convert tool structure from responses format to chat completions format
			function := `{"name":"","description":"","parameters":{}}`

			if name := tool.Get("name"); name.Exists() {
				function, _ = sjson.Set(function, "name", name.String())
			}

			if description := tool.Get("description"); description.Exists() {
				function, _ = sjson.Set(function, "description", description.String())
			}

			if parameters := tool.Get("parameters"); parameters.Exists() {
				function, _ = sjson.SetRaw(function, "parameters", parameters.Raw)
			}

			chatTool, _ = sjson.SetRaw(chatTool, "function", function)
			chatCompletionsTools = append(chatCompletionsTools, gjson.Parse(chatTool).Value())

			return true
		})

		if len(chatCompletionsTools) > 0 {
			out, _ = sjson.Set(out, "tools", chatCompletionsTools)
		}
	}

	// Handle reasoning effort: request value takes priority, then config default
	var effort string
	if reasoningEffort := root.Get("reasoning.effort"); reasoningEffort.Exists() {
		effort = strings.ToLower(strings.TrimSpace(reasoningEffort.String()))
	}
	// If no effort in request, use default from config
	if effort == "" && defaultReasoningEffort != "" {
		effort = strings.ToLower(strings.TrimSpace(defaultReasoningEffort))
	}
	if effort != "" {
		out, _ = sjson.Set(out, "reasoning_effort", effort)
	}

	// Convert tool_choice if present
	if toolChoice := root.Get("tool_choice"); toolChoice.Exists() {
		out, _ = sjson.Set(out, "tool_choice", toolChoice.String())
	}

	return []byte(out)
}
