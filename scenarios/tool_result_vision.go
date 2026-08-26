package scenarios

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/mltheuser/ai-router/api"
)

func init() {
	Register(&toolResultVision{})
}

type toolResultVision struct{}

func (s *toolResultVision) Name() string {
	return "tool_result_vision"
}

func (s *toolResultVision) Description() string {
	return "Verifies that image content inside a tool result reaches the model"
}

func (s *toolResultVision) RequiredCapabilities() []api.Capability {
	return []api.Capability{api.CapabilityChat, api.CapabilityTools, api.CapabilityVision}
}

func (s *toolResultVision) Run(ctx context.Context, baseURL string, modelID string) *api.ScenarioResult {
	url := fmt.Sprintf("%s/v1/chat/completions", baseURL)
	client := http.DefaultClient
	result := api.NewResult()

	tools := []api.ToolDefinition{
		{
			Name:        "take_photo",
			Description: "Take a photo with the camera and return it as an image",
			Parameters: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			},
		},
	}

	messages := []api.ChatMessage{
		{Role: api.RoleUser, Content: api.TextContent("Use the take_photo tool, then tell me what fruit the photo shows. Be concise.")},
	}

	resp, err := doToolRequest(ctx, client, url, api.ChatRequest{
		Model:    modelID,
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		result.Fail("tool invocation request", fmt.Sprintf("%v", err))
		return result
	}
	if resp.FinishReason != api.FinishReasonToolCalls || len(resp.Message.ToolCalls) == 0 {
		result.Fail("tool invocation request", fmt.Sprintf("expected a tool call, got finish_reason '%s'", resp.FinishReason))
		return result
	}
	result.Pass("tool invocation request")

	messages = append(messages, resp.Message)
	for _, tc := range resp.Message.ToolCalls {
		messages = append(messages, api.ChatMessage{
			Role: api.RoleTool,
			Content: []api.ContentPart{
				{Type: api.ContentPartText, Text: "Photo taken:"},
				{Type: api.ContentPartImage, MimeType: "image/png", Base64Data: testImageBase64},
			},
			ToolCallID: tc.ID,
		})
	}

	resp, err = doToolRequest(ctx, client, url, api.ChatRequest{
		Model:    modelID,
		Messages: messages,
		Tools:    tools,
	})
	if err != nil {
		result.Fail("tool result image incorporation", fmt.Sprintf("%v", err))
		return result
	}

	content := api.TextFromContent(resp.Message.Content)
	if !strings.Contains(strings.ToLower(content), "apple") {
		result.Fail("tool result image incorporation", fmt.Sprintf("response does not contain 'apple': %s", content))
		return result
	}
	result.Pass("tool result image incorporation")

	return result
}
