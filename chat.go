package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/sashabaranov/go-openai"
)

// GetLocalTools defines the schema for the model's capabilities
func GetLocalTools() []openai.Tool {
	return []openai.Tool{
		{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        "write_file",
				Description: "Write content to a file on disk",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"path":    map[string]interface{}{"type": "string", "description": "The file path"},
						"content": map[string]interface{}{"type": "string", "description": "The text to write"},
					},
					"required": []string{"path", "content"},
				},
			},
		},
	}
}

func main() {
	verbose := flag.Bool("verbose", false, "enable verbose logging")
	baseURL := flag.String("base-url", "http://localhost:11434/v1", "Base URL for the local LLM API")
	modelName := flag.String("model", "", "Specific model name to use (optional, will auto-detect if not provided)")
	flag.Parse()

	if *verbose {
		log.SetOutput(os.Stderr)
		log.SetFlags(log.LstdFlags | log.Lshortfile)
	} else {
		log.SetOutput(os.Stdout)
		log.SetFlags(0)
		log.SetPrefix("")
	}

	// 1. Setup Configuration
	config := openai.DefaultConfig("")
	config.BaseURL = *baseURL

	// 2. Initialize Client
	client := openai.NewClientWithConfig(config)

	// 3. Fetch Model Name automatically if not specified
	var actualModelName string
	var err error
	if *modelName != "" {
		actualModelName = *modelName
		if *verbose {
			log.Printf("Using specified model: %s", actualModelName)
		}
	} else {
		actualModelName, err = GetFirstAvailableModel(client)
		if err != nil {
			log.Fatalf("Error fetching model: %v", err)
		}
		if *verbose {
			log.Printf("Automatically using model: %s", actualModelName)
		}
	}

	// 4. Setup Input Scanner
	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	// 5. Initialize and Run Agent
	agent := NewAgent(client, actualModelName, getUserMessage, *verbose)
	err = agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

// --- Agent Logic ---

type Agent struct {
	client         *openai.Client
	getUserMessage func() (string, bool)
	verbose        bool
	modelName      string
}

func NewAgent(client *openai.Client, modelName string, getUserMessage func() (string, bool), verbose bool) *Agent {
	return &Agent{
		client:         client,
		modelName:      modelName,
		getUserMessage: getUserMessage,
		verbose:        verbose,
	}
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: "You are a coding agent. Use tools to help the user. If you see a file needs fixing, use write_file.",
		},
	}

	fmt.Println("Chat with Ollama (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok || userInput == "" {
			break
		}

		conversation = append(conversation, openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userInput,
		})

		resp, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}

		if len(resp.Choices) > 0 {
			msg := resp.Choices[0].Message
			conversation = append(conversation, msg)
			fmt.Printf("\u001b[93mOpenAI\u001b[0m: %s\n", msg.Content)
			
			// Note: To actually execute write_file, you'd handle msg.ToolCalls here
		}
	}
	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []openai.ChatCompletionMessage) (*openai.ChatCompletionResponse, error) {
	resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:     a.modelName,
		Messages:  conversation,
		MaxTokens: 1024,
		Tools:     GetLocalTools(),
	})
	return &resp, err
}

func GetFirstAvailableModel(client *openai.Client) (string, error) {
	ctx := context.Background()
	models, err := client.ListModels(ctx)
	if err != nil {
		return "", err
	}
	if len(models.Models) == 0 {
		return "", fmt.Errorf("no models found in Ollama")
	}
	return models.Models[0].ID, nil
}