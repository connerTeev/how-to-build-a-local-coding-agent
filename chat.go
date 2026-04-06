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

func RunLocalAgent(modelName string, userPrompt string) {
	// 1. Point to your local 5080 (Ollama)
	config := openai.DefaultConfig("local-key-not-needed")
	config.BaseURL = "http://localhost:11434/v1"
	client := openai.NewClientWithConfig(config)

	// 2. Setup the request with the chosen model
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: modelName,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleSystem,
					Content: "You are a coding agent. Use tools to help the user. If you see a file needs fixing, use write_file.",
				},
				{
					Role:    openai.ChatMessageRoleUser,
					Content: userPrompt,
				},
			},
			// Define tools here so the local model can 'see' them
			Tools: GetLocalTools(), 
		},
	)

	if err != nil {
		fmt.Printf("ChatCompletion error: %v\n", err)
		return
	}

	fmt.Println("Agent Response:", resp.Choices[0].Message.Content)
}

func main() {
	verbose := flag.Bool("verbose", false, "enable verbose logging")
	flag.Parse()

	if *verbose {
		log.SetOutput(os.Stderr)
		log.SetFlags(log.LstdFlags | log.Lshortfile)
		log.Println("Verbose logging enabled")
	} else {
		log.SetOutput(os.Stdout)
		log.SetFlags(0)
		log.SetPrefix("")
	}

	config := openai.DefaultConfig("ollama")
	config.BaseURL = "http://localhost:11434/v1" // Direct to your local Ollama instance
	client := openai.NewClientWithConfig(config)
	if *verbose {
		log.Println("Local Ollama client initialized")
	}

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	agent := NewAgent(client, getUserMessage, *verbose)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func NewAgent(client *openai.Client, getUserMessage func() (string, bool), verbose bool) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		verbose:        verbose,
	}
}

type Agent struct {
	client         *openai.Client
	getUserMessage func() (string, bool)
	verbose        bool
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []openai.ChatCompletionMessage{}

	if a.verbose {
		log.Println("Starting chat session")
	}
	fmt.Println("Chat with OpenAI (use 'ctrl-c' to quit)")

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			if a.verbose {
				log.Println("User input ended, breaking from chat loop")
			}
			break
		}

		// Skip empty messages
		if userInput == "" {
			if a.verbose {
				log.Println("Skipping empty message")
			}
			continue
		}

		if a.verbose {
			log.Printf("User input received: %q", userInput)
		}

		userMessage := openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: userInput,
		}
		conversation = append(conversation, userMessage)

		if a.verbose {
			log.Printf("Sending message to OpenAI, conversation length: %d", len(conversation))
		}

		message, err := a.runInference(ctx, conversation)
		if err != nil {
			if a.verbose {
				log.Printf("Error during inference: %v", err)
			}
			return err
		}

		if len(message.Choices) > 0 {
			responseMessage := message.Choices[0].Message
			conversation = append(conversation, responseMessage)

			if a.verbose {
				log.Printf("Received response from OpenAI with content: %s", responseMessage.Content)
			}

			fmt.Printf("\u001b[93mOpenAI\u001b[0m: %s\n", responseMessage.Content)
		}
	}

	if a.verbose {
		log.Println("Chat session ended")
	}
	return nil
}

func (a *Agent) runInference(ctx context.Context, conversation []openai.ChatCompletionMessage) (*openai.ChatCompletionResponse, error) {
	if a.verbose {
		log.Printf("Making API call to OpenAI model")
	}

	// Create a chat completion request
	resp, err := a.client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model:    "ollama", // Using ollama model
		Messages: conversation,
		MaxTokens: 1024,
	})

	if a.verbose {
		if err != nil {
			log.Printf("API call failed: %v", err)
		} else {
			log.Printf("API call successful, response received")
		}
	}

	return &resp, err
}
