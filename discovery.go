package main

import (
	"encoding/json"
	"fmt"
	"net/http"
)

type OllamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

// GetAvailableModels checks for local Ollama instance and returns available models
// This function is designed to work completely locally without external dependencies
func GetAvailableModels() []string {
	resp, err := http.Get("http://localhost:11434/api/tags")
	if err != nil {
		fmt.Println("Error: Local Ollama instance not found. Ensure it is running on port 11434.")
		fmt.Println("This is a local-first operation - no internet connectivity required.")
		return nil
	}
	defer resp.Body.Close()

	var tags OllamaTagsResponse
	json.NewDecoder(resp.Body).Decode(&tags)

	var names []string
	for _, m := range tags.Models {
		names = append(names, m.Name)
	}
	return names
}