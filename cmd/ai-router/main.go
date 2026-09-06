// Command ai-router is the process entry point for the AI Router binary.
// It delegates to the cobra command tree defined in the cli package.
package main

import "github.com/mltheuser/ai-router/cli"

func main() {
	cli.Execute()
}
