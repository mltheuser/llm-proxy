# Testing

Project relies on a built-in
scenario runner for end-to-end provider verification. Unit tests are the
exception, and there are no classic integration tests.

During development, write whatever throwaway tests you need, but do not commit
them. What lands in the tree is E2E scenarios; a unit test is only committed when
the owner has explicitly approved it.

## Linting

-   **Commands**: `make lint` (report) and `make fmt` (auto-format). Both bootstrap a
    pinned `golangci-lint` into `./bin` on first run; no global install needed.
-   **Config**: `.golangci.yml` (v2 schema) at the repo root.
-   **Scope**: The whole module (`./...`). `make lint` must exit clean: findings are
    fixed, never suppressed (no `//nolint` directives).

## Unit Tests

-   **Command**: `make test`
-   **Scope**: Isolated units of non-trivial logic with no external dependencies. No mocking. New unit tests
    require owner sign-off before they are committed.
-   **Location**: `*_test.go` files next to the code (e.g. `router/router_test.go`).

## End-to-End Tests (`/v1/test`)

The primary way to verify providers is the centralized, scenario-based E2E runner built into the server.

1.  **Build**: `make build` (always rebuild after changes).
2.  **Run Server**: `set -a && source .env && set +a && ./bin/ai-router serve --debug`
    - Cloud provider API keys live in `.env` at the project root (not committed).
    - Key naming convention matches the server's expected format: `AI_ROUTER_<PROVIDER>_API_KEY` (e.g. `AI_ROUTER_OPENROUTER_API_KEY`).
3.  **Trigger**: `curl -X POST http://localhost:8787/v1/test -d '{"provider": "ollama"}' | jq .`
    - Optionally pin a specific model: `curl -X POST http://localhost:8787/v1/test -d '{"provider": "openrouter", "model": "~anthropic/claude-sonnet-latest"}' | jq .`
    - The `model` value must match the exact ID as returned by `GET /v1/models`. When omitted, the best available model per scenario is auto-selected.

**What happens**: the server self-verifies by running `Verify()`, `ListModels()`, and executing applicable scenarios from `scenarios/`.

**Scenarios**: defined in `scenarios/`. Each declares its `RequiredCapabilities()` and runs a specific functional test. Scenarios are skipped (not failed) when the target model lacks a required capability.
