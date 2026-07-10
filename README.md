# Ratatosk Router

An API proxy that answers LLM prompts with the **cheapest model that produces a good-enough answer**. OpenAI-compatible — point any client at it and save money.

## How it works

1. **Classification:** a cheap model assesses the prompt's difficulty and picks a starting model in a cost-ordered ladder (`config.yaml`).
2. **Verification + escalation:** the answer is scored (1–10) by a cheap judge model. Below the threshold → escalate to the next, more expensive model, until the threshold is met or the ladder is exhausted (then the best answer so far is returned).

Simple questions get answered for ~1% of the flagship price; the average cost drops sharply.

## Getting started

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

export OPENROUTER_API_KEY=sk-or-...   # or set openrouter.api_key in config.yaml
# edit config.yaml: local api_keys, model ladder, thresholds

uvicorn app.main:app --port 8000
```

Prices are fetched automatically from OpenRouter's `GET /models` at startup and cached (`pricing_refresh_hours`). No prices are hardcoded.

## Usage

Drop-in replacement for any OpenAI-compatible client:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=sk-local-...   # a key from api_keys in config.yaml

curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

The `model` field controls routing:

| Value | Behavior |
| --- | --- |
| `auto` (or omitted/unknown) | routing active, threshold 7 |
| `auto:strict` | higher quality threshold (8) |
| `auto:cheap` | lower threshold (6), max 1 escalation |
| exact OpenRouter id, e.g. `anthropic/claude-sonnet-5` | passthrough, no routing |

`stream: true` is supported (the approved answer is streamed as SSE chunks, including tool calls).

**Tool calls:** requests with `tools` pass through unchanged. When a model responds with `tool_calls`, the judge is skipped and the response is returned immediately — tool-call turns can't be meaningfully quality-scored as text, and agentic clients verify via the tool result anyway.

The response is standard OpenAI format plus a `frugal` field with attempts, scores, cost, and savings versus sending the request straight to the most expensive ladder model:

```json
"frugal": {
  "attempts": [{"model": "google/gemini-2.5-flash-lite", "score": 8, "cost_usd": 0.00011}],
  "final_model": "google/gemini-2.5-flash-lite",
  "classifier_tier": 0,
  "difficulty": "easy",
  "total_cost_usd": 0.00013,
  "baseline_cost_usd": 0.021,
  "saved_pct": 99.4
}
```

## Using with opencode

[opencode](https://opencode.ai) speaks the OpenAI format natively via custom providers. Add this to `opencode.json` in your project, or globally in `~/.config/opencode/opencode.jsonc`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ratatosk": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ratatosk",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "<a key from api_keys in config.yaml>"
      },
      "models": {
        "auto": { "name": "Ratatosk auto" },
        "auto:cheap": { "name": "Ratatosk auto (cheap)" },
        "auto:strict": { "name": "Ratatosk auto (strict)" }
      }
    }
  },
  "model": "ratatosk/auto"
}
```

Then start `opencode` (the default model is set above), or pick a model explicitly:

```bash
opencode -m ratatosk/auto
```

Inside the TUI you can switch between `auto`, `auto:cheap`, and `auto:strict` with the `/models` command.

## Using with aider

[aider](https://aider.chat) supports OpenAI-compatible endpoints through the `openai/` model prefix:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=<a key from api_keys in config.yaml>

aider --model openai/auto
```

Or without environment variables:

```bash
aider --openai-api-base http://localhost:8000/v1 \
      --openai-api-key <key> \
      --model openai/auto
```

aider will warn that it doesn't recognize the model (`Warning: Unknown model openai/auto`) — that's harmless; it just means aider can't estimate context-window size and cost itself. Use `openai/auto:cheap` or `openai/auto:strict` for the other routing modes. You can persist the settings in `~/.aider.conf.yml`:

```yaml
openai-api-base: http://localhost:8000/v1
openai-api-key: <key>
model: openai/auto
```

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible chat
- `GET /v1/models` — the `auto` variants + the ladder models
- `GET /stats` — requests, cost, savings, escalation rate, per-model distribution, score distribution
- `GET /healthz` — health check (no auth)

All requests are logged to SQLite (`db_path` in config).

## Error handling

- OpenRouter 429/5xx: retry with exponential backoff (max 2), otherwise the model is skipped in the ladder
- Classifier down → defaults to `medium`; judge down → the answer passes through (fail open)
- The service degrades, it doesn't die

## Calibration

The judge is the bottleneck: too lenient lets bad answers through, too strict eats up the savings. Watch `score_distribution` in `GET /stats` and adjust `judge.threshold` against real-world usage.

## Tests

```bash
pytest
```
