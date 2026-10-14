# Ratatosk Router

An API proxy that answers LLM prompts with the **cheapest model that produces a good-enough answer**. OpenAI-compatible — point any client at it and save money.

## How it works

1. **Classification:** a cheap model labels the prompt's difficulty *and task type*. Together with what the router has learned (below), that picks a starting model in a cost-ordered ladder (`config.yaml`).
2. **Verification + escalation:** the answer is scored (1–10) by a cheap judge model. Below the threshold → escalate to the next, more expensive model, until the threshold is met or the ladder is exhausted (then the best answer so far is returned).
3. **Learning:** every judged attempt is logged, so the proxy accumulates a measured pass rate per (model, task type, difficulty) and uses it to pick the starting rung instead of the static tier table. Deterministic verifiers, shadow audits and client feedback keep those numbers honest. See [Learning](#learning).
4. **A local routing model:** trained on that log, a small model predicts per prompt which rung will pass, and takes over routing once it has proven on held-out traffic that it is cheaper. See [Local routing model](#local-routing-model).

Simple questions get answered for ~1% of the flagship price; the average cost drops sharply.

## Getting started

```bash
python3 -m venv .venv && source .venv/bin/activate   # needs Python 3.12+
pip install -e ".[dev]"

export OPENROUTER_API_KEY=sk-or-...   # or set openrouter.api_key in config.yaml
# edit config.yaml: replace the placeholder in api_keys with your own local key,
# and adjust the model ladder and thresholds

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
  "task_type": "factual_qa",
  "tier_source": "learned",
  "explored": false,
  "total_cost_usd": 0.00013,
  "baseline_cost_usd": 0.021,
  "saved_pct": 99.4
}
```

`tier_source` says what chose the starting model: `predictor` (the local model),
`neighbors`, `learned` (the skill matrix), `config` (the static tiers) or `cache`. When the
local model routed the request, `predicted_pass` holds its per-rung pass probabilities.
Attempts a deterministic verifier checked also carry `verifier` and `verifier_passed`.

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

## Learning

Difficulty alone is a poor routing key — a cheap model can be excellent at summarization
and hopeless at math. So the classifier also labels each prompt with a **task type**
(`code`, `math`, `extraction`, `summarization`, `factual_qa`, `creative`, `translation`,
`other`) in the same call, at no extra cost.

Because the judge scores every attempt, the request log is training data. Each attempt is
written to a queryable `attempts` table, and aggregating it gives a **skill matrix** —
P(score ≥ threshold | model, task type, difficulty). That matrix is the answer to "which
model is good enough for which task", and it replaces the static `classifier.tiers` map:

```yaml
learning:
  enabled: true
  policy: expected_cost # or: pass_rate
  min_samples: 20 # judged attempts a cell needs before it overrides `tiers`
  min_pass_rate: 0.8 # only used by the "pass_rate" policy
  lookback_days: 30
  refresh_minutes: 60
```

Two policies:

| Policy | Picks |
| --- | --- |
| `expected_cost` | the rung minimising `cost + (1 − pass rate) × cost of escalating onward` |
| `pass_rate` | the cheapest rung whose measured pass rate clears `min_pass_rate` |

`expected_cost` minimises **money, not latency**. With the usual ~10x jumps between rungs,
starting low and letting the judge escalate is cheaper in dollars even at a mediocre pass
rate — so it will happily accept extra round trips. If latency matters more than the last
few percent of savings, use `pass_rate`, which bounds the expected number of attempts.

Pass rates are computed against the *requesting mode's* threshold, so `auto`, `auto:cheap`
and `auto:strict` can each learn a different start rung from the same data.

Cells with fewer than `min_samples` judged attempts fall back to the `classifier.tiers`
map from `config.yaml`, so the static table stays the prior and the system degrades to
the static tiers when it has nothing to go on. `frugal.tier_source` reports which decided
each request.

Watch it fill in at `GET /learn`:

```bash
curl -H "Authorization: Bearer $OPENAI_API_KEY" http://localhost:8000/learn
```

The matrix is recomputed at startup and every `refresh_minutes`; `POST /learn/refresh`
forces it. Setting `learning.enabled: false` puts the static tier map back in its place
(the local model, if one is deployed, then works relative to the static tiers instead).

### Exploration

A learned table goes stale: models get updated, traffic drifts, and a rung the table
stopped choosing stops producing data. So with probability `epsilon` the router starts
**one rung below** the chosen one.

```yaml
exploration:
  enabled: true
  epsilon: 0.10
```

Exploring downward is the safe direction. If the cheaper model turns out to be
inadequate, the judge rejects it and escalation puts the request back on track — the
worst case is one extra cheap attempt, not a bad answer. That makes this an epsilon-greedy
contextual bandit in which the judge is the reward signal and the escalation ladder
bounds the cost of every bad arm pull. `frugal.explored` marks the requests that took the
detour, and `/stats` counts them. Each request also records the exploration probability in
effect when it was routed, which the local model's evaluation depends on.

### Keeping the numbers honest

The learned pass rates are only as good as the judge that produced them, and a cheap judge
that likes fluent-but-wrong answers would teach the router to under-route. Three
countermeasures:

**Deterministic verifiers** (free). Where a task has a mechanical correctness check, run
it: fenced `python` and `json` blocks are parsed, and arithmetic claims of the form
`12 * 3 = 36` are evaluated. These cannot be fooled by fluency. With
`verifiers.enforce: true` a hard failure forces escalation even when the judge was happy.
Verifiers are deliberately conservative — untagged code blocks, fragments with elided
bodies (`...`), REPL transcripts and unevaluable expressions all return no verdict rather
than risk a false alarm.

**Shadow audits** (~2% of a strong model's price). A random sample of *accepted* answers
is re-judged by a flagship model after the response has already gone out, so it costs the
request no latency. The disagreement is logged, which turns the cheap judge's false-accept
rate from an assumption into a measurement — reported overall and per task type under
`judge_calibration` in `/stats`. Off by default since it costs real money.

```yaml
audit:
  enabled: false
  # model: anthropic/claude-opus-4.8   # defaults to the top ladder rung
  sample_pct: 0.02
```

**Client feedback** (free, and the only real ground truth). `POST /v1/feedback` takes an
explicit verdict:

```bash
curl -X POST http://localhost:8000/v1/feedback \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"request_id": "...", "ok": false}'
```

Negatives are also inferred from the conversation itself: a short follow-up like "no,
that's wrong" or "try again", and — for agentic clients — a tool result that carries an
error immediately after a tool call. Since the proxy is stateless, the answer being
reacted to is matched by hashing it and looking the hash up in the log, so no answer text
needs to be stored to make the link.

Verifier verdicts and client feedback feed back into what the router learns: when the
matrix is rebuilt or the local model retrained, an attempt a verifier rejected or a client
marked wrong counts as a **failure regardless of its judge score**. Without that, the
router would only ever learn from the mistakes the cheap judge already caught. Shadow
audits are measurement only — they tell you whether to trust the judge, but don't change
any labels.

### Embedding-based routing

The skill matrix buckets prompts into (task type, difficulty) cells. Embeddings are the
finer-grained version of the same idea: instead of "hard code tasks pass on Haiku 80% of
the time", the router can say "prompts *like this one* passed on Haiku 9 times out of 10,
start there". When the neighbourhood is dense enough, neighbours outrank the matrix, which
outranks the static tier map; a deployed local model outranks all three.
`frugal.tier_source` reports which one decided.

As a bonus, a prompt nearly identical to one already answered can reuse the answer
outright — a 100% saving with no model call at all.

```yaml
embeddings:
  enabled: false
  # base_url / api_key default to the openrouter section; OpenRouter serves
  # POST /embeddings. Set them to use OpenAI or a local server instead.
  model: openai/text-embedding-3-small
  dimensions: 256
  neighbors: 10
  min_similarity: 0.75
  min_neighbors: 5 # a sparser neighbourhood defers to the skill matrix
  max_vectors: 2000
  cache_enabled: false
  cache_similarity: 0.97
```

Off by default: it adds an embedding call — latency, and a cost not counted in
`frugal.total_cost_usd` — to every request, while the skill matrix and the local model
already capture most of the win and cost nothing per request. Turning it on needs no
extra setup with OpenRouter, since it uses the same base URL and key; point
`embeddings.base_url` elsewhere (OpenAI, or a local server such as llama.cpp or
text-embeddings-inference) and drop the `openai/` prefix from the model name if the server
expects bare ids. `dimensions` is passed through to shrink the vectors; a provider that
ignores it returns full-size ones, which still work but make the scan slower. The
neighbour scan is brute-force cosine in pure Python over the most recent `max_vectors`
prompts — fine for a few thousand short vectors. A vector index is the upgrade path if the
log outgrows it.

Note that `cache_enabled: true` stores answer text in the database, independent of
`log_prompts`.

### Local routing model

The skill matrix answers "how often does Haiku pass hard code tasks?". The local model
answers "will Haiku pass *this* prompt?" It is a logistic regression per (judge threshold,
ladder rung) over hashed word n-grams plus a few structured features, trained on the request
log. It needs no GPU and no extra dependencies, and asks no API: routing a request is a
sparse dot product, microseconds. Its pass probabilities feed the same expected-cost
recursion as the matrix, priced at today's OpenRouter prices.

**Using it.** Data collection is on by default (`predictor.log_features`). Once a few
thousand requests have gone through, train:

```bash
python -m app.train
```

It trains a model on the older 80% of the log, evaluates it on the newest 20%, prints a
report, then retrains on all of it and writes `router_model.json` (`--no-refit` saves the
evaluated model instead). A running server picks the file up within ~30 seconds — no
restart — and routes with it only if the report says **DEPLOYABLE**. Run it on a schedule
(cron, for example) and the router keeps improving as traffic accumulates. `--dry-run`
evaluates without writing; `--json` is for scripts; `--force` trains below
`min_examples`. `GET /predictor` shows
whether the model is routing and how it was evaluated; `frugal.tier_source` is `predictor`
and `frugal.predicted_pass` holds its per-rung probabilities on requests it routed.

**Why it can be trusted with your money.** The log only records outcomes for rungs a
request actually tried, so every evaluation has to reason about outcomes nobody saw. The
design is built around not fooling itself about those:

- **Trust region.** The model may start a request at most **one rung below** where the
  skill matrix (or config) would. That is exactly how far exploration reaches, so every
  downward move it makes is one the log has real outcomes for — and a wrong model costs at
  most one wasted cheap attempt per request, the bound exploration already accepts.
  Moving *up* is unrestricted: it costs money, never quality.
- **Honest evaluation.** For moves one rung down, the replay uses a doubly robust
  estimator: the model's prediction, corrected by the explored requests (an unbiased
  coin-flip sample) weighted by 1/ε. Its expectation is the true value whatever the model
  believes. The exploration probability is recorded with each request, so changing
  `epsilon` later doesn't corrupt old evidence.
- **Proven, not estimated.** It deploys only if its saving over the baseline router is
  positive at a 95% lower confidence bound, losing at most a point of success rate — either under a
  worst case that charges every downward move as a failure (needs no exploration data), or
  under the doubly robust estimate backed by at least 20 explored requests.
- **No regressions.** A retrain that fails the check never replaces a model that passed
  one; the candidate is written to `router_model.candidate.json` instead (`--replace`
  overrides). Without this the router flapped in testing: a blocked retrain dropped routing
  back to config, exploration stopped reaching the cheap rungs, and the next retrain had
  less evidence than the last.
- **Never stale.** A model trained for a different ladder, feature scheme or bucket count
  is ignored, and routing falls back to the next source in line.

**What to expect.** In simulation against known ground truth (prompts whose words, not
just their classifier label, decide which model they need), a model trained on 6,000
requests cut costs 14–19% versus routing without it (static tiers or skill matrix),
recovering most of the available headroom, and its estimate tracked the truth to within
~3 points. Run through the real server for several retrain cycles, it moved easy prompts from the third rung to the cheapest
(−92% cost on those) while sending hard prompts straight to a capable model — which also
raised their success rate, because they no longer climbed past cheap models a noisy judge
sometimes waves through. Across 36 simulated scenarios, including ones built to make the
model useless or stale, it never deployed a model that lost money. Limits worth knowing:

- **It needs volume.** Expect thousands of requests before it deploys, and it walks down
  at most one rung per retrain cycle. That is deliberate, not a bug.
- **Its estimate is approximate.** With 1/ε weights the evidence is heavy-tailed; in the
  stale-model scenarios the reported saving overstated a real (positive) saving by up to
  ~22 points. The deployment decisions held; the headline number deserves some salt.
- **It is only as good as its labels** — the judge, verifiers and feedback above.
- **Changing the ladder means retraining**, and models trained on the old ladder stop
  routing until you do.

```yaml
predictor:
  enabled: true
  path: router_model.json
  log_features: true # hashed features, never prompt text
  buckets: 8192
  policy: expected_cost # or: pass_rate
  min_examples: 200
  holdout: 0.2
  lookback_days: 90
```

Exploration matters more once the model is on: `exploration.epsilon` (default 0.10) is the
rate at which it gathers the evidence for its next downward step.

### Privacy

Prompt text is **not** logged by default — only its length, plus the task type and
difficulty labels, and (for the local model) hashed word n-grams. Hashed features are not
the prompt, but they are not a privacy guarantee either: someone with a vocabulary could
test which words occurred. Set `predictor.log_features: false` to turn them off.

Other settings that store more: `log_prompts: true` keeps the (truncated) prompt text, for
inspecting traffic yourself — nothing else needs it. With embeddings enabled, each prompt's
embedding vector is stored, and `embeddings.cache_enabled: true` also stores answer text.

## Endpoints

- `POST /v1/chat/completions` — OpenAI-compatible chat
- `GET /v1/models` — the `auto` variants + the ladder models
- `GET /stats` — requests, cost, savings, escalation rate, explored requests, cache hits, per-model and per-task-type distribution, score distribution, skill matrix, verifier outcomes, judge calibration, feedback totals. Accepts `?since=7d` (`h`/`d`/`w`, or an ISO timestamp) to window the numbers.
- `GET /learn` — the skill matrix, the start rungs it implies, and the state of every learning subsystem
- `POST /learn/refresh` — recompute the matrix and neighbour index from the log now
- `POST /v1/feedback` — `{"request_id": ..., "ok": bool}` ground truth from the client
- `GET /predictor` — the local routing model: whether it routes, and its evaluation
- `POST /predictor/reload` — load a freshly trained model now
- `GET /healthz` — health check (no auth)

All requests are logged to SQLite (`db_path` in config): one row per request, one per
attempt, plus `audits`, `feedback`, `prompt_features` and `prompt_vectors`. An existing
database is migrated in place on startup.

## Error handling

- OpenRouter 429/5xx: retry with exponential backoff (max 2), otherwise the model is skipped in the ladder
- Classifier down → defaults to `medium` / `other`; judge down → the answer passes through (fail open)
- Learning fails open too: a missing or unusable local model, a failed matrix refresh or a
  failed embedding call each hand the decision to the next routing source
- The service degrades, it doesn't die

## Calibration

The judge is the bottleneck: too lenient lets bad answers through, too strict eats up the savings. Watch `score_distribution` in `GET /stats` and adjust `judge.threshold` against real-world usage.

This matters more now that routing learns from judge scores: the learned pass rates are
only as honest as the judge is. Rather than trusting it, measure it — turn on `audit` and
watch `judge_calibration.false_accept_rate` in `/stats`. A rate that climbs on one task
type (math is the usual suspect) says the cheap judge is the wrong instrument there, and
that raising `judge.threshold` will buy less than switching `judge.model` for that traffic.
See [Keeping the numbers honest](#keeping-the-numbers-honest).

## Tests

```bash
pytest
```
