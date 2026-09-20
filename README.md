# Agentic market-research pipeline

Turns one line of text — "a local dog-walking service, Telegram WebApp" — into a
six-section market report grounded in live web search, and then decides for itself
whether the result is good enough to hand to a paying customer.

This is the generalized version of a pipeline that has been running in production as a paid
product: customers pay by card, the report is generated unattended, and a human only sees the
ones the pipeline flags. The repo keeps the machinery and drops the billing integration.

## What it produces

A markdown report with six fixed sections (in Russian, the market it was built for):

1. The field and its key terminology
2. Existing players (domestic + 2–3 global)
3. Pricing and monetization models
4. Customer profile and what they use today
5. Unserved segments and market gaps
6. Positioning recommendations

Sections 2–5 are written against live search results and cite their sources as markdown
footnotes. The report ends with an internal block — verdict, unresolved issues, dead links,
footnotes that came from the model rather than from search — meant for the operator, not the
customer.

## Pipeline

```
brief ──> outline ──> search phrase ──> web search (3 queries × 4 results, sections 2–5)
                                              │
                                              ▼
                                  6 sections written in parallel
                                              │
             ┌────────────────────────────────▼────────────────────────────────┐
             │  per-section critic loop  (max 3 rounds, exits early on pass)    │
             │                                                                 │
             │   critic(section) ──> ok ──────────────> section leaves the loop │
             │        │                                                        │
             │        └─> fix + its own search queries                          │
             │                  │                                              │
             │                  ├─> run those queries (budgeted)                │
             │                  └─> rewrite ONLY this section ──> re-critique   │
             └─────────────────────────────────────────────────────────────────┘
                                              │
                    grammar pass ──> cross-section audit ──> URL + grounding checks
                                              │
                                              ▼
                                   SHIP  ──or──  HOLD (call a human)
```

## The three things worth looking at

**1. The loop is per-section, and it exits on merit, not on a counter.**
The naive version critiques the whole report once and rewrites the whole report once — one
pass, no verification that the rewrite fixed anything. Here each section is judged on its own
and only the failing ones are rewritten; a section leaves the loop the moment its critic
returns `ok`. `RESEARCH_MAX_CRITIC_ROUNDS` is a safety stop, not the exit condition. That
distinction matters: the cheap sections stop burning tokens early, and a stubborn section is
visible as a fact rather than hidden inside an averaged "the report was revised once".

**2. The critic chooses to use a tool, and writes the arguments itself.**
When a section fails because a claim has no source, the critic does not just complain — it
returns its own search queries:

```json
{"verdict": "fix",
 "issues": ["«около 40% пользователей» — процент без источника"],
 "search_queries": ["сервис выгула собак доля рынка 2026", "выгул собак статистика россия"]}
```

Those queries are run against DuckDuckGo and the results are appended to that section's
context before it is rewritten. This is the one place where the model, not the script, decides
that a tool is needed and with what arguments — everything else in the pipeline is a fixed
sequence, and the README would rather say so than oversell it.

**3. The run ends in a verdict, not in a delivery.**
Deterministic checks (sections that never passed, missing or too-short sections, dead footnote
URLs, footnotes whose domain never appeared in any search result, cross-section
contradictions) are combined with an editor model's judgement. `HOLD` is written into the
report's internal block, printed to stderr, optionally written to `$RESEARCH_VERDICT_FILE`,
and returned as **exit code 3** so the calling system can hold delivery and page a human.
A pipeline that can refuse to ship its own output is the difference between a demo and
something you can put a payment form in front of.

## Is this "agentic"?

Partly, and the honest answer is more useful than the buzzword:

| Property | Here |
|---|---|
| Multi-step plan with specialized roles (writer / critic / fixer / editor) | yes |
| Retrieval grounding with citations, plus a check that citations came from retrieval | yes |
| Self-evaluation that changes control flow (loop exits on the judge's verdict) | yes |
| Model-chosen tool invocation with model-authored arguments | yes, in one place (re-search) |
| Model-chosen *plan* — deciding the order of stages, or skipping them | no, the sequence is fixed |
| Open-ended tool catalogue the model picks from | no, one tool: web search |
| Memory across runs | no, every run starts cold |

So: an evaluator–optimizer workflow with one real tool-choice point and a quality gate — not
an autonomous agent, and it does not need to be one to be worth money.

## Running it

```bash
cp .env.example .env    # add LLM_API_KEY (any OpenAI-compatible endpoint)
docker build -t research-pipeline:latest .
./run.sh "локальный сервис выгула собак" > report.md; echo "exit=$?"   # 0 = ship, 3 = hold
```

Without Docker: `pip install "openai==2.38.0" "ddgs==9.14.4"` and
`python pipeline.py "..." > report.md`.

Defaults point at Yandex AI Studio with DeepSeek V3.2 (that is what production uses, for
Russian-language quality and domestic hosting); set `LLM_BASE_URL` and `LLM_MODEL` for OpenAI,
DeepSeek direct, OpenRouter, or a local vLLM.

## Tests

```bash
python tests/test_pipeline.py
```

Six control-flow cases run offline — no API key, no network, no LLM — by substituting scripted
critic verdicts:

```
PASS | all sections pass first round: verdict=SHIP searches=12
PASS | critic orders a re-search, then passes: verdict=SHIP searches=14
PASS | section never passes → hold: verdict=HOLD reasons=['разделы [2] не прошли критика за 3 круга']
PASS | garbage critic answer → treated as fix: verdict=SHIP
PASS | editor gate says hold: verdict=HOLD reasons=['редактор: цифры выглядят выдуманными']
PASS | search budget caps extra searches: verdict=SHIP searches=14
```

The fourth case is the one that matters in review: an unparseable critic answer is treated as a
failed section, never as a silent pass. Fail-closed, not fail-open.

## Cost and latency

One report is roughly 25–40 LLM calls and 12–20 search queries. End-to-end runtime in
production measured 17–35 minutes for the same brief — the spread comes from the provider, not
from the loop; the loop adds one critic call per section per round. Budget for the worst case,
not the average, if you promise a turnaround.

## Known limits

- No caching between runs; identical briefs pay full price twice.
- Search is search-result snippets only — pages are never fetched, so a fact that lives in the
  body of an article is invisible to the model.
- The critic and the writer are the same base model, which caps how much the critique can
  catch; a different (or larger) model as critic is the obvious next experiment.
- Section order and count are fixed; a brief that deserves a different shape of report will not
  get one.
- Russian-language prompts throughout — translating them is mechanical but not done here.

## Roadmap

- [ ] A real sample report committed as `examples/`, with the internal verdict block intact
- [ ] Cross-encoder rerank of search results before they enter the prompt
- [ ] Per-section cost/latency telemetry in the internal block
- [ ] Critic model separate from writer model

MIT licensed.
