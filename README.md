# Memory-Augmented Agent for Adaptive Code Review

An agentic AI code-review system built with Gemini 2.5 Flash and LangGraph. The agent can search a persistent memory (ChromaDB for past review comments, SQLite for extracted coding conventions) before reviewing a pull request, and reflects after each review to decide whether to save new conventions.

## How It Works

The agent (`agent_review_graph.py`) is a LangGraph state machine with three stages:
1. **Reasoning** — decides whether to call memory tools (`search_conventions`, `search_similar_reviews`) before finalizing a review
2. **Tools** — executes memory searches against ChromaDB / SQLite
3. **Reflection** — after the review, decides if anything new is worth saving back to memory

Memory itself is hybrid: **ChromaDB** stores raw past review comments for semantic recall, while **SQLite** stores consolidated, reusable conventions extracted from those comments — deduplicated via embedding similarity + an LLM confirmation pass, so repeated conventions reinforce a single entry instead of fragmenting into near-duplicates.

## Key Finding

A controlled evaluation (20 held-out PRs, both agent variants scored side-by-side in the same LLM-judge call to enforce consistent calibration) found the memory-augmented agent scored **lower** than the memory-less baseline:

| Metric | Memory-ON | Memory-OFF |
|---|---|---|
| Overlap with human review | 2.95 / 10 | **4.10 / 10** |
| Actionability | 3.55 / 10 | **6.90 / 10** |

Manual inspection suggests tool-calling **displaces careful diff reading** rather than supplementing it — the agent tends to treat "I checked our conventions" as a stopping point rather than continuing to scrutinize the code. This held even on PRs directly relevant to seeded conventions, ruling out simple irrelevance as the explanation.

The underlying agentic mechanics work correctly: the agent reliably recalls, cites, and applies specific stored conventions, and the reflection loop correctly identifies and saves new patterns without duplicating existing ones. The real finding is that *working* tool-use isn't automatically *helpful* tool-use — sequencing and prompting matter. **Future work:** test whether requiring a full independent read-through before tool access closes this gap.


## Tech Stack

Gemini 2.5 Flash · LangGraph · ChromaDB · SQLite · `sentence-transformers` · GitHub API
