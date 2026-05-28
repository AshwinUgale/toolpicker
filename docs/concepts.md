# Concepts

The five moving parts: BM25, semantic embeddings, intent classification, RRF fusion, and token-budget packing. Each does one thing; the picker composes them.

---

## BM25 — lexical retrieval

BM25 is the standard probabilistic ranking function for sparse retrieval. It scores a query against a "document" (in our case, a tool's `name + parameter names + keywords + description`) by summing per-term contributions weighted by inverse document frequency. ToolPicker ships an in-house BM25 (~80 LOC) so the core install stays zero-dep.

When BM25 wins:

- The query mentions a tool's parameter name verbatim ("look up the order for BAN 989678111" → `get_order_by_ban`).
- The query shares content vocabulary with the description ("read the file at /etc/hosts" → `read_file`).
- You're running offline or in CI without an API key.

When BM25 loses:

- The query uses synonyms or paraphrases ("ping the team" doesn't textually match "Send an email message to a recipient").
- The query asks about an intent rather than naming the action ("classify videos of cats" against a corpus of model descriptions).

### Stopword filtering

v0.5 added a curated stopword set (`DEFAULT_STOPWORDS`) applied at the BM25 tokeniser, after we noticed RRF was being pulled toward file-tool descriptions because of common words like `"a"`, `"at"`, `"in"`. Stopword filtering applies to both the index and the query, so it doesn't change what scores match — it just removes uninformative noise. Override with `ToolPicker(..., bm25_stopwords=frozenset())` to disable, or pass a custom set.

---

## Semantic — embedding-based retrieval

The semantic half embeds each tool's description once at construction time, embeds the query at `select()` time, and ranks by cosine similarity. Two embedders ship:

- **`OpenAIEmbeddings`** — real `text-embedding-3-small` (1536-d, the default) or `-3-large` (3072-d). Auto-batches at 2048; uses the OpenAI Python SDK.
- **`HashEmbedder`** — deterministic test double. Same vector for the same input every time. Not semantic; used in tests and as a no-key fallback when you want a reproducible-but-not-meaningful baseline.

Wrap either in `CachedEmbedder` to persist results to disk between runs. Cache keys are `sha256(text)`, so identical inputs hit immediately and the cache is safe to share across machines.

When semantic wins:

- The query phrases something the tool description doesn't lexically mention but means the same thing ("what's the temperature in SF?" against "Get the current weather for a city.").
- The corpus is small enough that one cosine pass is faster than your network round-trip to a vector DB (under ~10k tools is plenty).

When semantic loses:

- The query is content-free or stopword-heavy (semantic gives every weather tool a similar score, no discriminator).
- The query contains a distinctive token that lives in a parameter name BM25 already found.

---

## Intent — example-trained classification

The intent classifier answers a different question than the other two retrievers: "what other queries that I've seen before look like this query, and which tool did they map to?" It needs labelled training examples — `IntentExample(query="...", tool_id="...")` — that users supply at construction time. ToolPicker ships zero training data and zero default labels; the corpus is yours to own.

The reference implementation, `EmbeddingNNIntent`, is a k-NN classifier over embedded training queries. At construction it embeds every example once. At `classify(query, k=K)` it embeds the query, computes cosine similarity against each example, takes the top `neighbours` (default 5), and aggregates per `tool_id` by **sum** of similarities. Sum-aggregation makes voting natural: three of five nearest examples mapping to `send_email` gives ~3× the signal of a single isolated match.

When intent wins:

- Your queries are indirect or domain-specific in ways tool descriptions don't capture ("ping the team" → `send_email`, "block off the afternoon" → `create_calendar_event`).
- You have a backlog of call logs or user-curated examples — turn them into `IntentExample`s for free.

When intent loses (or doesn't help):

- You have no labelled examples. Intent classification with an empty corpus returns `[]`.
- Your query distribution differs sharply from the training-example distribution. Intent is a memorised look-up, not a generalisation engine.

---

## RRF — Reciprocal Rank Fusion

When you have multiple rankings of the same items and you want one combined ranking, Reciprocal Rank Fusion is the standard cheap solution. For each item it computes:

```
RRF_score(item) = sum over rankings: weight / (k_constant + rank_of_item_in_ranking)
```

`k_constant=60` is the value the original paper recommends and we keep it as the default. Items that don't appear in a ranking get a 0 contribution from that ranking.

RRF cares about **rank order, not absolute score**. That's why you can fuse BM25 (where scores are unbounded TF-IDF aggregates) with semantic (where scores are cosines in [-1, 1]) with intent (where scores are summed cosines) without normalising anything. The trade-off: a ranker that's confidently right at rank 1 with a huge score margin gets the same RRF contribution as a ranker that's barely right at rank 1.

ToolPicker over-fetches each ranker at 4× the final `k` so RRF has more "votes" to work with. Going past 4× has diminishing returns and adds latency on the semantic side (more cosine computations).

### Weighting

`bm25_weight`, `semantic_weight`, and `intent_weight` are all 1.0 by default. The honest reason: uniform weights are reproducible, and one corpus is not enough evidence to ship "tuned" defaults. The `evals/compare.py` runner exists precisely to let you test what weighting buys you on your corpus.

---

## Token-budget packing

LLM context windows aren't free. Serialising a tool's name + description + parameter schema into the OpenAI function-call envelope eats ~100–300 tokens per tool. Forty tools easily lands at 8k tokens — meaningful in any system prompt.

`select(query, *, k, token_budget=N)` packs the top-K candidates greedy first-fit:

1. Take candidates in rank order (after RRF).
2. For each, compute the serialised token cost.
3. Include it if it fits the remaining budget; otherwise skip and continue to the next candidate.

The "skip and continue" matters: a single oversized tool at rank 3 doesn't blow your budget and leave you with 2 tools; it gets dropped and you keep filling with smaller tools further down. Returned list is bounded by `min(k, number_that_fit)`.

Token counting uses `tiktoken` (`cl100k_base` encoding) when available, falling back to a `ceil(len(json) / 4)` approximation when it isn't. The fallback over-estimates slightly, which is the right direction to err in for a budget.

---

## The v0.6 finding

On both the 200-case in-repo synthetic corpus and a 500-case Gorilla slice, **pure semantic (`semantic-only`) beats every hybrid configuration under uniform-weight RRF**. Adding intent narrows the gap on synthetic (0.800 → 0.845 p@1) but doesn't close it.

Two reasons:

1. **OpenAI embeddings already capture intent reasonably well.** "Schedule a meeting tomorrow" embeds close to "Create a new event on the user's calendar" because the embedding model has seen those associations everywhere on the web.
2. **Uniform RRF treats noisy and confident rankings the same.** A BM25 ranker that confidently puts the wrong tool at rank 1 contributes the same RRF mass as a semantic ranker that confidently puts the right tool at rank 1. Their contributions cancel.

What that means for your routing decision: if your tool descriptions are decent natural language and you have OpenAI access, semantic-only is a respectable baseline. Add BM25 when you need lexical-token matches (account numbers, file paths, error codes). Add intent when you have a backlog of labelled examples and your queries are systematically indirect. The combo isn't free — it has more failure modes, not fewer.

Learned RRF weights (logistic regression over per-query rank vectors) is the natural next experiment and is on the v0.8+ roadmap.
