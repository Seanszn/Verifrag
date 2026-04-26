# Query Expansion Test Plan

## Goal

Measure whether query expansion improves retrieval for broad legal research questions without overfitting to one client's database. The target behavior is higher recall of relevant documents while keeping the answer verifier conservative: broad issue matches should usually be labeled possible support unless the retrieved text directly states the answer.

## Modes Under Test

- Raw query: send the user's query unchanged.
- Rule expansion: add deterministic legal synonyms, posture terms, and authority-seeking terms.
- LLM expansion: generate 3-5 alternate phrasings, fact-pattern terms, and legal concept terms without adding case names or citations not present in the query.
- Corpus-aware expansion: add terms derived from client-approved metadata facets such as jurisdiction, practice area, statute name, docket type, and document type.
- Hybrid expansion: combine rule and LLM expansions, then deduplicate and cap total terms.
- Reranked expansion: retrieve with expanded queries, merge candidates, then rerank against the original query.

## Query Families

- Legal concept: asks for examples of a doctrine without naming a case.
- Fact pattern: describes facts and asks what authorities may be relevant.
- Procedural posture: asks for cases involving standing, certiorari, injunctions, dismissal, summary judgment, or remand.
- Statute or rule: names a statute, regulation, rule, or constitutional provision but no case.
- Remedy or consequence: asks about damages, penalties, exclusion, suppression, sanctions, or remedies.
- Ambiguous broad query: contains terms that plausibly map to multiple doctrines.
- Negative control: asks for a topic not represented in the corpus.
- Uploaded-document comparison: asks for related authority from the public corpus, tested only when uploaded chunks are explicitly included.

## Dataset Construction

- For each client corpus, sample 30-50 seed documents across practice areas and jurisdictions.
- For each seed, write 2-4 queries that omit the case name, citation, and unique party names.
- Build a relevance set at document level first, then chunk level where practical.
- Include near-miss distractors that share vocabulary but not the legal issue.
- Hold out synonym lists by client or matter so expansion rules are not tuned to one database.
- Keep a versioned manifest with corpus id, document ids, expected relevant documents, and query family.

## Metrics

- Document recall at 5, 10, and 20.
- Chunk recall at 10 and 20.
- Mean reciprocal rank for the first relevant document.
- Distinct-case precision at 10 to detect noisy broad expansion.
- No-result accuracy for negative controls.
- Expansion drift rate: percent of expansions that introduce unsupported case names, citations, parties, or statutes.
- Verifier ceiling: percent of exploratory answers whose strongest label is no higher than possible support unless direct text supports a stronger label.
- Latency and token cost per mode.

## Experiment Design

1. Run raw-query retrieval as the baseline.
2. Run each expansion mode with the same retrieval depth, merge policy, and reranker settings.
3. Compare per-query-family gains, not just aggregate gains.
4. Inspect failures where recall improves but precision collapses.
5. Repeat on at least two corpora with different subject matter before accepting a mode as default.
6. Promote a mode only if it improves recall without materially increasing hallucinated authority or latency.

## Acceptance Gates

- Improves document recall at 10 by at least 10 percent relative over raw query on broad case-finding queries.
- Does not reduce distinct-case precision at 10 by more than 5 percent relative.
- Expansion drift stays below 2 percent of generated expansions.
- Negative-control no-result accuracy does not drop by more than 5 percent absolute.
- P95 retrieval plus expansion latency remains within the product target for interactive use.

## Logging Needed

- Original query.
- Selected answer mode.
- Expanded query variants.
- Expansion source: rule, LLM, corpus-aware, or hybrid.
- Retrieved chunk ids and document ids per variant.
- Merged and reranked candidate ids.
- Verifier labels for claims made from exploratory retrieval.

## Safeguards

- Do not add case names, citations, or party names unless they appear in the query, uploaded document, or corpus metadata.
- Cap expansion variants and terms per query.
- Preserve the original query for reranking.
- Mark research-leads responses as exploratory.
- Treat broad analogies as possible support unless a retrieved passage directly states the asserted rule or fact.
