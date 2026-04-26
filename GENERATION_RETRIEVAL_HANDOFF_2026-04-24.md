# Generation And Retrieval Handoff

Date: 2026-04-24

Purpose:
This file is a high-signal handoff for GPT 5.5 or another engineer continuing the generation and retrieval work from this session. It documents:

- what changed in retrieval and generation
- where those changes live in code
- what effect they had on measured results
- what problems still remain
- what to do next

Use this as a map, not as a source of truth over the code. The code paths listed below should be checked directly before making further changes.

## Read This First

1. Do not treat all metrics in this session as directly comparable.
2. Separate the evaluations into three buckets:
   - offline replay with proxy ground truth
   - live API runs with manual claim audits
   - prompt-structure probes for response length and formatting
3. The strongest offline numbers came from controlled replays with target-case proxy labels, not from fully manual live audits.
4. The main retrieval bottleneck has been reduced substantially.
5. The main remaining bottlenecks are generation drift, verifier false positives, and false contradictions.

## Most Relevant Code Paths

- `src/pipeline.py`
  - retrieval merge and prompt scoping
  - follow-up grounding
  - explicit-target metadata retrieval
  - query intent classification
  - sentence-evidence generation context
  - case-posture extraction
  - cert-denial override and safety guard
  - pre-generation refusal
  - verification scope selection
  - supported-claim repair
- `src/generation/prompts.py`
  - plain-text RAG prompt
  - concise vs detailed response rules
  - named-case precision instructions
  - structured case-posture prompt section
- `src/generation/ollama_backend.py`
  - `generate_with_context(...)`
  - `response_depth`
  - Ollama runtime options including `num_ctx`
- `src/retrieval/hybrid_retriever.py`
  - dense + sparse retrieval fusion
- `tests/test_generation_context.py`
  - regression coverage for grounding, intent, prompt chunk selection, cert-denial guard, and verification scoping
- `tests/test_pipeline.py`
  - integration-level pipeline behavior
- `tests/test_prompts.py`
  - prompt-format expectations
- `tests/test_nli_verifier.py`
  - contradiction-margin and support-dominance behavior

## Current Generation/Retrieval State In Code

As of the current checkout, the pipeline has these notable behaviors:

- conversation history is intentionally limited to the last `2` messages
  - `CONVERSATION_CONTEXT_MESSAGE_LIMIT = 2`
- explicit target retrieval now has a metadata-backed path
  - `TARGET_CASE_METADATA_CHUNK_LIMIT = 40`
- merged prompt retrieval is capped
  - `TARGET_CASE_PROMPT_CHUNK_LIMIT = 8`
- generation is sentence-evidence driven for named-case prompts
  - concise mode uses up to `3` answerable sentences
  - detailed mode uses up to `5`
- response repair is active when unsupported claims are too dominant
  - `REPAIR_UNSUPPORTED_RATIO = 0.5`
  - `REPAIR_MAX_CLAIMS = 3`

This means the system is no longer a naive “retrieve chunks and let Ollama freestyle” setup. It now has explicit target recovery, prompt scoping, evidence sentence selection, posture constraints, refusal logic, and a post-verification repair fallback.

## Evaluation Caveats

This session produced several kinds of metrics. They should not be merged casually.

### 1. Offline replay with proxy labels

Used in:

- `artifacts/test_reports/claim_verification_50_query_batch_after_followup_grounding_2026-04-21.md`
- `artifacts/test_reports/claim_verification_50_query_batch_max5_20260421T150912Z.md`
- `artifacts/test_reports/sean_rules_statistics_summary_2026-04-13.md`
- `artifacts/test_reports/sean_rules_canonicalized_statistics_summary_2026-04-13.md`

These are useful for regression tracking on retrieval grounding and claim-support rates, but they are not a substitute for manual live audits. Several of these runs have no meaningful negatives in the proxy ground truth, so specificity and false-positive safety cannot be inferred from them.

### 2. Live API manual audits

Used in:

- `artifacts/test_reports/live_api_mostly_normal_16_analytics_2026-04-21.md`
- `artifacts/test_reports/live_api_normal_vs_edge_full_intent_manual_audit_2026-04-21.md`
- `artifacts/test_reports/live_api_mostly_normal_16_rerun_20260421T183608Z_analysis.md`
- `artifacts/test_reports/ollama_case_targeted_gpu_analysis.md`
- `artifacts/test_reports/metadata_preserved_statistics_summary.md`

These are the most important artifacts for product safety because they show what the live stack did on actual generated responses.

### 3. Prompt-structure probes

Used in:

- `artifacts/test_reports/rag_in_depth_response_probe_2026-04-21.md`
- `artifacts/test_reports/live_api_richer_response_probe_20260421T184058Z.md`
- `artifacts/test_reports/richer_response_max_sentences_followup_2026-04-21.md`

These tell you whether the system can produce longer answers, multiple paragraphs, and bullets. They do not by themselves prove factual quality.

## Chronological Change Log

## 2026-04-07: Plain-Text Prompting And Markdown Cleanup

Main changes:

- Prompting was pushed toward plain-text answers rather than markdown-heavy structure.
- Claim decomposition cleanup was added so markdown artifacts and bullet markers were less likely to become garbage claims.

Where:

- `src/generation/prompts.py`
- `src/verification/claim_decomposer.py`
- audit artifact: `artifacts/test_reports/ollama_markdown_stripping_analysis.md`

Impact:

- Markdown stripping did not materially improve the audited metrics on the saved Ollama dataset.
- Strict accuracy stayed low at `48.98%` with very low recall.
- Lenient accuracy was `75.51%`, but that came with a `15.79%` false-positive rate.

Interpretation:

- Markdown formatting was a nuisance but not the main bottleneck after decomposer cleanup.
- The real issues were target alignment, drift, and verifier calibration.

## 2026-04-12: Metadata Preservation

Main changes:

- Retrieval/generation preserved richer metadata through the pipeline.
- Case names and other metadata survived prompt construction and downstream auditing.

Where:

- pipeline serialization and chunk handling
- audit artifact: `artifacts/test_reports/metadata_preserved_statistics_summary.md`

Impact:

- Top retrieved case matched the target in `10/10` audited queries.
- Off-target confirmed claims by best-case metadata dropped to `0`.
- Strict binary accuracy was still only `51.96%`.
- If `POSSIBLE_SUPPORT` is counted as supported, accuracy rose to `61.76%`, but false positives rose sharply.

Interpretation:

- Metadata preservation improved observability and retrieval targeting.
- It did not solve answer drift or decomposition noise by itself.

## 2026-04-13: Hybrid Retrieval, Case Canonicalization, And RAG Generation

Main changes:

- Added hybrid retrieval with dense + sparse fusion.
- Added reranker path.
- Tightened case targeting and canonicalization.
- Added duplicate-opinion collapse at the case-family level.
- The live pipeline started generating from retrieved context instead of using retrieval only for verification.
- Thresholds were recalibrated so `VERIFIED` became reachable again.

Where:

- `src/retrieval/hybrid_retriever.py`
- `src/pipeline.py`
- audit summary: `artifacts/test_reports/sean_rag_system_changes_summary_2026-04-13.md`

Impact on offline replay:

- `sean_rules_statistics_summary_2026-04-13.md`
  - strict accuracy: `68.51%`
  - lenient accuracy: `81.77%`
- `sean_rules_canonicalized_statistics_summary_2026-04-13.md`
  - strict accuracy: `69.78%`
  - lenient accuracy: `83.52%`

Interpretation:

- Retrieval got strong enough on the offline batch that it stopped looking like the dominant bottleneck.
- The bottleneck shifted toward generation quality, claim decomposition, and verification.

## 2026-04-20: Prompt Filtering Beat Retrieval Boosting

Main changes:

- A focused experiment compared:
  - baseline
  - retrieval boost
  - prompt filter
  - combined

Where:

- summary artifact: `artifacts/test_reports/case_targeting_bm25_only_experiment_6_query_summary.md`

Impact:

- `retrieval_boost` alone did not improve the 6-query drift-prone sample.
- `prompt_filter` improved outcomes from `2 correct / 0 partial / 4 wrong` to `2 correct / 2 partial / 2 wrong`.
- `combined` matched `prompt_filter` and did not beat it.

Interpretation:

- In this sample, the target case was already in retrieval.
- The real failure was that non-target chunks still reached the prompt.
- Prompt-time target-case filtering was more valuable than simply boosting retrieval scores.

## 2026-04-21: Query Grounding And Target Recovery

Main changes:

- Added follow-up grounding from conversation state.
- Added explicit-target metadata retrieval by case name or citation.
- Added short-name target resolution.
- Added confidence-gated issue-only target resolution from convergent retrieval.
- Added query intent classification.
- Added target-case prompt filtering and chunk limiting.

Where:

- `src/pipeline.py`
  - `_ground_followup_query`
  - `_resolve_query_grounding`
  - `_retrieve_target_chunks_from_metadata`
  - `_resolve_short_name_target`
  - `_resolve_convergent_retrieval_target`
  - `_select_prompt_chunks_for_generation`
- tests:
  - `tests/test_generation_context.py`
  - `tests/test_pipeline.py`

Impact on live audit:

- Before these changes were fully in place, `live_api_mostly_normal_16_analytics_2026-04-21.md` showed:
  - strict accuracy: `75.0%`
  - lenient accuracy: `78.1%`
  - 3 dangerous strict false positives
  - failure modes dominated by generic cert-denial outputs, follow-up grounding misses, and case-name normalization misses

Impact on offline 50-query replay:

- `claim_verification_50_query_batch_after_followup_grounding_2026-04-21.md`
  - top-1 canonical match: `100.00%`
  - strict claim support rate: `71.25%`
  - lenient claim support rate: `97.50%`

Interpretation:

- Retrieval targeting and prompt scoping improved substantially.
- On the offline replay, retrieval ceased to be the obvious limiting factor.
- Live safety problems still remained because generation and verification could still mis-handle the retrieved evidence.

## 2026-04-21: Sentence-Evidence Generation Context, Canonical Fact Injection, And Refusal Logic

Main changes:

- Instead of passing arbitrary chunks straight through, named-case prompts now build generation context from answerable sentences.
- The pipeline derives:
  - `explicit_holding_sentence`
  - `canonical_answer_fact`
  - `best_answer_sentence`
  - `answerable_sentences`
- Added pre-generation refusal if the explicit target is not actually retrieved or if the query is off-corpus.

Where:

- `src/pipeline.py`
  - `_build_generation_context`
  - `_select_named_case_evidence_sentences`
  - `_canonical_answer_fact`
  - `_build_pre_generation_refusal`

Impact:

- This improved grounding discipline and created cleaner fallback material for later repair steps.
- It also enabled later generation guards to replace cert-denial garbage with the actual holding sentence or refusal.

Interpretation:

- This was a meaningful architecture improvement even when raw accuracy gains were not always visible immediately.
- It created the data structures the later repair and detailed-answer logic depend on.

## 2026-04-21: Case Posture, Cert-Denial Controls, And Concise Overrides

Main changes:

- Extracted structured case posture from target-case chunks.
- Added a cert-denial override for true cert-denial cases.
- Added a cert-denial safety guard that blocks unauthorized “denied certiorari” language when the evidence/posture does not authorize it.
- Added an explicit-holding override for concise named-case answers.

Where:

- `src/pipeline.py`
  - `_apply_case_posture_response_override`
  - `_apply_explicit_holding_response_override`
  - `_apply_cert_denial_safety_guard`
- `src/generation/prompts.py`
  - `format_case_posture(...)`
  - `_named_case_precision_instructions(...)`

Impact:

- This directly addressed the dominant live failure mode from the mostly-normal audit: generic cert-denial generation on merits cases.
- It reduced some of the worst concise-answer drift, especially for named-case holding/posture prompts.

But:

- Detailed mode intentionally bypasses the concise override.
- That later became one reason richer prompts still drifted.

## 2026-04-21: Detailed Response Support

Main changes:

- Added response-depth classification:
  - concise
  - detailed
- Prompt rules now allow 2 to 4 short paragraphs or simple hyphen bullets in detailed mode.
- Detailed mode uses a larger evidence sentence budget.
- Added supported-claim repair and evidence-based fallback repair after verification.

Where:

- `src/pipeline.py`
  - `_classify_response_depth`
  - `_generation_evidence_sentence_limit`
  - `_apply_supported_claim_repair`
  - `_repair_fallback_evidence_response`
- `src/generation/prompts.py`
  - `_DETAILED_RESPONSE_FORMAT_INSTRUCTIONS`

Impact:

- The first in-depth live probe still failed structurally.
  - `rag_in_depth_response_probe_2026-04-21.md`
  - average words: `23.88`
  - average sentences: `1.12`
  - bullets: `0/8`
- Richer prompting initially caused severe drift.
  - `live_api_richer_response_probe_20260421T184058Z.md`
  - strict accuracy: `58.3%`
  - lenient accuracy: `16.7%`
- Root-cause analysis showed the main reasons:
  - `OLLAMA_NUM_CTX=512` was too small
  - explicit case extraction was too narrow for rich command wording
  - explicit-case retrieval was not forced by metadata
  - detailed mode had no equivalent of the concise holding override
  - verification still allowed some off-topic generic claims through

Relevant artifact:

- `artifacts/test_reports/richer_prompt_failure_root_cause_2026-04-21.md`

Later impact after follow-up fixes:

- `richer_response_max_sentences_followup_2026-04-21.md`
  - live rich prompt set:
    - average words: `159.83`
    - average sentences: `7.25`
    - average paragraphs: `4.00`
    - items with bullets: `11/12`
  - offline replay with `max_sentences=5`:
    - total claims rose from `76` to `98`
    - support rate fell from `92.11%` to `85.71%`

Interpretation:

- The system can now produce longer answers.
- Longer answers increase claim volume and exposure to unsupported claims unless grounding and repair are equally strong.
- Response richness improved faster than factual safety.

## 2026-04-21: Verification Scope Tightening And False-Contradiction Mitigation

Main changes:

- Verification now prefers generation-source chunks and prompt-scoped evidence before broader retrieved context.
- Added low-value/rhetorical claim filtering.
- Added contradiction-margin and support-dominance behavior in the verifier.
- Added named-evidence consistency guards that can block unsupported named-entity hallucinations from being surfaced as supported.

Where:

- `src/pipeline.py`
  - `_select_chunks_for_verification`
  - `_filter_claims_for_verification`
- `src/verification/nli_verifier.py`
- `tests/test_nli_verifier.py`
- `tests/test_pipeline.py`

Impact:

- `false_contradiction_investigation_2026-04-21.md` showed that the dominant cause of false contradictions was broad scoped evidence plus max-contradiction aggregation.
- `post_patch_false_contradiction_reprocess_2026-04-21.md` showed a meaningful improvement:
  - old labels: `{'CONTRADICTED': 8, 'VERIFIED': 2}`
  - new labels: `{'SUPPORTED': 3, 'POSSIBLE_SUPPORT': 3, 'VERIFIED': 1, 'SKIPPED_CAVEAT_OR_CONNECTIVE': 1, 'UNSUPPORTED': 2}`

Interpretation:

- This was a real improvement.
- It reduced one major source of false negatives.
- It did not fully solve verifier reliability, especially on named-entity hallucinations and cross-sentence legal contradictions.

## Result Summary By Change Cluster

| Change cluster | What improved | Evidence | What did not improve enough |
| --- | --- | --- | --- |
| metadata preservation | target-case observability | top retrieved case matched target `10/10` | binary accuracy remained weak |
| prompt filtering / target scoping | reduced cross-case contamination | BM25-only prompt filter outperformed retrieval boost | still did not force correctness when the model drifted |
| follow-up grounding / explicit target recovery | much stronger target resolution on offline replay | 50-query canonical top-1 `100%` | live manual audits still showed false positives |
| sentence-evidence generation context | cleaner grounding inputs and fallback facts | enabled canonical fact injection and repair | does not guarantee the model obeys those facts |
| cert-denial controls | reduced a major generic-generation failure mode | targeted directly at live mostly-normal failure pattern | detailed mode still needed more control |
| detailed response support | much richer answer structure became possible | avg words `159.83`, bullets `11/12` in follow-up rich probe | longer answers lowered support rate |
| contradiction mitigation | reduced false contradiction collapse | 8 contradicted claims reprocessed into mostly non-contradicted labels | some contradiction and false-positive instability remains |

## Metrics That Matter Most

If GPT 5.5 needs a concise summary of the most decision-relevant numbers, use these:

### Retrieval/grounding improvement signals

- metadata-preserved target-case match: `10/10`
  - `artifacts/test_reports/metadata_preserved_statistics_summary.md`
- BM25-only experiment:
  - prompt filter materially helped
  - retrieval boost alone did not
  - `artifacts/test_reports/case_targeting_bm25_only_experiment_6_query_summary.md`
- 50-query replay after follow-up grounding:
  - top-1 canonical match: `100.00%`
  - `artifacts/test_reports/claim_verification_50_query_batch_after_followup_grounding_2026-04-21.md`

### Live manual audit safety signals

- mostly-normal 16 live audit:
  - strict accuracy: `75.0%`
  - lenient accuracy: `78.1%`
  - 3 dangerous strict false positives
  - `artifacts/test_reports/live_api_mostly_normal_16_analytics_2026-04-21.md`
- later rerun analysis on the same general set:
  - strict accuracy: `71.9%`
  - precision: `46.2%`
  - recall: `75.0%`
  - `artifacts/test_reports/live_api_mostly_normal_16_rerun_20260421T183608Z_analysis.md`

Interpretation:

- Retrieval improvements did not automatically translate into safe live support labels.
- Manual audits still show verifier precision problems.

### Rich-response tradeoff signals

- initial rich-response failure:
  - strict accuracy: `58.3%`
  - lenient accuracy: `16.7%`
  - `artifacts/test_reports/live_api_richer_response_probe_20260421T184058Z.md`
- later richer follow-up:
  - average words: `159.83`
  - average paragraphs: `4.00`
  - bullets in `11/12` items
  - but offline support rate dropped as claim count rose
  - `artifacts/test_reports/richer_response_max_sentences_followup_2026-04-21.md`

Interpretation:

- Richness is no longer the blocker.
- Grounded richness is the blocker.

## Persistent Issues

These issues were not fully solved during the session.

### 1. Dangerous verifier false positives still exist

Example:

- Burnett prompt produced Jenkins hallucinations that were marked `VERIFIED`
- documented in `artifacts/test_reports/live_api_answer_blocks_eval_review_2026-04-21.md`

Why this matters:

- This is the most dangerous product failure mode because unsupported claims can be shown as supported.

Current likely cause:

- NLI instability
- named-entity/citation mismatch not being penalized strongly enough
- support being inferred from semantically nearby but wrong legal text

### 2. False contradictions are reduced, not solved

Current cause:

- claims are still sometimes compared against broad same-case text that contains rejected arguments, lower-court reasoning, or separate-opinion material
- max-contradiction aggregation can still overpower direct support in edge cases

### 3. Detailed mode is still more fragile than concise mode

Current cause:

- concise mode can fall back to explicit holdings and cert-denial guards more aggressively
- detailed mode permits more generation freedom
- more generated sentences means more opportunities for unsupported claims

### 4. Longer answers lower support rate unless grounding gets stronger too

Evidence:

- offline replay support rate dropped when `max_sentences` increased from `3` to `5`

### 5. Some decomposition noise still leaks through

Examples seen during the session:

- citation fragments
- connective/rhetorical fragments
- short malformed clauses

This is better than earlier in the session, but not fully eliminated.

### 6. Offline replay is stronger than live API behavior

Why:

- offline replay is more controlled
- live generation is still vulnerable to runtime prompt-window, parser, and drift issues
- proxy labels in replay artifacts do not expose the same safety problems as manual live audits

## What Seemed To Help Most

These were the highest-value changes from this session.

1. Prompt-time target-case filtering
2. Explicit-target metadata retrieval
3. Follow-up grounding from conversation state
4. Sentence-evidence generation context with canonical answer fact injection
5. Cert-denial safety guard plus explicit-holding override
6. Contradiction-margin and support-dominance logic
7. Response repair when unsupported claims dominate

## What Did Not Help Enough On Its Own

1. Markdown cleanup alone
2. Retrieval boosting without prompt filtering
3. Lowering standards by counting `POSSIBLE_SUPPORT` as supported
4. Simply asking for richer answers without tightening grounding

## Recommended Next Steps

Priority order below is deliberate.

### Priority 1: Eliminate dangerous false positives before chasing richer prose

Add a stronger entity/citation consistency gate for support labels.

Concrete rule:

- if a claim names a case, justice, statute, citation, constitutional provision, agency, or party that does not appear in the best supporting evidence, cap the label at `unsupported` or `possibly_supported`

Reason:

- this directly targets the Jenkins/Burnett-type failures

### Priority 2: Make verification evidence-first, not case-wide-first

Current direction is better than before, but continue tightening:

- verify first against generation-source sentences
- then prompt chunks
- only then broader target-case chunks
- do not let a broad same-case contradiction automatically override direct support without a meaningful margin

### Priority 3: Make detailed answers more deterministic

For detailed mode, use an explicit evidence plan:

- opening sentence from canonical answer fact or explicit holding
- one short support paragraph from top answerable sentences
- optional bullet list only from remaining answerable sentences

Avoid asking Ollama to “explain more” without a bounded evidence plan.

### Priority 4: Gate detailed mode by evidence quality

Only allow rich detailed output when there is enough grounded material.

Practical gate:

- enough answerable sentences
- target resolved
- target-case prompt filter applied
- no pre-generation refusal

Otherwise fall back to concise mode.

### Priority 5: Add a fixed verifier sanity set

Create a small stable set of:

- entailment pairs
- unrelated pairs
- true contradiction pairs
- same-case background-vs-holding pairs

Use it to validate the live NLI model directly before changing thresholds again.

### Priority 6: Keep manual live audits as the main acceptance test

Use offline replay for regression speed, but do not accept generation/retrieval changes based only on offline proxy improvements.

## Suggested Starting Plan For GPT 5.5

If continuing this work, the best first pass is:

1. Read:
   - `src/pipeline.py`
   - `src/generation/prompts.py`
   - `tests/test_generation_context.py`
   - `tests/test_pipeline.py`
   - `tests/test_nli_verifier.py`
2. Read these artifacts in order:
   - `artifacts/test_reports/sean_rag_system_changes_summary_2026-04-13.md`
   - `artifacts/test_reports/case_targeting_bm25_only_experiment_6_query_summary.md`
   - `artifacts/test_reports/live_api_mostly_normal_16_analytics_2026-04-21.md`
   - `artifacts/test_reports/false_contradiction_investigation_2026-04-21.md`
   - `artifacts/test_reports/post_patch_false_contradiction_reprocess_2026-04-21.md`
   - `artifacts/test_reports/richer_prompt_failure_root_cause_2026-04-21.md`
   - `artifacts/test_reports/richer_response_max_sentences_followup_2026-04-21.md`
3. First implementation target:
   - entity/citation consistency gate before `SUPPORTED` or `VERIFIED`
4. Second implementation target:
   - stricter detailed-mode evidence plan and fallback
5. Then rerun:
   - mostly-normal live audit
   - richer-response probe
   - fixed verifier sanity set

## Do Not Re-Learn These Lessons

- Do not assume retrieval is the main problem anymore.
- Do not assume `POSSIBLE_SUPPORT` is safe to auto-promote.
- Do not assume richer prompts help unless prompt scoping and context budget are under control.
- Do not assume offline proxy accuracy means live product safety is acceptable.
- Do not optimize only for answer length.

## Final Bottom Line

The session materially improved target-case retrieval, prompt scoping, follow-up grounding, and generation discipline. Those changes made the system much less likely to answer from obviously wrong cases and made concise answers more controllable.

However, the session did not fully solve the most important product-risk issue: false supported labels on bad claims. The next phase should focus on verifier precision and evidence/entity consistency before pushing response richness much further.
