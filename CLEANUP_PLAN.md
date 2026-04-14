# LegalVerifiRAG Cleanup Plan

This plan prioritizes readability and maintainability work identified from the
current codebase.

## P0: Make Scaffolding Explicit

Goal: remove ambiguity about what is implemented versus planned.

Files:

- `src/ingestion/chunker.py`
- `src/verification/citation_verifier.py`
- `src/verification/contradiction.py`
- `src/verification/verdict.py`
- `scripts/evaluate.py`
- `tests/test_pipeline.py`
- `tests/test_chunker.py`
- `tests/test_citation_verifier.py`
- `tests/test_llm_backends.py`

Actions:

- Replace bare TODO-only files with one of:
  implemented code, a clear scaffolding docstring, or `NotImplementedError`.
- Mark placeholder tests as skipped with a reason if they must remain.
- Update README status notes when a module is intentionally incomplete.

Why first:

- These files make the repository look more complete than it is.
- They waste review time because a reader must open each file to learn it is a
  stub.

## P1: Split `corpus_builder.py` By Responsibility

Goal: reduce the largest readability hotspot in implemented code.

Current problems:

- One file owns HTTP client behavior, sync orchestration, parsing, JSONL write
  logic, and sync-state persistence.
- Several broad exception handlers hide failure modes.
- The file is harder to test in focused units.

Suggested split:

- `src/ingestion/courtlistener_client.py`
- `src/ingestion/corpus_sync.py`
- `src/ingestion/corpus_storage.py`
- Keep document parsing helpers close to document models or a dedicated parser
  module if reused.

Required changes:

- Replace broad `except Exception` with specific exceptions where possible.
- Remove silent `pass` for date parsing or document why it is acceptable.
- Type the progress callback explicitly.

## P2: Normalize Typing Style

Goal: make type syntax consistent across the repository.

Files to update first:

- `src/config.py`
- `src/ingestion/document.py`
- `src/indexing/base_store.py`
- `src/generation/base_llm.py`
- `src/verification/nli_verifier.py`

Actions:

- Move toward built-in generics and `|` unions in touched files.
- Keep one style per file.
- Avoid mixing `Optional[str]` with `str | None` in the same module.

## P3: Add Lightweight Tooling

Goal: enforce the standards mechanically instead of relying on memory.

Recommended additions:

- `pyproject.toml` for shared tool configuration
- `ruff` for linting and import cleanup
- `pytest` config
- `.editorconfig`

Suggested initial rules:

- import order
- unused imports
- consistent type syntax in edited files
- obvious readability issues such as unreachable code and bare excepts

## P4: Align Tests With Actual Functionality

Goal: make test coverage match the real feature surface.

Actions:

- Expand tests around implemented modules first:
  API flow, pipeline, storage, retrieval, PDF parsing, and claim decomposition.
- Remove or clearly mark placeholder suites until the underlying modules exist.
- Prefer behavior-level tests like the current API and vector-store examples.

## P5: Reduce Minor Readability Drift

Goal: clean up small inconsistencies that add friction over time.

Actions:

- Normalize import grouping.
- Avoid decorative Unicode separators in source comments.
- Keep script output conventions consistent.
- Add short docstrings where public helpers are currently implicit.

## Definition Of Done

A cleanup item is done when:

- the file's purpose is obvious from its name and docstring,
- incomplete code is clearly marked,
- error handling is specific,
- tests reflect actual behavior,
- and the file matches the standards in `CODING_STANDARDS.md`.
