# LegalVerifiRAG Coding Standards

This document makes the repository's implicit engineering conventions explicit.
It is based on the current codebase, with the cleaner modules treated as the
reference style.

## Scope

These standards apply to:

- `src/`
- `tests/`
- `scripts/`

## Baseline

- Target Python is `3.10+`.
- New code should be fully type-annotated.
- New behavior should ship with tests unless the code is pure scaffolding and is
  clearly marked as such.
- Readability is preferred over dense abstraction.

## File Structure

- Start each Python module with a short module docstring.
- Keep files focused on one primary responsibility.
- If a file starts mixing transport, parsing, persistence, orchestration, and
  CLI concerns, split it.
- Prefer package boundaries that match the current structure:
  `api`, `auth`, `client`, `generation`, `indexing`, `ingestion`,
  `retrieval`, `storage`, and `verification`.

## Typing

- Use modern typing syntax in new and edited code:
  `list[str]`, `dict[str, Any]`, `str | None`.
- Add return types to public functions, methods, and non-trivial helpers.
- Use dataclasses for internal domain models.
- Use Pydantic models for API request and response schemas.
- Avoid mixing multiple typing styles in the same file.

## Functions And Classes

- Prefer small, single-purpose functions.
- Extract helpers when a function becomes hard to scan or mixes unrelated work.
- Public classes and functions should have concise docstrings.
- Constructors should validate important invariants early.
- Abstract base classes should document expected inputs and outputs clearly.

## Error Handling

- Do not use broad `except Exception` unless there is a clear boundary where the
  code must prevent a crash.
- When broad exception handling is justified, log enough context to debug it and
  keep the protected block narrow.
- Prefer specific exception types over silent fallback behavior.
- Do not swallow errors with bare `pass` unless the failure is genuinely
  irrelevant and documented.

## Imports

- Group imports in this order: standard library, third-party, local.
- Keep imports at module scope unless a local import is needed to avoid a cycle
  or optional dependency cost.
- Avoid unused imports.

## Logging And CLI Output

- Use `logging` in library and application code.
- `print()` is acceptable in CLI scripts for progress and summaries.
- Keep log messages factual and actionable.

## Comments And Docstrings

- Write comments only when the code's intent is not obvious from the code.
- Avoid decorative separator comments and non-essential Unicode characters that
  render poorly in some Windows terminals.
- Prefer comments that explain why, not what.

## Tests

- Tests should describe behavior, not implementation details.
- Prefer straightforward fixtures and helpers over complex indirection.
- Placeholder tests should not remain ambiguous:
  either implement them, mark them as skipped with a reason, or remove them.

## Scripts

- Scripts should expose a clear `main()` entry point.
- Use `argparse` for user-facing scripts.
- Keep script orchestration separate from reusable library logic.

## TODO And Scaffolding Policy

- Do not leave stub modules that look production-ready unless they are clearly
  identified as scaffolding.
- If a module is intentionally incomplete, make that explicit in its docstring
  and, where appropriate, raise `NotImplementedError`.
- Prefer tracking unfinished work in a cleanup plan instead of scattering bare
  TODO files across core packages.

## Review Checklist

Before merging a change, check:

- Is the file still single-purpose?
- Are public interfaces typed?
- Is error handling specific and debuggable?
- Is the module docstring still accurate?
- Are tests present and readable?
- Does the code match the dominant style already used nearby?
