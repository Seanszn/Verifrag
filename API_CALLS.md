# LegalVerifiRAG API Calls

This file documents the HTTP calls currently made by the repository code in `legalverifirag/`.

It covers:

- The app's own FastAPI endpoints
- The Streamlit client's calls into that API
- The backend's outbound calls to Ollama
- The corpus builder's outbound calls to CourtListener

It does not cover database calls, local file I/O, or library-internal network activity that is not directly expressed in the project code.

## Overview

Current call flow:

1. Streamlit client calls the local FastAPI backend.
2. FastAPI handles auth, conversations, and query execution.
3. Query execution calls the local Ollama HTTP API.
4. Corpus download scripts call the CourtListener REST API.

Default base URLs from [`src/config.py`](./src/config.py):

- FastAPI backend: `http://127.0.0.1:8000`
- Ollama: `http://localhost:11434`
- CourtListener: `https://www.courtlistener.com/api/rest/v4/`

## Common Conventions

### Content Type

The Streamlit client sends:

```http
Content-Type: application/json
```

### Auth Header

Protected backend routes require:

```http
Authorization: Bearer <session-token>
```

### Common Error Shapes

Application-defined errors usually look like:

```json
{
  "detail": "Invalid username or password."
}
```

FastAPI validation errors usually look like:

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "password"],
      "msg": "String should have at least 8 characters",
      "input": "short",
      "ctx": {
        "min_length": 8
      }
    }
  ]
}
```

## Internal API: Streamlit Client -> FastAPI

These are the API routes implemented in [`src/api/main.py`](./src/api/main.py) and consumed by the Streamlit client in [`src/client/ui.py`](./src/client/ui.py).

### `GET /health`

Purpose: simple health check for the backend.

Request body: none

Example response:

```json
{
  "status": "ok"
}
```

### `POST /api/auth/register`

Purpose: create a new user and immediately create a session token.

Called from:

- Register form in the Streamlit client

Request body:

```json
{
  "username": "alice",
  "password": "password123"
}
```

Field notes:

- `username`: string, min length 3, max length 64
- `password`: string, min length 8, max length 256

Successful response: `201 Created`

```json
{
  "token": "kW3nBv...example-token...",
  "user": {
    "id": 1,
    "username": "alice",
    "created_at": "2026-03-17T14:22:31.123456+00:00"
  }
}
```

Possible error responses:

- `409 Conflict`

```json
{
  "detail": "Username already exists."
}
```

- `422 Unprocessable Entity`

```json
{
  "detail": [
    {
      "loc": ["body", "username"],
      "msg": "String should have at least 3 characters",
      "type": "string_too_short"
    }
  ]
}
```

### `POST /api/auth/login`

Purpose: authenticate an existing user and create a new session token.

Called from:

- Login form in the Streamlit client

Request body:

```json
{
  "username": "alice",
  "password": "password123"
}
```

Successful response: `200 OK`

```json
{
  "token": "X2m3q...example-token...",
  "user": {
    "id": 1,
    "username": "alice",
    "created_at": "2026-03-17T14:22:31.123456+00:00"
  }
}
```

Possible error responses:

- `401 Unauthorized`

```json
{
  "detail": "Invalid username or password."
}
```

- `422 Unprocessable Entity`

```json
{
  "detail": [
    {
      "loc": ["body", "password"],
      "msg": "String should have at least 8 characters",
      "type": "string_too_short"
    }
  ]
}
```

### `POST /api/auth/logout`

Purpose: invalidate the current session token.

Called from:

- Logout button in the Streamlit sidebar

Headers:

```http
Authorization: Bearer <session-token>
```

Request body: none

Successful response: `204 No Content`

Response body: empty

Possible error responses:

- `401 Unauthorized`

```json
{
  "detail": "Missing bearer token."
}
```

or

```json
{
  "detail": "Invalid or expired token."
}
```

### `GET /api/conversations`

Purpose: list the current user's conversations.

Called from:

- Streamlit app startup after login
- Refresh flow after auth state changes

Headers:

```http
Authorization: Bearer <session-token>
```

Request body: none

Successful response: `200 OK`

```json
[
  {
    "id": 3,
    "user_id": 1,
    "title": "Explain Miranda warnings",
    "created_at": "2026-03-17T14:25:00.000000+00:00",
    "updated_at": "2026-03-17T14:25:04.000000+00:00"
  },
  {
    "id": 2,
    "user_id": 1,
    "title": "Fourth Amendment search basics",
    "created_at": "2026-03-17T14:20:00.000000+00:00",
    "updated_at": "2026-03-17T14:21:10.000000+00:00"
  }
]
```

Possible error responses:

- `401 Unauthorized`

```json
{
  "detail": "Missing bearer token."
}
```

or

```json
{
  "detail": "Invalid or expired token."
}
```

### `POST /api/conversations`

Purpose: create a conversation directly.

Current repository note:

- This route exists in the backend.
- The current Streamlit client does not call it directly.
- New conversations are usually created implicitly by `POST /api/query` when `conversation_id` is `null` or omitted.

Headers:

```http
Authorization: Bearer <session-token>
```

Request body:

```json
{
  "title": "Miranda warning research"
}
```

Field notes:

- `title`: string, min length 1, max length 200

Successful response: `201 Created`

```json
{
  "id": 4,
  "user_id": 1,
  "title": "Miranda warning research",
  "created_at": "2026-03-17T14:30:00.000000+00:00",
  "updated_at": "2026-03-17T14:30:00.000000+00:00"
}
```

Possible error responses:

- `401 Unauthorized`
- `422 Unprocessable Entity`

Example `422`:

```json
{
  "detail": [
    {
      "loc": ["body", "title"],
      "msg": "String should have at least 1 character",
      "type": "string_too_short"
    }
  ]
}
```

### `GET /api/conversations/{conversation_id}/messages`

Purpose: list all messages in one conversation, oldest first.

Called from:

- When a user selects a conversation in the Streamlit sidebar

Headers:

```http
Authorization: Bearer <session-token>
```

Path params:

- `conversation_id`: integer

Request body: none

Successful response: `200 OK`

```json
[
  {
    "id": 10,
    "conversation_id": 3,
    "role": "user",
    "content": "Explain Miranda warnings",
    "created_at": "2026-03-17T14:25:01.000000+00:00",
    "metadata_json": null
  },
  {
    "id": 11,
    "conversation_id": 3,
    "role": "assistant",
    "content": "The court held that Miranda warnings are required.",
    "created_at": "2026-03-17T14:25:04.000000+00:00",
    "metadata_json": "{\"llm_provider\":\"ollama\",\"llm_backend_status\":\"ok\",\"retrieval_used\":false,\"claim_count\":1,\"claims\":[{\"claim_id\":\"clm_b389f3f9c7d6\",\"text\":\"The court held that Miranda warnings are required.\",\"claim_type\":\"holding\",\"source\":\"court\",\"certainty\":\"found\",\"doc_section\":\"body\",\"span\":{\"doc_id\":\"assistant_response\",\"para_id\":0,\"sent_id\":1,\"start_char\":0,\"end_char\":50}}]}"
  }
]
```

Possible error responses:

- `401 Unauthorized`

```json
{
  "detail": "Missing bearer token."
}
```

or

```json
{
  "detail": "Invalid or expired token."
}
```

- `404 Not Found`

```json
{
  "detail": "Conversation not found."
}
```

### `POST /api/query`

Purpose: submit a legal question, create or reuse a conversation, store both the user and assistant messages, and return pipeline metadata.

Called from:

- The Streamlit chat input

Headers:

```http
Authorization: Bearer <session-token>
```

Request body:

```json
{
  "query": "Explain Miranda warnings",
  "conversation_id": null
}
```

Field notes:

- `query`: string, min length 1, max length 8000
- `conversation_id`: optional integer
- If `conversation_id` is missing or `null`, the backend creates a new conversation with an auto-generated title based on the query text.

Successful response: `200 OK`

```json
{
  "conversation": {
    "id": 1,
    "user_id": 1,
    "title": "Explain Miranda warnings",
    "created_at": "2026-03-17T14:25:00.000000+00:00",
    "updated_at": "2026-03-17T14:25:04.000000+00:00"
  },
  "user_message": {
    "id": 1,
    "conversation_id": 1,
    "role": "user",
    "content": "Explain Miranda warnings",
    "created_at": "2026-03-17T14:25:01.000000+00:00",
    "metadata_json": null
  },
  "assistant_message": {
    "id": 2,
    "conversation_id": 1,
    "role": "assistant",
    "content": "The court held that Miranda warnings are required.",
    "created_at": "2026-03-17T14:25:04.000000+00:00",
    "metadata_json": "{\"llm_provider\":\"ollama\",\"llm_backend_status\":\"ok\",\"retrieval_used\":false,\"claim_count\":1,\"claims\":[{\"claim_id\":\"clm_b389f3f9c7d6\",\"text\":\"The court held that Miranda warnings are required.\",\"claim_type\":\"holding\",\"source\":\"court\",\"certainty\":\"found\",\"doc_section\":\"body\",\"span\":{\"doc_id\":\"assistant_response\",\"para_id\":0,\"sent_id\":1,\"start_char\":0,\"end_char\":50}}]}"
  },
  "pipeline": {
    "llm_provider": "ollama",
    "llm_backend_status": "ok",
    "retrieval_used": false,
    "claim_count": 1,
    "claims": [
      {
        "claim_id": "clm_b389f3f9c7d6",
        "text": "The court held that Miranda warnings are required.",
        "claim_type": "holding",
        "source": "court",
        "certainty": "found",
        "doc_section": "body",
        "span": {
          "doc_id": "assistant_response",
          "para_id": 0,
          "sent_id": 1,
          "start_char": 0,
          "end_char": 50
        }
      }
    ]
  }
}
```

Possible error responses:

- `401 Unauthorized`

```json
{
  "detail": "Missing bearer token."
}
```

or

```json
{
  "detail": "Invalid or expired token."
}
```

- `422 Unprocessable Entity`

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "String should have at least 1 character",
      "type": "string_too_short"
    }
  ]
}
```

Behavior note:

- If the Ollama call fails, the route still returns `200 OK`, but the assistant content becomes a fallback error message and `pipeline.llm_backend_status` changes to something like `error:ConnectionError`.

Example degraded-success response excerpt:

```json
{
  "assistant_message": {
    "role": "assistant",
    "content": "The backend could not reach the configured LLM provider. Check Ollama availability and server configuration."
  },
  "pipeline": {
    "llm_provider": "ollama",
    "llm_backend_status": "error:ConnectionError",
    "retrieval_used": false,
    "claim_count": 1
  }
}
```

## Outbound API: FastAPI Backend -> Ollama

These calls are made in [`src/generation/ollama_backend.py`](./src/generation/ollama_backend.py).

### `POST {OLLAMA_HOST}/api/generate`

Default URL:

```text
http://localhost:11434/api/generate
```

Purpose: generate the assistant's legal answer.

Request body:

```json
{
  "model": "deepseek-r1:8b",
  "prompt": "You are a legal assistant... Explain Miranda warnings ...",
  "stream": false,
  "options": {
    "temperature": 0.1,
    "num_predict": 2048
  }
}
```

Field notes:

- `model`: taken from `LLM_MODEL`
- `prompt`: built server-side from the user's query
- `stream`: always `false` in current code
- `options.temperature`: fixed at `0.1`
- `options.num_predict`: uses `max_tokens` override if provided, else default `2048`

Representative successful response:

```json
{
  "model": "deepseek-r1:8b",
  "created_at": "2026-03-17T14:25:03.000000Z",
  "response": "The court held that Miranda warnings are required.",
  "done": true,
  "done_reason": "stop",
  "context": [1, 2, 3],
  "total_duration": 1823000000,
  "load_duration": 12000000,
  "prompt_eval_count": 48,
  "eval_count": 19
}
```

What the backend actually uses:

- Only the `response` field is consumed

Failure modes:

- Connection failure to Ollama
- Non-2xx HTTP response from Ollama
- Invalid/malformed JSON

Current backend behavior:

- Exceptions are caught inside the query pipeline
- The user still gets a `200 OK` from `POST /api/query`
- The assistant message contains a fallback explanation

### `GET {OLLAMA_HOST}/api/tags`

Default URL:

```text
http://localhost:11434/api/tags
```

Purpose: health/reachability check for Ollama.

Request body: none

Representative successful response:

```json
{
  "models": [
    {
      "name": "deepseek-r1:8b",
      "model": "deepseek-r1:8b",
      "modified_at": "2026-03-10T18:00:00Z",
      "size": 4661224676
    }
  ]
}
```

Current repository note:

- This helper exists in code, but it is not currently called by the query route.

## Outbound API: Corpus Builder -> CourtListener

These calls are made in [`src/ingestion/corpus_builder.py`](./src/ingestion/corpus_builder.py), typically through `scripts/download_corpus.py`.

Headers sent:

```http
Accept: application/json
```

Optional auth header when `COURTLISTENER_TOKEN` is set:

```http
Authorization: Token <courtlistener-token>
```

### `GET https://www.courtlistener.com/api/rest/v4/clusters/`

Purpose: paginate through opinion clusters for a court.

Query params used by the project:

- `docket__court`: court id such as `scotus` or `ca9`
- `page_size`: page size, default `20`
- `date_modified__gte`: optional ISO timestamp for incremental sync

Example request shape:

```text
GET /api/rest/v4/clusters/?docket__court=scotus&page_size=20&date_modified__gte=2026-02-01T00:00:00+00:00
```

Representative response:

```json
{
  "count": 1284,
  "next": "https://www.courtlistener.com/api/rest/v4/clusters/?cursor=cD0yMDI2LTAyLTAx...",
  "previous": null,
  "results": [
    {
      "id": 123456,
      "case_name": "Miranda v. Arizona",
      "date_filed": "1966-06-13",
      "citations": [
        {
          "volume": 384,
          "reporter": "U.S.",
          "page": "436"
        }
      ],
      "sub_opinions": [
        "https://www.courtlistener.com/api/rest/v4/opinions/654321/"
      ]
    }
  ]
}
```

What the project uses from this response:

- `results`
- `next`
- Per-cluster fields like `id`, `case_name`, `date_filed`, `citations`, and `sub_opinions`

Repository-specific note:

- The code intentionally does not send an `ordering` parameter.

### `GET <sub_opinion_url>`

Purpose: fetch a specific opinion record referenced by a cluster's `sub_opinions` list.

Example URL:

```text
https://www.courtlistener.com/api/rest/v4/opinions/654321/
```

Request body: none

Representative response:

```json
{
  "id": 654321,
  "cluster": 123456,
  "plain_text": "The court held that Miranda warnings are required before custodial interrogation.",
  "html_with_citations": "<p>The court held ...</p>",
  "html": "<p>The court held ...</p>",
  "html_lawbox": "",
  "html_columbia": "",
  "xml_harvard": ""
}
```

What the project uses from this response:

- First non-empty text field in this order:
  - `plain_text`
  - `html_with_citations`
  - `html`
  - `html_lawbox`
  - `html_columbia`
  - `xml_harvard`

### `GET https://www.courtlistener.com/api/rest/v4/opinions/?cluster_id=<id>`

Purpose: fallback query when `sub_opinions` does not yield usable text.

Example request shape:

```text
GET /api/rest/v4/opinions/?cluster_id=123456
```

Representative response:

```json
{
  "count": 2,
  "next": null,
  "previous": null,
  "results": [
    {
      "id": 654321,
      "cluster": 123456,
      "plain_text": "Opinion text here."
    },
    {
      "id": 654322,
      "cluster": 123456,
      "plain_text": "Concurring opinion text here."
    }
  ]
}
```

What the project uses from this response:

- `results`
- The same opinion text fields described above

Failure modes handled by the project:

- `429 Too Many Requests`
  - The code reads `Retry-After`, sleeps, and retries.
- `5xx` responses
  - The code retries with exponential backoff.
- Other `aiohttp` client errors
  - Retried up to 5 attempts.

## Client Call Map

The current Streamlit client makes these calls:

- Login form -> `POST /api/auth/login`
- Register form -> `POST /api/auth/register`
- Logout button -> `POST /api/auth/logout`
- App load after login -> `GET /api/conversations`
- Selecting a conversation -> `GET /api/conversations/{conversation_id}/messages`
- Sending a chat prompt -> `POST /api/query`

## Notes And Constraints

- `POST /api/conversations` exists but is not used by the current Streamlit UI.
- `GET /health` exists but is not used by the current Streamlit UI.
- `POST /api/query` is the main orchestration endpoint.
- Retrieval is currently reported as `false` in the pipeline metadata returned by the backend.
- The `assistant_message.metadata_json` field stores pipeline metadata as a JSON-encoded string, while the top-level `pipeline` field returns the same information as an object.
