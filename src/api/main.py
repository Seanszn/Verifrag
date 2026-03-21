"""FastAPI backend for auth, conversations, and query execution."""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, status

from src.api.dependencies import get_database, require_user
from src.api.schemas import (
    AuthResponse,
    ConversationCreateRequest,
    ConversationResponse,
    LoginRequest,
    MessageResponse,
    QueryRequest,
    QueryResponse,
    RegisterRequest,
    UserResponse,
)
from src.auth.local_auth import hash_password, verify_password
from src.pipeline import QueryPipeline
from src.storage.database import Database

app = FastAPI(title="LegalVerifiRAG API", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    get_database().initialize()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    db: Database = Depends(get_database),
) -> AuthResponse:
    existing = db.get_user_by_username(payload.username.strip())
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists.")

    user = db.create_user(
        username=payload.username.strip(),
        password_hash=hash_password(payload.password),
    )
    token = db.create_session(user["id"])
    return AuthResponse(token=token, user=UserResponse(**user))


@app.post("/api/auth/login", response_model=AuthResponse)
def login(
    payload: LoginRequest,
    db: Database = Depends(get_database),
) -> AuthResponse:
    row = db.get_user_by_username(payload.username.strip())
    if not row or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    user = db.get_user_by_id(row["id"])
    assert user is not None
    token = db.create_session(user["id"])
    return AuthResponse(token=token, user=UserResponse(**user))


@app.post("/api/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    user: dict[str, object] = Depends(require_user),
    db: Database = Depends(get_database),
) -> None:
    db.delete_session(str(user["token"]))


@app.get("/api/conversations", response_model=list[ConversationResponse])
def list_conversations(
    user: dict[str, object] = Depends(require_user),
    db: Database = Depends(get_database),
) -> list[ConversationResponse]:
    conversations = db.list_conversations(int(user["id"]))
    return [ConversationResponse(**item) for item in conversations]


@app.post("/api/conversations", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreateRequest,
    user: dict[str, object] = Depends(require_user),
    db: Database = Depends(get_database),
) -> ConversationResponse:
    conversation = db.create_conversation(int(user["id"]), payload.title.strip())
    return ConversationResponse(**conversation)


@app.get("/api/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
def list_messages(
    conversation_id: int,
    user: dict[str, object] = Depends(require_user),
    db: Database = Depends(get_database),
) -> list[MessageResponse]:
    conversation = db.get_conversation(conversation_id, int(user["id"]))
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = db.list_messages(conversation_id, int(user["id"]))
    return [MessageResponse(**item) for item in messages]


@app.post("/api/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    user: dict[str, object] = Depends(require_user),
    db: Database = Depends(get_database),
) -> QueryResponse:
    pipeline = QueryPipeline(db=db)
    result = pipeline.run(
        user_id=int(user["id"]),
        query=payload.query.strip(),
        conversation_id=payload.conversation_id,
    )
    return QueryResponse(**result)
