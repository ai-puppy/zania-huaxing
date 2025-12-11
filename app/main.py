import uuid

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory

from app.database import get_async_connection
from app.schemas import ChatRequest, ChatResponse, HistoryResponse, MessageOut

app = FastAPI(title="Zania Q&A API")


@app.get("/")
async def root():
    return {"message": "Zania Q&A API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response. Creates new session if not provided."""
    session_id = request.session_id or str(uuid.uuid4())

    async with get_async_connection() as conn:
        history = PostgresChatMessageHistory(
            "langchain_chat_histories",
            session_id,
            async_connection=conn,
        )

        # Add user message to history
        await history.aadd_messages([HumanMessage(content=request.message)])

        # TODO: Replace with actual LLM call
        ai_response = f"Echo: {request.message}"

        # Add AI response to history
        await history.aadd_messages([AIMessage(content=ai_response)])

    return ChatResponse(session_id=session_id, response=ai_response)


@app.get("/chat/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Get chat history for a session."""
    async with get_async_connection() as conn:
        history = PostgresChatMessageHistory(
            "langchain_chat_histories",
            session_id,
            async_connection=conn,
        )

        messages = await history.aget_messages()

        if not messages:
            raise HTTPException(status_code=404, detail="Session not found")

        return HistoryResponse(
            session_id=session_id,
            messages=[
                MessageOut(
                    role="user" if isinstance(msg, HumanMessage) else "assistant",
                    content=msg.content,
                )
                for msg in messages
            ],
        )
