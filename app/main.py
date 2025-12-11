import os
import uuid

from fastapi import FastAPI, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory

from app.database import get_async_connection
from app.document_loader import (
    chunk_documents,
    load_document,
    load_questions,
    save_upload_file_temp,
)
from app.rag_chain import answer_questions, create_qa_chain, create_vector_store
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    MessageOut,
    QAResponse,
)

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


@app.post("/qa", response_model=QAResponse)
async def process_qa(
    questions_file: UploadFile,
    document_file: UploadFile,
):
    """
    Process Q&A request.

    Upload a questions file (JSON or CSV) and a document file (PDF or JSON).
    Returns answers for each question based on the document content.
    """
    questions_suffix = os.path.splitext(questions_file.filename or "")[1]
    document_suffix = os.path.splitext(document_file.filename or "")[1]

    questions_path = await save_upload_file_temp(
        questions_file, suffix=questions_suffix
    )
    document_path = await save_upload_file_temp(document_file, suffix=document_suffix)

    try:
        questions = load_questions(questions_path)
        if not questions:
            raise HTTPException(status_code=400, detail="No questions found in file")

        documents = load_document(document_path)
        if not documents:
            raise HTTPException(status_code=400, detail="No content found in document")

        chunks = chunk_documents(documents)
        vector_store = create_vector_store(chunks)
        qa_chain = create_qa_chain(vector_store)
        answers = await answer_questions(qa_chain, questions)

        return QAResponse(answers=answers)

    finally:
        if os.path.exists(questions_path):
            os.unlink(questions_path)
        if os.path.exists(document_path):
            os.unlink(document_path)
