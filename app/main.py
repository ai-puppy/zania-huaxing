import logging
import os

from fastapi import FastAPI, HTTPException, UploadFile

from app.document_loader import (
    chunk_documents,
    load_document,
    load_questions,
    save_upload_file_temp,
)
from app.rag_chain import answer_questions, create_qa_chain, create_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Zania Q&A API")


@app.get("/")
async def root():
    return {"message": "Zania Q&A API is running"}


@app.post("/qa")
async def process_qa(
    questions_file: UploadFile,
    document_file: UploadFile,
) -> dict[str, str]:
    """
    Process Q&A request.

    Upload a questions file (JSON) and a document file (PDF or JSON).
    Returns JSON pairing each question with its answer.
    """
    logger.info("Starting Q&A processing")

    questions_suffix = os.path.splitext(questions_file.filename or "")[1]
    document_suffix = os.path.splitext(document_file.filename or "")[1]

    logger.info("Saving uploaded files to temporary storage")
    questions_path = await save_upload_file_temp(
        questions_file, suffix=questions_suffix
    )
    document_path = await save_upload_file_temp(document_file, suffix=document_suffix)
    logger.info(f"Files saved: questions={questions_path}, document={document_path}")

    try:
        logger.info("Loading questions from file")
        questions = load_questions(questions_path)
        if not questions:
            raise HTTPException(status_code=400, detail="No questions found in file")
        logger.info(f"Loaded {len(questions)} questions")

        logger.info("Loading document content")
        documents = load_document(document_path)
        if not documents:
            raise HTTPException(status_code=400, detail="No content found in document")
        logger.info(f"Loaded {len(documents)} document(s)")

        logger.info("Chunking documents")
        chunks = chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        logger.info("Creating vector store")
        vector_store = create_vector_store(chunks)
        logger.info("Vector store created successfully")

        logger.info("Creating QA chain")
        qa_chain = create_qa_chain(vector_store)
        logger.info("QA chain created successfully")

        logger.info("Answering questions")
        answers = await answer_questions(qa_chain, questions)
        logger.info(f"Generated {len(answers)} answers")

        logger.info("Q&A processing completed successfully")
        return answers  # Direct dict: {"question": "answer", ...}

    finally:
        logger.info("Cleaning up temporary files")
        if os.path.exists(questions_path):
            os.unlink(questions_path)
        if os.path.exists(document_path):
            os.unlink(document_path)
        logger.info("Cleanup completed")
