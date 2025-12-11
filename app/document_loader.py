import json
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF file and return documents."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_json_document(file_path: str) -> list[Document]:
    """Load a JSON file as a document source."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        content = "\n\n".join(
            json.dumps(item, indent=2) if isinstance(item, dict) else str(item)
            for item in data
        )
    elif isinstance(data, dict):
        content = json.dumps(data, indent=2)
    else:
        content = str(data)

    return [Document(page_content=content, metadata={"source": file_path})]


def load_document(file_path: str) -> list[Document]:
    """Load a document from file path. Supports PDF and JSON."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix == ".json":
        return load_json_document(file_path)
    else:
        raise ValueError(f"Unsupported document type: {suffix}")


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents(documents)


def load_questions_json(file_path: str) -> list[str]:
    """Load questions from a JSON file.

    Expects a list of strings or objects with 'question' key.
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict) and "question" in item:
                questions.append(item["question"])
    return questions


def load_questions(file_path: str) -> list[str]:
    """Load questions from a JSON file.

    Per interview spec: Questions file must be JSON format.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix != ".json":
        raise ValueError(f"Questions file must be JSON, got: {suffix}")

    return load_questions_json(file_path)


async def save_upload_file_temp(upload_file, suffix: str = "") -> str:
    """Save an uploaded file to a temporary location and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload_file.read()
        tmp.write(content)
        return tmp.name
