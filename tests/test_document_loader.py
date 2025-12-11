import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.document_loader import (
    chunk_documents,
    load_document,
    load_json_document,
    load_pdf,
    load_questions,
)


class TestLoadPdf:
    def test_load_pdf_returns_documents(self, pdf_file: Path):
        """Test that load_pdf returns a list of Document objects."""
        documents = load_pdf(str(pdf_file))

        assert isinstance(documents, list)
        assert len(documents) > 0
        assert all(isinstance(doc, Document) for doc in documents)

    def test_load_pdf_has_content(self, pdf_file: Path):
        """Test that loaded PDF documents have content."""
        documents = load_pdf(str(pdf_file))

        for doc in documents:
            assert doc.page_content is not None
            assert len(doc.page_content) > 0


class TestLoadJsonDocument:
    def test_load_json_dict(self, tmp_path: Path):
        """Test loading a JSON file with a dict structure."""
        json_file = tmp_path / "test.json"
        data = {"key": "value", "nested": {"inner": "data"}}
        json_file.write_text(json.dumps(data))

        documents = load_json_document(str(json_file))

        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert "key" in documents[0].page_content

    def test_load_json_list(self, tmp_path: Path):
        """Test loading a JSON file with a list structure."""
        json_file = tmp_path / "test.json"
        data = [{"item": 1}, {"item": 2}]
        json_file.write_text(json.dumps(data))

        documents = load_json_document(str(json_file))

        assert len(documents) == 1
        assert "item" in documents[0].page_content


class TestLoadDocument:
    def test_load_pdf_document(self, pdf_file: Path):
        """Test load_document with PDF file."""
        documents = load_document(str(pdf_file))

        assert isinstance(documents, list)
        assert len(documents) > 0

    def test_load_json_document(self, tmp_path: Path):
        """Test load_document with JSON file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"test": "data"}')

        documents = load_document(str(json_file))

        assert isinstance(documents, list)
        assert len(documents) == 1

    def test_unsupported_format_raises_error(self, tmp_path: Path):
        """Test that unsupported file formats raise ValueError."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some text")

        with pytest.raises(ValueError, match="Unsupported document type"):
            load_document(str(txt_file))


class TestChunkDocuments:
    def test_chunk_documents_splits_long_document(self):
        """Test that long documents are split into chunks."""
        long_content = "word " * 1000  # ~5000 chars
        documents = [Document(page_content=long_content)]

        chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=50)

        assert len(chunks) > 1
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_chunk_documents_preserves_short_document(self):
        """Test that short documents remain as single chunks."""
        short_content = "This is a short document."
        documents = [Document(page_content=short_content)]

        chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=100)

        assert len(chunks) == 1
        assert chunks[0].page_content == short_content


class TestLoadQuestions:
    def test_load_questions_from_example_file(self, questions_file: Path):
        """Test loading questions from the example questions.json file."""
        questions = load_questions(str(questions_file))

        assert isinstance(questions, list)
        assert len(questions) > 0
        assert all(isinstance(q, str) for q in questions)

    def test_load_questions_list_of_strings(self, tmp_path: Path):
        """Test loading questions from a list of strings."""
        json_file = tmp_path / "questions.json"
        data = ["Question 1?", "Question 2?", "Question 3?"]
        json_file.write_text(json.dumps(data))

        questions = load_questions(str(json_file))

        assert questions == data

    def test_load_questions_list_of_dicts(self, tmp_path: Path):
        """Test loading questions from a list of dicts with 'question' key."""
        json_file = tmp_path / "questions.json"
        data = [
            {"question": "Question 1?"},
            {"question": "Question 2?"},
        ]
        json_file.write_text(json.dumps(data))

        questions = load_questions(str(json_file))

        assert questions == ["Question 1?", "Question 2?"]

    def test_load_questions_non_json_raises_error(self, tmp_path: Path):
        """Test that non-JSON questions file raises ValueError."""
        txt_file = tmp_path / "questions.txt"
        txt_file.write_text("Question 1?")

        with pytest.raises(ValueError, match="Questions file must be JSON"):
            load_questions(str(txt_file))
