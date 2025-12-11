import os
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.document_loader import chunk_documents, load_document, load_questions
from app.rag_chain import answer_questions, create_qa_chain, create_vector_store
from tests.conftest import requires_openai_api_key


def _get_judge_llm() -> ChatOpenAI:
    """Get an LLM instance for judging."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def llm_judge_relevance(
    query: str, retrieved_content: str, expected_topic: str
) -> bool:
    """Use LLM to judge if retrieved content is relevant to the query.

    Args:
        query: The search query
        retrieved_content: The content retrieved from vector store
        expected_topic: What the content should be about

    Returns:
        True if the LLM judges the content as relevant
    """
    llm = _get_judge_llm()

    prompt = f"""You are evaluating a retrieval system. Given a query and retrieved \
content, determine if the retrieved content is relevant to the query.

Query: {query}
Expected topic: {expected_topic}
Retrieved content: {retrieved_content}

Is the retrieved content relevant to the query and about the expected topic?
Answer only "YES" or "NO"."""

    response = llm.invoke(prompt)
    return response.content.strip().upper() == "YES"


def llm_judge_answer_quality(question: str, answer: str, context: str = "") -> dict:
    """Use LLM to judge if an answer properly addresses a question.

    Args:
        question: The question that was asked
        answer: The answer provided
        context: Optional context about what the answer should contain

    Returns:
        dict with 'is_valid' (bool) and 'reason' (str)
    """
    llm = _get_judge_llm()

    context_section = f"\nExpected context: {context}" if context else ""

    prompt = f"""You are evaluating a Q&A system's answer quality.

Question: {question}
Answer: {answer}{context_section}

Evaluate if the answer:
1. Directly addresses the question
2. Is coherent and well-formed
3. Does not simply say "I don't know" or refuse to answer without reason

Respond in this exact format:
VALID: YES or NO
REASON: <brief explanation>"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    is_valid = "VALID: YES" in content.upper()
    reason = content.split("REASON:")[-1].strip() if "REASON:" in content else content

    return {"is_valid": is_valid, "reason": reason}


@requires_openai_api_key
class TestCreateVectorStore:
    def test_create_vector_store_from_documents(self):
        """Test creating a vector store from documents."""
        documents = [
            Document(page_content="The data center is located in Virginia."),
            Document(page_content="We use AES-256 encryption for all data."),
        ]

        vector_store = create_vector_store(documents)

        assert vector_store is not None

        # Test that we can search the vector store and get relevant results
        query = "data center location"
        results = vector_store.similarity_search(query, k=1)
        assert len(results) > 0

        # Use LLM as judge to verify relevance
        retrieved_content = results[0].page_content
        is_relevant = llm_judge_relevance(
            query=query,
            retrieved_content=retrieved_content,
            expected_topic="data center location",
        )
        assert is_relevant, f"Retrieved content not relevant: {retrieved_content}"

    def test_vector_store_retrieves_correct_document(self):
        """Test that vector store retrieves the most relevant document."""
        documents = [
            Document(page_content="The data center is located in Virginia."),
            Document(page_content="We use AES-256 encryption for all data."),
            Document(page_content="Our company was founded in 2015."),
        ]

        vector_store = create_vector_store(documents)

        # Query about encryption should retrieve encryption document
        query = "What encryption do you use?"
        results = vector_store.similarity_search(query, k=1)

        is_relevant = llm_judge_relevance(
            query=query,
            retrieved_content=results[0].page_content,
            expected_topic="encryption",
        )
        assert (
            is_relevant
        ), f"Expected encryption content, got: {results[0].page_content}"


@requires_openai_api_key
class TestCreateQaChain:
    def test_create_qa_chain(self):
        """Test creating a QA chain from a vector store."""
        documents = [
            Document(page_content="Our headquarters is in San Francisco."),
        ]
        vector_store = create_vector_store(documents)

        qa_chain = create_qa_chain(vector_store)

        assert qa_chain is not None


@requires_openai_api_key
class TestAnswerQuestions:
    @pytest.mark.asyncio
    async def test_answer_single_question(self):
        """Test answering a single question."""
        documents = [
            Document(page_content="The company was founded in 2020."),
            Document(page_content="The CEO is John Smith."),
        ]
        vector_store = create_vector_store(documents)
        qa_chain = create_qa_chain(vector_store)

        answers = await answer_questions(qa_chain, ["When was the company founded?"])

        assert len(answers) == 1
        assert "When was the company founded?" in answers
        assert "2020" in answers["When was the company founded?"]

    @pytest.mark.asyncio
    async def test_answer_multiple_questions(self):
        """Test answering multiple questions."""
        documents = [
            Document(page_content="The company was founded in 2020."),
            Document(page_content="The CEO is John Smith."),
            Document(page_content="The headquarters is in New York."),
        ]
        vector_store = create_vector_store(documents)
        qa_chain = create_qa_chain(vector_store)

        questions = [
            "When was the company founded?",
            "Who is the CEO?",
        ]
        answers = await answer_questions(qa_chain, questions)

        assert len(answers) == 2
        assert all(q in answers for q in questions)


@requires_openai_api_key
class TestEndToEndWithExampleInput:
    """End-to-end tests using the example_input files."""

    @pytest.mark.asyncio
    async def test_qa_with_example_pdf_and_questions(
        self, pdf_file: Path, questions_file: Path
    ):
        """Test the full Q&A pipeline with example PDF and questions."""
        # Load and process documents
        documents = load_document(str(pdf_file))
        chunks = chunk_documents(documents)

        # Create vector store and QA chain
        vector_store = create_vector_store(chunks)
        qa_chain = create_qa_chain(vector_store)

        # Load questions (just use first 3 for faster test)
        questions = load_questions(str(questions_file))[:3]

        # Get answers
        answers = await answer_questions(qa_chain, questions)

        # Verify we got answers for all questions
        assert len(answers) == len(questions)
        for question in questions:
            assert question in answers
            answer = answers[question]
            assert isinstance(answer, str)
            assert len(answer) > 0

            # Use LLM as judge to verify answer quality
            judgment = llm_judge_answer_quality(
                question=question,
                answer=answer,
                context="SOC2 compliance document",
            )
            assert judgment[
                "is_valid"
            ], f"Answer quality check failed for '{question}': {judgment['reason']}"

    @pytest.mark.asyncio
    async def test_qa_returns_relevant_answers(self, pdf_file: Path):
        """Test that answers are relevant to the context in the PDF."""
        documents = load_document(str(pdf_file))
        chunks = chunk_documents(documents)
        vector_store = create_vector_store(chunks)
        qa_chain = create_qa_chain(vector_store)

        # Ask a question about data centers (common SOC2 topic)
        question = "Where are your data centres located?"
        answers = await answer_questions(qa_chain, [question])

        answer = answers[question]
        assert isinstance(answer, str)
        assert len(answer) > 0

        # Use LLM as judge to verify the answer addresses the question
        judgment = llm_judge_answer_quality(
            question=question,
            answer=answer,
            context="Should mention location(s) or indicate if not in document",
        )
        assert judgment[
            "is_valid"
        ], f"Answer quality check failed: {judgment['reason']}"
