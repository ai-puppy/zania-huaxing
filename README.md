# Zania Q&A

A RAG (Retrieval-Augmented Generation) powered Q&A application that answers questions based on uploaded documents. Built with FastAPI, LangChain, and Streamlit.

## Features

- Upload PDF or JSON documents and get answers to your questions
- RAG-based question answering using OpenAI embeddings and GPT-4o-mini
- ChromaDB for in-memory vector storage
- Streamlit UI for easy interaction
- FastAPI backend with REST endpoints

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- OpenAI API key

## Environment Setup

1. Copy the example environment file:

```bash
cp example.env .env
```

2. Edit `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
```

## Installation

Install dependencies using uv:

```bash
uv sync
```

## Running the Application

### Option 1: FastAPI Backend

Start the FastAPI server:

```bash
uv run uvicorn app.main:app --port 8008 --reload
```

The API will be available at `http://localhost:8008`.

**API Endpoints:**

- `GET /` - Health check
- `POST /qa` - Upload questions (JSON) and document (PDF/JSON) files to get answers

### Option 2: Streamlit UI

First, make sure the FastAPI server is running (see Option 1), then start Streamlit:

```bash
uv run streamlit run streamlit_app.py
```

The Streamlit UI will be available at `http://localhost:8501`.

## Usage

### Using the Streamlit UI

1. Start both FastAPI and Streamlit servers
2. Open `http://localhost:8501` in your browser
3. Upload a questions file (JSON format with a list of questions)
4. Upload a document file (PDF or JSON)
5. Click "Get Answers" to process

### Using the API Directly

```bash
curl -X POST "http://localhost:8008/qa" \
  -F "questions_file=@example_input/questions.json" \
  -F "document_file=@example_input/soc2-type2.pdf"
```

### Example Input Files

The `example_input/` directory contains sample files:
- `questions.json` - Example questions file
- `soc2-type2.pdf` - Example PDF document

## Development

### Pre-commit Hooks

Install and set up pre-commit hooks to ensure code quality before commits:

```bash
uv run pre-commit install
```

This will run linting and formatting checks automatically on each commit.

### Code Quality

The project uses ruff for linting and formatting:

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .
```

### Running Tests

```bash
uv run pytest
```

## Project Structure

```
zania-huaxing/
├── app/
│   ├── main.py            # FastAPI application and endpoints
│   ├── document_loader.py # Document loading and chunking
│   └── rag_chain.py       # RAG chain with OpenAI and ChromaDB
├── example_input/         # Example input files
├── streamlit_app.py       # Streamlit UI
├── pyproject.toml         # Project dependencies
└── example.env            # Environment template
```
