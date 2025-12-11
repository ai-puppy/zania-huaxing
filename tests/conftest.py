import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Path to example input files
EXAMPLE_INPUT_DIR = Path(__file__).parent.parent / "example_input"
QUESTIONS_FILE = EXAMPLE_INPUT_DIR / "questions.json"
PDF_FILE = EXAMPLE_INPUT_DIR / "soc2-type2.pdf"
CSV_FILE = EXAMPLE_INPUT_DIR / "Sample JSON.xlsx - Sheet1.csv"


def has_openai_api_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


# Skip decorator for tests requiring OpenAI API key
requires_openai_api_key = pytest.mark.skipif(
    not has_openai_api_key(),
    reason="OPENAI_API_KEY environment variable not set",
)


@pytest.fixture
def example_input_dir() -> Path:
    """Path to example_input directory."""
    return EXAMPLE_INPUT_DIR


@pytest.fixture
def questions_file() -> Path:
    """Path to example questions.json file."""
    return QUESTIONS_FILE


@pytest.fixture
def pdf_file() -> Path:
    """Path to example PDF file."""
    return PDF_FILE


@pytest.fixture
def csv_file() -> Path:
    """Path to example CSV file."""
    return CSV_FILE
