import requests
import streamlit as st

st.set_page_config(page_title="Zania Q&A", page_icon="📚", layout="wide")

st.title("📚 Zania Q&A API Tester")
st.markdown(
    "Upload a **questions file (JSON)** and a **document file (PDF or JSON)** "
    "to get answers."
)

API_URL = "http://localhost:8008/qa"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Questions File (JSON)")
    questions_file = st.file_uploader(
        "Upload questions",
        type=["json"],
        help="JSON file containing list of questions",
    )

with col2:
    st.subheader("Document File (PDF or JSON)")
    document_file = st.file_uploader(
        "Upload document",
        type=["pdf", "json"],
        help="PDF or JSON file to search for answers",
    )

if st.button(
    "Get Answers", type="primary", disabled=not (questions_file and document_file)
):
    with st.spinner("Processing... This may take a minute."):
        try:
            files = {
                "questions_file": (
                    questions_file.name,
                    questions_file.getvalue(),
                    "application/json",
                ),
                "document_file": (
                    document_file.name,
                    document_file.getvalue(),
                    (
                        "application/pdf"
                        if document_file.name.endswith(".pdf")
                        else "application/json"
                    ),
                ),
            }
            response = requests.post(API_URL, files=files, timeout=300)

            if response.status_code == 200:
                answers = response.json()
                st.success(f"Found {len(answers)} answers!")

                for i, (question, answer) in enumerate(answers.items(), 1):
                    with st.expander(f"Q{i}: {question[:80]}...", expanded=True):
                        st.markdown(f"**Question:** {question}")
                        st.markdown(f"**Answer:** {answer}")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to API. "
                "Make sure the FastAPI server is running on port 8008."
            )
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("Make sure FastAPI is running: `uv run uvicorn app.main:app --port 8008`")
