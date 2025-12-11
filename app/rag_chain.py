import os

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

QA_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question.
If you cannot find a direct answer in the context, provide the most relevant
information available. Be concise and specific.

Context:
{context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def create_vector_store(documents: list[Document]) -> Chroma:
    """Create an in-memory ChromaDB vector store from documents."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )
    return vector_store


def create_qa_chain(vector_store: Chroma) -> RetrievalQA:
    """Create a RetrievalQA chain with the vector store."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    return qa_chain


async def answer_questions(
    qa_chain: RetrievalQA,
    questions: list[str],
) -> dict[str, str]:
    """Answer a list of questions using the QA chain."""
    answers = {}
    for question in questions:
        result = await qa_chain.ainvoke({"query": question})
        answers[question] = result.get("result", "Unable to find answer")
    return answers
