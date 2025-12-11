import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

QA_PROMPT = ChatPromptTemplate.from_template(
    """Use the following pieces of context to answer the question.
If you cannot find a direct answer in the context, provide the most relevant
information available. Be concise and specific.

Context:
{context}

Question: {question}

Answer:"""
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


def _format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(vector_store: Chroma):
    """Create a RAG chain using LCEL (LangChain Expression Language)."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    # LCEL chain: retrieve → format → prompt → llm → parse
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain


async def answer_questions(
    rag_chain,
    questions: list[str],
) -> dict[str, str]:
    """Answer a list of questions using the RAG chain."""
    answers = {}
    for question in questions:
        answer = await rag_chain.ainvoke(question)
        answers[question] = answer
    return answers
