from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def create_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def build_vectorstore(chunks, embeddings):
    return FAISS.from_documents(chunks, embeddings)


def build_rag_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the provided context.

        Context:
        {context}

        Question:
        {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vectorstore.as_retriever()

    return create_retrieval_chain(retriever, document_chain)


def create_rag_pipeline(documents):
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    rag_chain = build_rag_chain(vectorstore)
    return rag_chain
