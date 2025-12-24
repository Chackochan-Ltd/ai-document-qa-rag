from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ------------------ Document Chunking ------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


# ------------------ Embeddings (LOCAL, STABLE) ------------------
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ------------------ Vector Store ------------------
def build_vectorstore(chunks, embeddings):
    texts = [doc.page_content for doc in chunks if doc.page_content.strip()]

    if not texts:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The document may be scanned or image-based."
        )

    return FAISS.from_texts(texts, embeddings)



# ------------------ RAG Chain (LCEL – Modern & Stable) ------------------
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Use the context below to answer the question.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain



# ------------------ Pipeline Entry ------------------
def create_rag_pipeline(documents):
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    return build_rag_chain(vectorstore)
