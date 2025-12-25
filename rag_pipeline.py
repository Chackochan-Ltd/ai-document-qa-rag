import re
from transformers import pipeline

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import BaseOutputParser


# ------------------ Custom Output Parser ------------------
class CleanBulletParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Remove instruction/prompt leakage
        blacklist = [
            "Human:",
            "You are",
            "Your task",
            "IMPORTANT",
            "RULES",
            "Document:",
            "Question:",
            "Answer:",
            "Final Answer:"
        ]

        for item in blacklist:
            text = re.sub(rf"{item}.*", "", text, flags=re.IGNORECASE)

        # Split into sentences
        lines = re.split(r"[•\-\n]+", text)
        clean_lines = [line.strip() for line in lines if len(line.strip()) > 5]

        if not clean_lines:
            return "I have no clue. Please ask something that is within this PDF."

        # Force bullet points
        return "\n".join(f"• {line}" for line in clean_lines)


# ------------------ Document Chunking ------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


# ------------------ Local Embeddings ------------------
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ------------------ Vector Store ------------------
def build_vectorstore(chunks, embeddings):
    texts = [doc.page_content for doc in chunks if doc.page_content.strip()]

    if not texts:
        raise ValueError(
            "I could not read text from this PDF. "
            "It may be scanned or image-based."
        )

    return FAISS.from_texts(texts, embeddings)


# ------------------ RAG Chain ------------------
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    hf_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        do_sample=False
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using ONLY the information from the document.

Document:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | CleanBulletParser()
    )

    return rag_chain


# ------------------ Pipeline Entry ------------------
def create_rag_pipeline(documents):
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    return build_rag_chain(vectorstore)
