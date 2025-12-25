import re
from transformers import pipeline

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import BaseOutputParser


# ------------------ CLEAN OUTPUT PARSER ------------------
class CleanBulletParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Remove any leaked prompt or instruction text
        blacklist = [
            "Human:",
            "You are",
            "Your task",
            "IMPORTANT",
            "RULES",
            "Document:",
            "Question:",
            "Answer:",
            "Summarize",
            "ONLY",
            "Return",
        ]

        for item in blacklist:
            text = re.sub(rf"{item}.*", "", text, flags=re.IGNORECASE)

        # Normalize separators
        text = text.replace("•", ".")
        text = text.replace("·", ".")
        text = text.replace("–", ".")
        text = text.replace("-", ".")

        # Split into sentences
        sentences = re.split(r"\.\s+", text)

        # Clean, deduplicate, and order bullets
        bullets = []
        for s in sentences:
            s = s.strip()
            if len(s) > 6 and s.lower() not in [b.lower() for b in bullets]:
                bullets.append(s)

        if not bullets:
            return "I have no clue. Please ask something that is within this PDF."

        # Force one bullet per line
        return "\n".join(f"• {b}" for b in bullets)


# ------------------ SPLIT DOCUMENT ------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


# ------------------ LOCAL EMBEDDINGS ------------------
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ------------------ VECTOR STORE ------------------
def build_vectorstore(chunks, embeddings):
    texts = [doc.page_content for doc in chunks if doc.page_content.strip()]

    if not texts:
        raise ValueError(
            "I could not read text from this PDF. "
            "It may be scanned or image-based."
        )

    return FAISS.from_texts(texts, embeddings)


# ------------------ RAG CHAIN ------------------
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
Summarize the document content to answer the question.

Requirements:
- Answer only what the question asks
- Summarize, do not copy sentences
- Use simple language
- Return clear bullet points
- Each bullet must express one idea

If the answer is not found, return exactly:
I have no clue. Please ask something that is within this PDF.

Document:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | CleanBulletParser()
    )


# ------------------ ENTRY POINT ------------------
def create_rag_pipeline(documents):
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    return build_rag_chain(vectorstore)
