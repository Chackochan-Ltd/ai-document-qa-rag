import re
from transformers import pipeline

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import BaseOutputParser


# ------------------ QUESTION INTENT DETECTOR ------------------
def is_explanatory_question(question: str) -> bool:
    keywords = [
        "explain",
        "describe",
        "elaborate",
        "why",
        "how",
        "in detail",
        "how does",
        "purpose",
    ]
    q = question.lower()
    return any(k in q for k in keywords)


# ------------------ CLEAN & ADAPTIVE OUTPUT PARSER ------------------
class CleanAnswerParser(BaseOutputParser):
    def parse(self, text: str, question: str) -> str:
        # Remove leaked instructions or prompt text
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

        text = text.strip()

        if not text:
            return "I have no clue. Please ask something that is within this PDF."

        # ------------------ PARAGRAPH MODE ------------------
        if is_explanatory_question(question):
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        # ------------------ BULLET MODE ------------------
        text = re.sub(r"[•\-–·]", ".", text)
        sentences = re.split(r"\.\s+", text)

        bullets = []
        for s in sentences:
            s = s.strip()
            if len(s) > 8 and s.lower() not in [b.lower() for b in bullets]:
                bullets.append(s)

        if not bullets:
            return "I have no clue. Please ask something that is within this PDF."

        return "\n".join(f"• {b}" for b in bullets)


# ------------------ DOCUMENT SPLITTING ------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
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
            "I could not read text from this PDF. It may be scanned or image-based."
        )

    return FAISS.from_texts(texts, embeddings)


# ------------------ RAG CHAIN ------------------
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Summarization model (NO prompt echo)
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        max_length=200,
        min_length=80,
        do_sample=False
    )

    llm = HuggingFacePipeline(pipeline=summarizer)

    prompt = ChatPromptTemplate.from_template(
        """
{context}

Question:
{question}
"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def final_output(inputs):
        raw_answer = inputs["answer"]
        question = inputs["question"]
        return CleanAnswerParser().parse(raw_answer, question)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | (lambda answer, q=RunnablePassthrough(): final_output(
            {"answer": answer, "question": q}
        ))
    )

    return rag_chain


# ------------------ ENTRY POINT ------------------
def create_rag_pipeline(documents):
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    return build_rag_chain(vectorstore)
