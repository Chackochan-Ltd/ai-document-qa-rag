from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


# ----------- Split PDF into chunks -----------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


# ----------- Local embeddings (stable) -----------
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ----------- Vector Store -----------
def build_vectorstore(chunks, embeddings):
    texts = [doc.page_content for doc in chunks if doc.page_content.strip()]
    if not texts:
        raise ValueError(
            "I could not read text from this PDF. "
            "It may be scanned or image-based."
        )
    return FAISS.from_texts(texts, embeddings)


# ----------- RAG Chain -----------
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    hf_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant.

Your task:
Understand the question and answer it using ONLY the information found in the document.

IMPORTANT:
- Do NOT include instructions
- Do NOT include explanations about how you answered
- Do NOT repeat the question
- Do NOT mention the document or rules
- Return ONLY the final answer content
- Format the answer strictly as bullet points.

If the answer is not present in the document, return exactly this sentence:
I have no clue. Please ask something that is within this PDF.

Document:
{context}

Question:
{question}

Final Answer:
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
        | StrOutputParser()
    )


# ----------- Entry Point -----------
def create_rag_pipeline(documents):
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    return build_rag_chain(vectorstore)
