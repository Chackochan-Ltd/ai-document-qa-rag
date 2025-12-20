
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

from rag_pipeline import create_rag_pipeline


# ---------- UI SETUP ----------
st.set_page_config(page_title="AI Document Q&A", layout="centered")
st.title("📄 AI Document Q&A Assistant")
st.write("Upload a PDF and ask questions based on its content.")

st.write("✅ App loaded successfully")


# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing document..."):

        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Create RAG pipeline
        qa_chain = create_rag_pipeline(documents)

    st.success("Document processed successfully!")


    # ---------- QUESTION INPUT ----------
    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(question)

        st.subheader("Answer")
        st.write(answer)
