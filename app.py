import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from rag_pipeline import create_rag_pipeline


# ---------- PAGE SETUP ----------
st.set_page_config(page_title="AI Document Q&A", layout="centered")
st.title("📄 AI Document Q&A Assistant")
st.write("Upload a PDF and ask questions based on its content.")


# ---------- SESSION STATE ----------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "history" not in st.session_state:
    st.session_state.history = []


# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        try:
            st.session_state.qa_chain = create_rag_pipeline(documents)
            st.success("Document processed successfully!")
        except ValueError as e:
            st.error(str(e))
            st.stop()


# ---------- QUESTION INPUT ----------
if st.session_state.qa_chain:
    question = st.text_input("Ask a question about the document")

    if question:
        with st.spinner("Generating answer..."):
            answer = st.session_state.qa_chain.invoke(question)

        st.session_state.history.append((question, answer))


# ---------- DISPLAY RESULTS ----------
for q, a in reversed(st.session_state.history):
    with st.container():
        st.markdown(
            f"""
            <div style="
                border:1px solid #444;
                border-radius:10px;
                padding:15px;
                margin-bottom:15px;
                background-color:#111;
            ">
                <b>Question:</b><br>
                {q}<br><br>
                <b>Answer:</b><br>
                {a.replace('-', '•')}
            </div>
            """,
            unsafe_allow_html=True
        )
