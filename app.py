import re
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


# ---------- QUESTION INTENT ----------
def is_explanatory_question(question: str) -> bool:
    keywords = [
        "explain", "describe", "why", "how",
        "elaborate", "in detail", "purpose"
    ]
    q = question.lower()
    return any(k in q for k in keywords)


# ---------- ANSWER FORMATTER ----------
def format_answer(answer: str, question: str) -> str:
    if not answer or not answer.strip():
        return "I have no clue. Please ask something that is within this PDF."

    # Paragraph mode
    if is_explanatory_question(question):
        return re.sub(r"\s+", " ", answer).strip()

    # Bullet mode
    answer = re.sub(r"[•\-–·]", ".", answer)
    sentences = re.split(r"\.\s+", answer)

    bullets = []
    for s in sentences:
        s = s.strip()
        if len(s) > 8 and s.lower() not in [b.lower() for b in bullets]:
            bullets.append(s)

    if not bullets:
        return "I have no clue. Please ask something that is within this PDF."

    return "\n".join(f"• {b}" for b in bullets)


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
            raw_answer = st.session_state.qa_chain.invoke(question)
            final_answer = format_answer(raw_answer, question)

        st.session_state.history.append((question, final_answer))


# ---------- DISPLAY HISTORY ----------
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
                <b>Question:</b><br>{q}<br><br>
                <b>Answer:</b><br>{a.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )
