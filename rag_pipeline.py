def split_documents(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def create_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings()


def build_vectorstore(chunks, embeddings):
    from langchain_community.vectorstores import FAISS
    return FAISS.from_documents(chunks, embeddings)


def build_qa_chain(vectorstore):
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA

    llm = ChatOpenAI(temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )


def create_rag_pipeline(documents):
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    qa_chain = build_qa_chain(vectorstore)

    return qa_chain
