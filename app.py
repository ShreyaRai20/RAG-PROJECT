import os
import shutil
import streamlit as st

from dotenv import load_dotenv

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings
)

from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()


# -----------------------------
# Config
# -----------------------------

CHROMA_PATH = "temp_chroma_db"
UPLOAD_PATH = "uploads"


os.makedirs(UPLOAD_PATH, exist_ok=True)



# -----------------------------
# Delete temporary database
# -----------------------------

def cleanup_session():

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    if os.path.exists(UPLOAD_PATH):

        for file in os.listdir(UPLOAD_PATH):
            os.remove(
                os.path.join(
                    UPLOAD_PATH,
                    file
                )
            )

    st.session_state.clear()



# -----------------------------
# Create Vector Database
# -----------------------------

def create_vector_database(files):

    documents = []


    for file in files:

        file_path = os.path.join(
            UPLOAD_PATH,
            file.name
        )

        with open(file_path,"wb") as f:
            f.write(file.getbuffer())


        loader = PyPDFLoader(file_path)

        docs = loader.load()

        documents.extend(docs)



    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


    chunks = splitter.split_documents(
        documents
    )


    embeddings = OpenAIEmbeddings()


    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )


    return vector_db



# -----------------------------
# Initialize chatbot
# -----------------------------

def get_response(query):

    retriever = st.session_state.vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )


    docs = retriever.invoke(query)


    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )


    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful AI assistant.

                Use ONLY the provided context.

                If answer is not present:
                say:
                I could not find the answer in the document.
                """
            ),

            (
                "human",
                """
                Context:

                {context}


                Question:

                {question}
                """
            )
        ]
    )


    final_prompt = prompt.invoke(
        {
            "context":context,
            "question":query
        }
    )


    model = ChatOpenAI(
        model="gpt-5.5"
    )


    response = model.invoke(
        final_prompt
    )


    return response.content



# -----------------------------
# Streamlit UI
# -----------------------------


st.title(
    "📚 Temporary Document RAG Assistant"
)


# Upload documents

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)



if uploaded_files and "vector_db" not in st.session_state:


    with st.spinner(
        "Processing documents..."
    ):

        db = create_vector_database(
            uploaded_files
        )

        st.session_state.vector_db = db
        st.success(
            "Documents ready!"
        )



# Chat section

if "messages" not in st.session_state:

    st.session_state.messages=[]



for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.write(
            msg["content"]
        )



if "vector_db" in st.session_state:


    query = st.chat_input(
        "Ask something about your documents"
    )


    if query:


        st.session_state.messages.append(
            {
                "role":"user",
                "content":query
            }
        )


        with st.chat_message("user"):
            st.write(query)



        answer = get_response(
            query
        )


        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )


        with st.chat_message("assistant"):
            st.write(answer)



# End session button

if "vector_db" in st.session_state:

    if st.button(
        "End Session and Delete Data"
    ):

        cleanup_session()

        st.success(
            "All documents and database deleted."
        )

        st.rerun()