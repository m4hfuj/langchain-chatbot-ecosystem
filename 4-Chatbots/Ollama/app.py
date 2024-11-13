# STEP 1: IMPORTS AND SET ENVIRONMENT VARIABLES
import chromadb.api


import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
## Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


## set up streamlit
st.title("Conversation RAG with PDF upload and chat history")
st.write("Upload Pdf's and chat with their content")

groq_api_key = os.getenv("GROQ_API_KEY")

## Load llm model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
session_id = st.text_input("Session ID", value="default_session")

## statefully manage chat history
if 'store' not in st.session_state:
    st.session_state.store={}

uploaded_files = st.file_uploader("Choose A Pdf file", type="pdf", accept_multiple_files=True)


"""
Got the line bellow cause of the error: ChromaDB - ValueError: Could not connect to tenant default_tenant. Are you sure it exists?
Got the solution from: https://github.com/langchain-ai/langchain/issues/26884
"""
chromadb.api.client.SharedSystemClient.clear_system_cache()


### Process uploaded pdf
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temp = f"./temp.pdf"
        with open(temp, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader(temp)
        docs = loader.load()
        documents.extend(docs)

    #### Split and create embeddings and store them in vectorstore
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # chromadb.api.client.SharedSystemClient.clear_system_cache()

    ## Prompts creation and history_aware_retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history "
        "formulate a stand alone question which can be understood "
        "without the chat history. Do not answer the question, "
        "just formulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


    ## Answer Question
    system_prompt = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


    user_input = st.text_input("Your question:")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversation_rag_chain.invoke(
            {"input": user_input},
            config= {
                "configurable": {"session_id": session_id}
            },
        )
        # st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)

