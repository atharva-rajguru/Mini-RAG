import streamlit as st
import tempfile
import os
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if "OPENAI_API_KEY" in os.environ:
    openai_api_key = os.environ["OPENAI_API_KEY"]
elif "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.error("Missing OpenAI API Key! Please configure it in your environment or secrets.")
    st.stop()

st.title("RAG Document Chatbot")

# 1. Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # We use session state to ensure we only process the document once
    # rather than re-embedding it every time the user sends a chat message.
    if "uploaded_filename" not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
        with st.spinner("Processing document..."):
            # Save the uploaded file temporarily so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Load document into 'doc' variable as requested
            loader = PyPDFLoader(tmp_file_path)
            doc = loader.load()
            
            # We need to break complete data equally in such a way that it will preserve the -
            # meaning and will be divided in chunks too.
            splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
            
            # Pass these document and break into chunks
            chunk = splits.split_documents(doc)
            
            # Use OpenAI's model to change these chunks into embeddings
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
            
            # Get these chunked data converted into embeddings and store it into chromaDB
            my_db = Chroma.from_documents(documents=chunk, embedding=embeddings)
            
            # Store db in session state so it persists across chat messages
            st.session_state.my_db = my_db
            st.session_state.uploaded_filename = uploaded_file.name
            
            # Reset chat history when a new file is uploaded
            st.session_state.messages = [] 
            
            # Clean up the temporary file
            os.remove(tmp_file_path)
            st.success("Document processed successfully!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if question := st.chat_input("Ask a question about your document"):
        # Display user message in chat message container
        st.chat_message("user").markdown(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            # Call any llm which will help process this retrieved data.
            llm = ChatOpenAI(model='gpt-5.4-mini', temperature=0.5) 
            
            # Perform similarity search on our embeddings to collect most relevent info.
            fetched = st.session_state.my_db.similarity_search(question, k=3)
            
            # Pass ths raw info to llm to process into human understandable value
            response = llm.invoke(f'{question} Answer this using these provided context only. If context does not has relevent info, DO NOT ANSWER the question: {fetched}.')
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response.content)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.content})
else:
    st.info("Please upload a PDF file to begin.")
