import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import tempfile
import os
import requests
import shutil


def get_installed_models():
    """Gets the list of installed models in Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json()
            # Extract only model names
            return [model['name'] for model in models['models']]
        return []
    except:
        st.error("Could not connect to Ollama. Please check if it's running.")
        return []

# Streamlit page configuration
st.set_page_config(page_title="Docater", layout="wide")
st.title("Docater")

# Get installed models
installed_models = get_installed_models()

if not installed_models:
    st.error("No models found or Ollama is not running.")
    st.stop()

# Model selectors in sidebar
with st.sidebar:
    st.header("Model Settings")
    model_llm = st.selectbox(
        "Choose LLM model:",
        options=installed_models,
        index=1
    )
    
    model_embedding = st.selectbox(
        "Choose Embedding model:",
        options=installed_models,
        index=0
    )

    # Add separator
    st.markdown("---")
    
    # Language selector
    st.header("Language Settings")
    output_language = st.selectbox(
        "Response Language:",
        options=["English", "Portuguese", "Spanish", "French", "German"],
        index=0
    )

    # Add separator
    st.markdown("---")
    
    # Database management section
    st.header("Database Management")
    if st.button("Clear All Processed Files", type="secondary"):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            os.makedirs("./chroma_db")
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.sidebar.success("All processed files have been cleared!")
            st.rerun()
        else:
            st.sidebar.info("No processed files to clear.")

# Session state initialization
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# PDF file upload
uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

if uploaded_file is not None:
    # Get file name without extension for the directory name
    file_name = os.path.splitext(uploaded_file.name)[0]
    chroma_dir = os.path.join("./chroma_db", file_name)

    # Create temporary file for PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Check if this PDF was already processed
    if os.path.exists(chroma_dir):
        st.info("This document was previously processed. Loading existing data...")
        
        # Initialize embeddings with selected model and CUDA config
        embeddings = OllamaEmbeddings(
            model=model_embedding,
        )

        # Load existing vectorstore
        vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings
        )

        # Initialize selected LLM model with CUDA config
        llm = Ollama(
            model=model_llm,
        )

        # Create QA chain
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        st.success("Document loaded successfully!")

    # Process new PDF
    elif st.button("Process PDF"):
        with st.spinner("Processing document..."):
            # Load PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            # Initialize embeddings with selected model and CUDA config
            embeddings = OllamaEmbeddings(
                model=model_embedding
            )

            # Create and persist vectorstore
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=chroma_dir
            )
            
            # Initialize selected LLM model with CUDA config
            llm = Ollama(
                model=model_llm
            )

            # Create QA chain
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            
            st.success("Document processed successfully!")

    # Chat interface
    if st.session_state.qa_chain is not None:
        st.subheader("Chat about the document")
        
        # Container for chat history
        chat_container = st.container()
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f'**👤 You:** {question}')
                # Assistant message
                st.markdown(f'**🤖 Assistant:** {answer}')
                # Separator between messages
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")
        
        # Input for new question
        with st.form(key='chat_form'):
            question = st.text_input("Your question:", key="question_input")
            submit_button = st.form_submit_button("Send")
            
            if submit_button and question:
                with st.spinner("Generating response..."):
                    # Add language instruction to the question
                    language_prompt = f"Please respond in {output_language}. "
                    full_question = language_prompt + question
                    
                    response = st.session_state.qa_chain.run(full_question)
                    # Add question and answer to history
                    st.session_state.chat_history.append((question, response))
                    # Clear input
                    st.rerun()

    # Button to clear history
    if st.session_state.chat_history:
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()

    # Clean up temporary file
    os.unlink(tmp_path)

# Create chroma_db directory if it doesn't exist
if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db")