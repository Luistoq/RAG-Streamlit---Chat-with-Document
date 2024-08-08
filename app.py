import glob
from tempfile import NamedTemporaryFile

# Importing necessary libraries
import ollama
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, SimpleDirectoryReader, ServiceContext, VectorStoreIndex
import streamlit as st

def response_generator(stream):
    """Generator that yields chunks of data from a stream response.

    Args:
        stream: The stream object from which to read data chunks.

    Yields:
        bytes: The next chunk of data from the stream response.
    """
    for chunk in stream.response_gen:
        yield chunk

@st.cache_resource(show_spinner=False)
def load_data(document, model_name: str) -> VectorStoreIndex:
    """Loads and indexes Streamlit documentation using Ollama and LlamaIndex.

    This function performs the following actions:
    1. Initializes an Ollama instance using the provided model name.
    2. Reads the provided PDF document using SimpleDirectoryReader.
    3. Splits the loaded document into sentences and generates embeddings.
    4. Creates a ServiceContext object with all necessary components.
    5. Creates a VectorStoreIndex instance from the processed documents.

    Args:
        document (UploadedFile): The uploaded PDF document to be processed.
        model_name (str): The name of the LLM model to be used by Ollama.

    Returns:
        VectorStoreIndex: An instance containing the indexed documents and embeddings.
    """
    # Initialize the large language model (LLM)
    llm = Ollama(model=model_name, request_timeout=30.0)

    # Temporary file to hold the uploaded PDF document
    with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
        f.write(document.getbuffer())
        with st.spinner(text="Loading and indexing the document. This should take 1-2 minutes."):
            # Load the document using SimpleDirectoryReader
            docs = SimpleDirectoryReader(".").load_data()

            # Initialize the text splitter and embedding model
            text_splitter = SentenceSplitter(chunk_size=512)
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
            
            # Create a service context with all necessary components
            service_context = ServiceContext.from_defaults(
                llm=llm,
                embed_model=embed_model,
                text_splitter=text_splitter,
                system_prompt="You are a Python expert and your job is to answer technical questions. Keep your answers technical and based on facts."
            )

            # Create a VectorStoreIndex instance
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    
    return index

def main() -> None:
    """Controls the main chat application logic using Streamlit and Ollama.

    This function orchestrates the chat application with the following tasks:
    1. Sets up the Streamlit page configuration.
    2. Manages model selection and stores the chosen model in session state.
    3. Initializes the chat history list in session state.
    4. Calls load_data to create a VectorStoreIndex from the provided model name.
    5. Initializes the chat engine using the VectorStoreIndex instance.
    6. Displays chat history messages.
    7. Handles user input and displays messages in the chat interface.
    8. Generates responses to user prompts and displays them.

    Args:
        None
    """
    st.set_page_config(page_title="Chatbot", page_icon="🦙", layout="centered", initial_sidebar_state="auto")
    st.title("RAG-Streamlit---Chat-with-your-Document")
    
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    with st.sidebar:
        # Model selection
        if "model" not in st.session_state:
            st.session_state["model"] = ""
        models = [model["name"] for model in ollama.list()["models"]]
        st.session_state["model"] = st.selectbox("Select a model", models)
        
        # Initialize the LLM
        llm = Ollama(model=st.session_state["model"], request_timeout=30.0)

        # File uploader for the PDF document
        document = st.file_uploader("Upload a PDF file to query", type=['pdf'], accept_multiple_files=False)

        # Process the uploaded file
        if st.button('Process file'):
            index = load_data(document, st.session_state["model"])
            st.session_state.activate_chat = True

    if st.session_state.activate_chat:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize the chat engine
        if "chat_engine" not in st.session_state:
            st.session_state.chat_engine = index.as_chat_engine(chat_mode="context", streaming=True)

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("How can I help you?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            if st.session_state.messages[-1]["role"] != "assistant":
                message_placeholder = st.empty()
                with st.chat_message("assistant"):
                    stream = st.session_state.chat_engine.stream_chat(prompt)
                    response = st.write_stream(response_generator(stream))
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.markdown("<span style='font-size:15px;'><b>Upload a PDF to start chatting</b></span>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
