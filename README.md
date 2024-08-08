# RAG-Streamlit---Chat-with-Document
RAG Streamlit - Chat with Document is an interactive chat application that leverages the power of Ollama and LlamaIndex to provide intelligent responses to technical questions based on the content of a provided PDF document. This application is built using Python 3.10 and Streamlit.

## Features

- **Interactive Chat Interface**: Engage in a conversation with the bot through an intuitive chat interface.
- **Document Upload**: Upload a PDF document, and the bot will process and index the content for querying.
- **Model Selection**: Choose from various large language models (LLMs) to tailor the bot's responses.
- **Efficient Indexing**: Utilizes advanced techniques to split text into sentences and generate embeddings for efficient searching.
- **Real-time Response Streaming**: Get instant responses streamed in real-time as you interact with the bot.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/rag-docbot.git
   cd rag-docbot
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

2. **Open your web browser and navigate to the local Streamlit server, typically `http://localhost:8501`.**

3. **Upload a PDF document**: Use the sidebar to upload a PDF document that you want to query.

4. **Select a Model**: Choose a language model from the dropdown menu in the sidebar.

5. **Start Chatting**: Once the document is processed, start interacting with the bot by typing your questions into the chat input field.

## Code Explanation

### Main Components

- **`load_data(document, model_name)`**:
  - Initializes an Ollama instance with the specified model.
  - Reads the provided PDF document.
  - Splits the text into sentences and generates embeddings.
  - Creates a `VectorStoreIndex` for efficient querying.

- **`main()`**:
  - Sets up the Streamlit page configuration.
  - Manages model selection and file upload in the sidebar.
  - Initializes chat history and chat engine.
  - Handles user input and displays chat messages.

## Dependencies

- Python 3.10
- Streamlit
- Ollama
- LlamaIndex
- HuggingFace
- Other required libraries are listed in `requirements.txt`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.ai/)
- [LlamaIndex](https://llama-index.readthedocs.io/)
- [HuggingFace](https://huggingface.co/)

Feel free to adjust the repository URL, license, and any other specifics to fit your actual project details.
