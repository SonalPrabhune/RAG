# Retrieval-Augmented Generation (RAG) Application

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions based on indexed knowledge. It combines a Flask backend API with a Streamlit frontend for an interactive user experience. The application leverages Chroma for document embedding and retrieval and OpenAI's GPT for generating responses.

---

## Features

- **Backend**: A Flask-based API implementing the RAG logic.
- **Frontend**: A Streamlit interface for interacting with the RAG system.
- **Document Retrieval**: Uses Chroma for embedding documents and similarity-based retrieval.
- **Configurable Responses**: Customize prompts, retrieval limits, and other settings in the Streamlit interface.

---

## Prerequisites

1. **Python**: Ensure Python 3.8 or later is installed.
2. **API Key**: Obtain an OpenAI API key.
3. **Dependencies**: Install required Python libraries (see below).

---

## Installation

1. **Clone the Repository**:
   ```bash
     git clone https://github.com/SonalPrabhune/RAG.git
     cd RAG
   ```

2. **Install Dependencies**:
   ```bash
     pip install -r requirements.txt
   ```

3. **Set Up Environment Variables: Export your OpenAI API key as an environment variable**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```
   
5. **Prepare the Data**:
   - Place your PDF files in the app/data directory for indexing.

---

## Usage

### Step 1: Start the Backend
Navigate to the app/backend directory:
```bash
  cd app/backend
```
Run the Flask server:
```bash
  python app.py
```
  The backend will start at http://127.0.0.1:5000.

### Step 2: Start the Frontend
Navigate to the app directory:
```bash
  cd app
```
Start the Streamlit app:
```bash
  streamlit run chat.py
```
Open http://localhost:8501 in your browser to access the chat interface.

---

## API Details

### POST /chat
Handles user queries and returns responses based on the retrieval-augmented generation system.

**Request Body**:
{
    "history": [{"user": "Question 1", "bot": "Response 1"}],
    "retrievalstrategy": "crs",
    "overrides": {
        "prompt_template": "Custom prompt (optional)",
        "exclude_category": "Category to exclude (optional)",
        "top": 3,
        "suggest_followup_questions": true
    }
}

**Response**:
{
    "data_points": ["Relevant sources"],
    "answer": "Generated answer",
    "thoughts": "Details of the query process"
}

---

## Project Structure
```bash
  RAG/
  ├── app/
  │   ├── backend/
  │   │   ├── app.py       # Flask backend server
  │   │   ├── chatretrievalstrategy.py  # RAG logic
  │   ├── chat.py          # Streamlit frontend
  │   ├── data/            # Directory for input PDFs
  │   ├── db/              # Directory for persisting the vector database
  ├── requirements.txt     # Python dependencies
```

---

## Customization
- **Prompt Configuration**: Modify the prompt directly in the Streamlit sidebar or backend logic in chatretrievalstrategy.py.
- **Data Directory**: Add or update PDFs in the app/data folder for indexing.

---

## Notes
- The application stores embeddings in a persistent directory (db) for efficient querying. Ensure the persist_directory path in app.py is correctly configured.
- For large document sets, consider increasing system memory or optimizing the chunk size in the CharacterTextSplitter.


---

## Troubleshooting
- **API Connection Error**: Ensure the Flask API is running on http://127.0.0.1:5000.
- **Streamlit Not Loading**: Verify Streamlit is running on http://localhost:8501.
- **Missing Data**: Ensure PDFs are present in the app/data directory.


---

## License
This project is licensed under the MIT License.

