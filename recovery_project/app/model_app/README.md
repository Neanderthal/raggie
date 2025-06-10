# pgvector-rag
A Python application for building RAG (Retrieval-Augmented Generation) systems using pgvector and PostgreSQL. Provides command-line tools for database setup, document ingestion, and chat functionality with local LLMs via LM Studio.

Key Features:
- Create and manage vector databases with pgvector
- Import documents (PDF, MD, DOCX) and generate embeddings
- Chat interface with local LLMs through LM Studio
- Streamlit-based UI for interactive chat

## Requirements
- Python3
- PostgreSQL
- pgvector

## Install

Clone the repository

```
git clone git@github.com:gulcin/pgvector-rag.git
cd pgvector-rag
```

Install Dependencies

```
virtualenv env -p `which python`
source env/bin/activate
pip install -r requirements.txt
```

Add your .env variable

```
cp .env-example .env
```

## Commands

### Database Setup
```
python app.py create-db
```
Creates a PostgreSQL database with pgvector extension.

### Import Documents
```
python app.py import-data <document_path>
```
Processes documents (PDF, MD, DOCX) and stores their embeddings in the database.

### Chat Interface
```
python app.py chat
```
Starts a chat interface using a local LLM via LM Studio.

### Help
```
python app.py --help
```
Shows all available commands and options.

## Streamlit UI

The application includes a Streamlit-based UI for interactive chat:

1. Install Streamlit:
```
pip install streamlit
```

2. Configure LM Studio:
Create `.streamlit/secrets.toml` with your LM Studio configuration:
```
# .streamlit/secrets.toml
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LM_STUDIO_MODEL_NAME = "google/gemma-3-12b"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
```

3. Run the UI:
```
streamlit run streamlit/chatgptui.py
```
