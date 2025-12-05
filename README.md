# RAG Client

Lightweight RAG (Retrieval-Augmented Generation) system that logs performance metrics to a remote server.

## What This Does

- Embeds queries using sentence-transformers
- Searches documents using Qdrant vector database
- Generates answers using OpenAI
- Logs all performance metrics to a remote logging server

## Setup

1. **Clone the repo:**
```bash
git clone https://github.com/al-gent/rag-client.git
cd rag-client
```

2. **Create `.env` file:**
```bash
cp .env.example .env
# Edit .env with your values
```

Required variables:
- `RAG_HARDWARE_ID` - Identifier for your hardware (e.g., `laptop-mac`, `server-gpu`)
- `LLM_HARDWARE_ID` - Where LLM runs (e.g., `openai-api`, `local-gpu`)
- `MODEL_NAME` - Which model to use (e.g., `gpt-4o-mini`)
- `LOG_SERVER_URL` - Remote logging server (e.g., `https://rag-api.adamlgent.com`)
- `OPENAI_API_KEY` - Your OpenAI API key

3. **Start the system:**
```bash
docker-compose up -d
```

4. **Load documents:**
```bash
# Put PDF or TXT files in ./data directory
curl -X POST http://localhost:8000/load-documents
```

5. **Query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

## What Gets Logged

Every query logs:
- Embedding time
- Vector search time
- LLM generation time
- Total time
- Cost
- Success/failure

All metrics are sent to the remote logging server for comparison across different hardware setups.

## Endpoints

- `POST /query` - Ask a question
- `POST /upload` - Upload a document
- `POST /load-documents` - Load all documents from ./data
- `GET /health` - Health check
- `GET /documents` - List indexed documents
