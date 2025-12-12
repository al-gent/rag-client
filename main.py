from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from pathlib import Path
import pypdf
import uuid
from datetime import datetime
import time
from pydantic import BaseModel
from typing import Optional 
import requests 
import traceback

# Logging API request models
class QueryLogRequest(BaseModel):
    query_id: str
    rag_hardware_id: str
    llm_hardware_id: str
    model_name: str
    question: str
    answer: str
    num_chunks_retrieved: int
    avg_similarity_score: float
    embedding_time_ms: float
    vector_search_time_ms: float
    llm_time_ms: float
    total_time_ms: float
    estimated_cost_usd: float
    success: bool
    error_message: Optional[str] = None

class UploadLogRequest(BaseModel):
    upload_id: str
    rag_hardware_id: str
    filename: str
    num_chunks: int
    file_size_bytes: int
    text_extraction_time_ms: float
    chunking_time_ms: float
    embedding_time_ms: float
    vector_store_time_ms: float
    total_time_ms: float
    success: bool
    error_message: Optional[str] = None

RAG_HARDWARE_ID = os.getenv("RAG_HARDWARE_ID", "local-dev")
LLM_HARDWARE_ID = os.getenv("LLM_HARDWARE_ID", "local-dev")
MODEL_NAME = os.getenv("LLM_MODEL")
LOG_SERVER_URL = os.getenv("LOG_SERVER_URL", None)

app = FastAPI()

# CORS - allow your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
qdrant = QdrantClient(host="qdrant", port=6333)
embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

def get_llm_client():
    provider = os.getenv("LLM_PROVIDER", "openai")
    
    if provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    else:
        # For local/self-hosted providers
        return OpenAI(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key="not-needed"  # Most local servers don't validate this
        )
# Usage
client = get_llm_client()

COLLECTION_NAME = "documents"

def init_collection():
    collections = qdrant.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME not in collection_names:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

def send_log_to_server(endpoint: str, log_data: dict):
    """Send log data to remote logging server"""
    if not LOG_SERVER_URL:
        return False  # No server configured
    
    try:
        response = requests.post(
            f"{LOG_SERVER_URL}{endpoint}",
            json=log_data,
            timeout=5
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to send log to server: {e}")
        return False

init_collection()

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

# Helper functions
def chunk_text(text: str, chunk_size: int = 5000, overlap: int = 200):
    """Simple chunking by character count with overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def extract_text_from_file(file_path: str):
    """Extract text from PDF or text file"""
    if file_path.endswith('.pdf'):
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

# Endpoints
@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT file and add it to the vector store"""
    upload_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
        
        # Save temporarily and get file size
        file_path = f"/tmp/{file.filename}"
        content = await file.read()
        file_size_bytes = len(content)
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Extract text
        extract_start = time.time()
        text = extract_text_from_file(file_path)
        text_extraction_time_ms = (time.time() - extract_start) * 1000
        
        # Chunk it
        chunk_start = time.time()
        chunks = chunk_text(text)
        chunking_time_ms = (time.time() - chunk_start) * 1000
        
        # Embed and store
        embed_start = time.time()
        points = []
        for i, chunk in enumerate(chunks):
            embedding = embed_model.encode(chunk).tolist()
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "source": file.filename,
                    "chunk_id": i
                }
            )
            points.append(point)
        embedding_time_ms = (time.time() - embed_start) * 1000
        
        # Store in vector DB
        store_start = time.time()
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        vector_store_time_ms = (time.time() - store_start) * 1000
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Log to server
        log_data = {
            "upload_id": upload_id,
            "rag_hardware_id": RAG_HARDWARE_ID,
            "filename": file.filename,
            "num_chunks": len(chunks),
            "file_size_bytes": file_size_bytes,
            "text_extraction_time_ms": text_extraction_time_ms,
            "chunking_time_ms": chunking_time_ms,
            "embedding_time_ms": embedding_time_ms,
            "vector_store_time_ms": vector_store_time_ms,
            "total_time_ms": total_time_ms,
            "success": True,
            "error_message": None
        }
        
        if not LOG_SERVER_URL:
            print("WARNING: LOG_SERVER_URL not configured - upload not logged!")
        elif not send_log_to_server("/api/log-upload", log_data):
            print(f"ERROR: Failed to send upload log to {LOG_SERVER_URL}")
        
        return {
            "message": f"Uploaded {file.filename} with {len(chunks)} chunks",
            "upload_id": upload_id,
            "timing": {
                "text_extraction_ms": round(text_extraction_time_ms, 2),
                "chunking_ms": round(chunking_time_ms, 2),
                "embedding_ms": round(embedding_time_ms, 2),
                "vector_store_ms": round(vector_store_time_ms, 2),
                "total_ms": round(total_time_ms, 2)
            }
        }
        
    except Exception as e:
        total_time_ms = (time.time() - start_time) * 1000
        log_data = {
            "upload_id": upload_id,
            "rag_hardware_id": RAG_HARDWARE_ID,
            "filename": file.filename if file.filename else "unknown",
            "num_chunks": 0,
            "file_size_bytes": 0,
            "text_extraction_time_ms": 0.0,
            "chunking_time_ms": 0.0,
            "embedding_time_ms": 0.0,
            "vector_store_time_ms": 0.0,
            "total_time_ms": total_time_ms,
            "success": False,
            "error_message": str(e)
        }
        
        if not LOG_SERVER_URL:
            print("WARNING: LOG_SERVER_URL not configured - error not logged!")
        elif not send_log_to_server("/api/log-upload", log_data):
            print(f"ERROR: Failed to send error log to {LOG_SERVER_URL}")
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-documents")
def load_documents():
    """Load all PDFs and text files from data/ directory into Qdrant"""
    data_dir = Path("data")
    files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt"))
    
    if not files:
        raise HTTPException(status_code=404, detail="No PDF or TXT files found in data/ directory")
    
    points = []
    
    for file in files:
        upload_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Get file size
            file_size_bytes = file.stat().st_size
            
            # Extract text
            extract_start = time.time()
            text = extract_text_from_file(str(file))
            text_extraction_time_ms = (time.time() - extract_start) * 1000
            
            # Chunk it
            chunk_start = time.time()
            chunks = chunk_text(text)
            chunking_time_ms = (time.time() - chunk_start) * 1000
            
            # Embed chunks
            embed_start = time.time()
            file_points = []
            for i, chunk in enumerate(chunks):
                embedding = embed_model.encode(chunk).tolist()
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "source": file.name,
                        "chunk_id": i
                    }
                )
                file_points.append(point)
            embedding_time_ms = (time.time() - embed_start) * 1000
            
            # Upload to vector store
            vector_start = time.time()
            qdrant.upsert(collection_name=COLLECTION_NAME, points=file_points)
            vector_store_time_ms = (time.time() - vector_start) * 1000
            
            total_time_ms = (time.time() - start_time) * 1000
            
            # Log success
            log_data = {
                "upload_id": upload_id,
                "rag_hardware_id": RAG_HARDWARE_ID,
                "filename": file.name,
                "num_chunks": len(chunks),
                "file_size_bytes": file_size_bytes,
                "text_extraction_time_ms": text_extraction_time_ms,
                "chunking_time_ms": chunking_time_ms,
                "embedding_time_ms": embedding_time_ms,
                "vector_store_time_ms": vector_store_time_ms,
                "total_time_ms": total_time_ms,
                "success": True,
                "error_message": None
            }
            
            if LOG_SERVER_URL:
                send_log_to_server("/api/log-upload", log_data)
            else:
                print("WARNING: LOG_SERVER_URL not configured - upload not logged!")
            
            points.extend(file_points)
            
        except Exception as e:
            # Log failure
            total_time_ms = (time.time() - start_time) * 1000
            log_data = {
                "upload_id": upload_id,
                "rag_hardware_id": RAG_HARDWARE_ID,
                "filename": file.name,
                "num_chunks": 0,
                "file_size_bytes": 0,
                "text_extraction_time_ms": 0.0,
                "chunking_time_ms": 0.0,
                "embedding_time_ms": 0.0,
                "vector_store_time_ms": 0.0,
                "total_time_ms": total_time_ms,
                "success": False,
                "error_message": str(e)
            }
            
            if LOG_SERVER_URL:
                send_log_to_server("/api/log-upload", log_data)
            
            print(f"Error processing {file.name}: {str(e)}")
            # Continue with other files
    
    return {
        "message": f"Loaded {len(files)} documents with {len(points)} chunks",
        "files": [f.name for f in files]
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Query the RAG system"""
    
    query_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 1. Embed the query
        embed_start = time.time()
        query_vector = embed_model.encode(request.question).tolist()
        embedding_time_ms = (time.time() - embed_start) * 1000
        
        # 2. Search Qdrant
        search_start = time.time()
        search_results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        vector_search_time_ms = (time.time() - search_start) * 1000
        
        if not search_results:
            # Log failed query
            log_data = {
                "query_id": query_id,
                "rag_hardware_id": RAG_HARDWARE_ID,
                "llm_hardware_id": LLM_HARDWARE_ID,
                "model_name": MODEL_NAME,
                "question": request.question,
                "answer": "",
                "num_chunks_retrieved": 0,
                "avg_similarity_score": 0.0,
                "embedding_time_ms": embedding_time_ms,
                "vector_search_time_ms": vector_search_time_ms,
                "llm_time_ms": 0.0,
                "total_time_ms": (time.time() - start_time) * 1000,
                "estimated_cost_usd": 0.0,
                "success": False,
                "error_message": "No relevant documents found"
            }
            
            if not LOG_SERVER_URL:
                print("WARNING: LOG_SERVER_URL not configured - failed query not logged!")
            elif not send_log_to_server("/api/log-query", log_data):
                print(f"ERROR: Failed to send log to {LOG_SERVER_URL}")
            
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # 3. Build context from retrieved chunks
        context = "\n\n".join([
            f"Source: {hit.payload['source']}\n{hit.payload['text']}"
            for hit in search_results
        ])
        
        # Calculate avg similarity score
        avg_similarity_score = sum(hit.score for hit in search_results) / len(search_results)
        
        # 4. Call OpenAI
        llm_start = time.time()
        prompt = f"""Answer the following question based only on the provided context.

        Context:
        {context}

        Question: {request.question}

        Answer:"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        llm_time_ms = (time.time() - llm_start) * 1000
        
        answer = response.choices[0].message.content
        
        # Get token counts
        if hasattr(response, 'usage') and response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        else:
            # Estimate for local models
            input_tokens = len(prompt) // 4
            output_tokens = len(answer) // 4

        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000
        
        log_data = {
            "query_id": query_id,
            "rag_hardware_id": RAG_HARDWARE_ID,
            "llm_hardware_id": LLM_HARDWARE_ID,
            "model_name": MODEL_NAME,
            "question": request.question,
            "answer": answer,
            "num_chunks_retrieved": len(search_results),
            "avg_similarity_score": avg_similarity_score,
            "embedding_time_ms": embedding_time_ms,
            "vector_search_time_ms": vector_search_time_ms,
            "llm_time_ms": llm_time_ms,
            "total_time_ms": total_time_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": 0.0,  # don't need it but db schema expects it
            "success": True,
            "error_message": None
        }
        print(f"DEBUG: Sending log with times - embed:{embedding_time_ms}, search:{vector_search_time_ms}, llm:{llm_time_ms}, total:{total_time_ms}")
        
        if not LOG_SERVER_URL:
            print("WARNING: LOG_SERVER_URL not configured - query not logged!")
        elif not send_log_to_server("/api/log-query", log_data):
            print(f"ERROR: Failed to send log to {LOG_SERVER_URL}")
        
        # 6. Format sources
        sources = [
            {
                "text": hit.payload['text'],
                "source": hit.payload['source'],
                "score": hit.score
            }
            for hit in search_results
        ]
        
        return QueryResponse(answer=answer, sources=sources)
        
    except HTTPException:
        raise
    except Exception as e:
            # Log unexpected errors
            total_time_ms = (time.time() - start_time) * 1000
            log_data = {
                "query_id": query_id,
                "rag_hardware_id": RAG_HARDWARE_ID,
                "llm_hardware_id": LLM_HARDWARE_ID,
                "model_name": MODEL_NAME,
                "question": request.question,
                "answer": "",
                "num_chunks_retrieved": 0,
                "avg_similarity_score": 0.0,
                "embedding_time_ms": 0.0,
                "vector_search_time_ms": 0.0,
                "llm_time_ms": 0.0,
                "total_time_ms": total_time_ms,
                "estimated_cost_usd": 0.0,
                "success": False,
                "error_message": str(e)
            }
            
            if not LOG_SERVER_URL:
                print("WARNING: LOG_SERVER_URL not configured - error not logged!")
            elif not send_log_to_server("/api/log-query", log_data):
                print(f"ERROR: Failed to send error log to {LOG_SERVER_URL}")
            
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def list_documents():
    """List what's in the vector store"""
    result = qdrant.count(collection_name=COLLECTION_NAME)
    return {
        "total_chunks": result.count,
        "collection": COLLECTION_NAME
    }
