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

app = FastAPI()

# CORS - allow your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://adamlgent.com"],  # Update with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
qdrant = QdrantClient(host="qdrant", port=6333)  # "qdrant" is the service name in docker-compose
embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
COLLECTION_NAME = "documents"


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT file and add it to the vector store"""
    
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    # Save temporarily
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Extract text
    text = extract_text_from_file(file_path)
    
    # Chunk it
    chunks = chunk_text(text)
    
    # Embed and store
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
    
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    
    return {"message": f"Uploaded {file.filename} with {len(chunks)} chunks"}

def init_collection():
    collections = qdrant.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME not in collection_names:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

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

@app.post("/load-documents")
def load_documents():
    """Load all PDFs and text files from data/ directory into Qdrant"""
    data_dir = Path("data")
    files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt"))
    
    if not files:
        raise HTTPException(status_code=404, detail="No PDF or TXT files found in data/ directory")
    
    points = []
    for file in files:
        # Extract text
        text = extract_text_from_file(str(file))
        
        # Rest stays the same...
        chunks = chunk_text(text)
        
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
            points.append(point)
    
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    
    return {
        "message": f"Loaded {len(files)} documents with {len(points)} chunks",
        "files": [f.name for f in files]
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Query the RAG system"""
    
    # 1. Embed the query
    query_vector = embed_model.encode(request.question).tolist()
    
    # 2. Search Qdrant
    search_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=3  # top 3 chunks
    )
    
    if not search_results:
        raise HTTPException(status_code=404, detail="No relevant documents found")
    
    # 3. Build context from retrieved chunks
    context = "\n\n".join([
        f"Source: {hit.payload['source']}\n{hit.payload['text']}"
        for hit in search_results
    ])
    
    # call openai
    prompt = f"""Answer the following question based only on the provided context.

Context:
{context}

Question: {request.question}

Answer:"""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # or "gpt-4o" if you want the better model
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content
    
    # 5. Format sources
    sources = [
        {
            "text": hit.payload['text'],
            "source": hit.payload['source'],
            "score": hit.score
        }
        for hit in search_results
    ]
    
    return QueryResponse(answer=answer, sources=sources)

@app.get("/documents")
def list_documents():
    """List what's in the vector store"""
    # Just do a simple count query instead of get_collection
    result = qdrant.count(collection_name=COLLECTION_NAME)
    return {
        "total_chunks": result.count,
        "collection": COLLECTION_NAME
    }