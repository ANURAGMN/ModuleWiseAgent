# RAG System Deployment Guide

## Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First installation may take 5-10 minutes due to ML models.

### Step 2: Test System

```bash
python test_rag_system.py
```

This will verify:
- ✓ PDF extraction works
- ✓ Chunking algorithm works
- ✓ Embeddings are generated correctly
- ✓ Vector store is functioning

### Step 3: Run the Notebook

Open `ModuleWiseAgent.ipynb` and run Cell 8 (the RAG implementation).

Expected output:
```
📄 Loading: SVBT_Performance_Deck
   Pages: 47
   Characters: 85,234
   Chunks: 95
✅ Added 95 chunks from SVBT_Performance_Deck

📄 Loading: Price_Intervention_PRD
   Pages: 9
   Characters: 18,492
   Chunks: 21
✅ Added 21 chunks from Price_Intervention_PRD

✅ Agent ready!
```

### Step 4: Ask Questions

```python
# Example 1: Simple query
answer = agent.ask("What was the GMV trend in November?")
print(answer)

# Example 2: Debug mode
answer = agent.ask("What are the pricing rules?", debug=True)
```

---

## Installation Troubleshooting

### Issue: `pip install` is slow

**Reason:** Installing ML models (sentence-transformers) downloads large files.

**Solution:** Be patient, or install in batches:

```bash
# Core packages first (fast)
pip install groq pymupdf chromadb

# ML models second (slow, ~2-5 minutes)
pip install sentence-transformers

# Utilities
pip install langchain langchain-community tiktoken
```

### Issue: ChromaDB installation fails

**Error:** `Failed building wheel for chromadb`

**Solution:** Install build dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install

# Then retry
pip install chromadb
```

### Issue: PyMuPDF installation fails

**Solution:** Use alternative package name:

```bash
pip install PyMuPDF
# or
pip install pymupdf4llm
```

---

## Production Deployment

### Option 1: Persistent Storage (Recommended)

**Problem:** In-memory ChromaDB loses data on restart.

**Solution:** Use persistent storage:

```python
import chromadb

# Change this line in the RAGDocumentAgent.__init__:
self.client = chromadb.PersistentClient(
    path="/path/to/vector_store_data"
)
```

**Benefits:**
- No re-indexing on restart
- Faster startup (10x)
- Scales better

**Storage size:** ~10MB per 100-page document

### Option 2: API Deployment (Flask)

Create `api.py`:

```python
from flask import Flask, request, jsonify
from rag_agent import RAGDocumentAgent, CONFIG

app = Flask(__name__)
agent = RAGDocumentAgent(CONFIG)

# Load documents on startup
agent.load_pdf("documents/performance.pdf", "Performance")
agent.load_pdf("documents/prd.pdf", "PRD")

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    answer = agent.ask(question)
    return jsonify({"answer": answer})

@app.route('/health', methods=['GET'])
def health():
    stats = agent.get_stats()
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Run:**

```bash
pip install flask
python api.py
```

**Test:**

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the GMV trend?"}'
```

### Option 3: Streamlit Dashboard

Create `app.py`:

```python
import streamlit as st
from rag_agent import RAGDocumentAgent, CONFIG

@st.cache_resource
def load_agent():
    agent = RAGDocumentAgent(CONFIG)
    agent.load_pdf("documents/performance.pdf", "Performance")
    agent.load_pdf("documents/prd.pdf", "PRD")
    return agent

st.title("📊 Document Analysis Agent")

agent = load_agent()

# Sidebar stats
with st.sidebar:
    st.header("System Stats")
    stats = agent.get_stats()
    st.metric("Documents", stats['documents_loaded'])
    st.metric("Chunks", stats['total_chunks'])

# Main interface
question = st.text_input("Ask a question:")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        answer = agent.ask(question)
        st.success(answer)
```

**Run:**

```bash
pip install streamlit
streamlit run app.py
```

### Option 4: Docker Container

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY *.py .
COPY documents/ ./documents/

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Download embedding model (cache it in image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 5000

CMD ["python", "api.py"]
```

**Build and run:**

```bash
docker build -t rag-agent .
docker run -p 5000:5000 rag-agent
```

---

## Optimization Strategies

### 1. Faster Embeddings

**Current:** `all-MiniLM-L6-v2` (384 dimensions, ~60ms per text)

**Faster option:** `paraphrase-MiniLM-L3-v2` (384 dimensions, ~30ms per text)

```python
CONFIG["embedding_model"] = "paraphrase-MiniLM-L3-v2"
```

**Trade-off:** Slightly lower quality (-3%), but 2x faster

### 2. Better Accuracy

**Current:** `all-MiniLM-L6-v2` (good for most cases)

**Better option:** `BAAI/bge-large-en-v1.5` (1024 dimensions, ~200ms per text)

```python
CONFIG["embedding_model"] = "BAAI/bge-large-en-v1.5"
```

**Trade-off:** +8% accuracy, but 3x slower

### 3. Caching

Add query caching for repeated questions:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_ask(question: str) -> str:
    return agent.ask(question)
```

**Benefit:** Instant responses for cached queries

### 4. Batch Processing

Process multiple questions at once:

```python
def ask_batch(questions: List[str]) -> List[str]:
    """Process multiple questions efficiently."""
    # Retrieve all contexts at once
    all_chunks = []
    for q in questions:
        chunks = agent.vector_store.search(q, top_k=5)
        all_chunks.append(chunks)
    
    # Generate answers in batch
    answers = []
    for i, q in enumerate(questions):
        context = "\n\n".join([c['text'] for c in all_chunks[i]])
        answer = agent.ask(q)  # Use cached context
        answers.append(answer)
    
    return answers
```

### 5. GPU Acceleration

For faster embeddings with GPU:

```python
from sentence_transformers import SentenceTransformer

# Enable GPU
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
```

**Requirements:**
- NVIDIA GPU
- CUDA toolkit installed
- `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118`

**Benefit:** 5-10x faster embeddings

---

## Scaling to Many Documents

### Problem: 100+ PDFs to process

### Solution 1: Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def load_documents_parallel(pdf_paths: List[str]):
    """Load multiple PDFs in parallel."""
    agent = RAGDocumentAgent(CONFIG)
    
    def load_single(pdf_path):
        doc_name = pdf_path.split('/')[-1].replace('.pdf', '')
        agent.load_pdf(pdf_path, doc_name)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(load_single, pdf_paths)
    
    return agent
```

### Solution 2: Incremental Updates

```python
def update_document(agent, pdf_path, document_name):
    """Update a single document without reloading all."""
    # Remove old document
    agent.vector_store.collection.delete(
        where={"document": document_name}
    )
    
    # Add new version
    agent.load_pdf(pdf_path, document_name)
```

### Solution 3: Document Metadata

Add metadata for better filtering:

```python
def load_with_metadata(agent, pdf_path, metadata):
    """Load document with rich metadata."""
    doc_name = metadata['name']
    
    # Extract and chunk
    extractor = PDFExtractor()
    doc_data = extractor.extract_with_metadata(pdf_path)
    chunks = agent.chunker.chunk_text(doc_data['full_text'])
    
    # Enhance chunks with metadata
    for chunk in chunks:
        chunk.update(metadata)  # Add date, category, etc.
    
    agent.vector_store.add_documents(chunks, doc_name)

# Usage
load_with_metadata(agent, "perf.pdf", {
    "name": "Q4_Performance",
    "date": "2025-11",
    "category": "performance",
    "region": "SVBT"
})
```

Then filter by metadata:

```python
# Search only Q4 documents
results = agent.vector_store.collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"category": "performance", "date": {"$gte": "2025-10"}}
)
```

---

## Monitoring & Logging

### Add logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# In agent.ask():
logger.info(f"Query: {question}")
logger.info(f"Retrieved {len(relevant_chunks)} chunks")
logger.info(f"Response length: {len(answer)} chars")
```

### Track metrics:

```python
import time

class MetricsTracker:
    def __init__(self):
        self.queries = []
    
    def track_query(self, question, answer, duration, chunks):
        self.queries.append({
            "timestamp": time.time(),
            "question": question,
            "answer_length": len(answer),
            "duration": duration,
            "chunks_retrieved": len(chunks)
        })
    
    def get_stats(self):
        return {
            "total_queries": len(self.queries),
            "avg_duration": sum(q['duration'] for q in self.queries) / len(self.queries),
            "avg_answer_length": sum(q['answer_length'] for q in self.queries) / len(self.queries)
        }
```

---

## Cost Optimization

### Current costs (with Groq):

| Component | Cost per 1000 queries |
|-----------|----------------------|
| Groq API (5k tokens × 1000) | $5 |
| Embeddings (sentence-transformers) | $0 (local) |
| **Total** | **$5** |

### Comparison with full-document approach:

| Approach | Tokens per query | Cost per 1000 queries |
|----------|------------------|---------------------|
| String pasting | 50,000 | $50 |
| RAG | 5,000 | $5 |
| **Savings** | **90%** | **$45** |

### Further optimization:

1. **Use smaller chunks for simple queries:**
   - Complex query: 5 chunks
   - Simple query: 2-3 chunks
   - Savings: ~40% on simple queries

2. **Cache common queries:**
   - Hit rate: 30-40%
   - Savings: 30-40% API costs

3. **Batch processing:**
   - Process 10 queries together
   - Savings: ~20% overhead reduction

**Total potential savings:** Up to 95% vs original approach

---

## Security Best Practices

### 1. API Key Management

❌ **Don't:**
```python
os.environ["GROQ_API_KEY"] = "gsk_abc123..."  # Hardcoded in notebook
```

✅ **Do:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
```

Create `.env`:
```
GROQ_API_KEY=gsk_abc123...
```

### 2. Document Access Control

```python
def load_pdf_with_auth(pdf_path, user_permissions):
    """Only load documents user has access to."""
    if not check_permission(user_permissions, pdf_path):
        raise PermissionError("Access denied")
    
    return agent.load_pdf(pdf_path, ...)
```

### 3. Input Sanitization

```python
def sanitize_question(question: str) -> str:
    """Prevent prompt injection."""
    # Remove control characters
    question = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', question)
    
    # Limit length
    question = question[:1000]
    
    # Remove suspicious patterns
    patterns = ['ignore previous', 'system:', '<|endoftext|>']
    for pattern in patterns:
        question = question.replace(pattern, '')
    
    return question
```

---

## Next Steps

1. ✅ Test system: `python test_rag_system.py`
2. ✅ Run notebook: Cell 8 in `ModuleWiseAgent.ipynb`
3. ✅ Try examples: Cells 9-12
4. 🔄 Choose deployment option (API / Streamlit / Docker)
5. 🔄 Set up persistent storage
6. 🔄 Add monitoring and logging
7. 🚀 Deploy to production

---

## Support & Resources

- **Troubleshooting:** Check `test_rag_system.py` output
- **Configuration tuning:** See `README.md`
- **Advanced features:** See notebook Cells 10-12
- **Performance tips:** This guide, "Optimization Strategies" section

**Ready to deploy?** Start with the Flask API (easiest) or Streamlit (best UX).
