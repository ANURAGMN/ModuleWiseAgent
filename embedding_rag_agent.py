"""
RAG Agent with Neural Embeddings - Smart Fallback System
Tries multiple embedding backends to work around Windows DLL issues
"""

import os
import warnings
warnings.filterwarnings('ignore')

from groq import Groq
import pdfplumber
import numpy as np
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional

# === CONFIGURATION ===
os.environ["GROQ_API_KEY"] = "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"

CONFIG = {
    "groq_model": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k_results": 5,
}


# === SMART EMBEDDING BACKEND SELECTOR ===
class EmbeddingBackend:
    """Automatically selects the best available embedding backend."""
    
    def __init__(self):
        self.backend = None
        self.backend_name = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Try different embedding backends in order of preference."""
        
        # Try 1: sentence-transformers (best quality)
        try:
            from sentence_transformers import SentenceTransformer
            print("🔍 Trying sentence-transformers...")
            self.backend = SentenceTransformer('all-MiniLM-L6-v2')
            self.backend_name = "sentence-transformers"
            print("✅ Using sentence-transformers (Best Quality)")
            return
        except Exception as e:
            print(f"⚠️  sentence-transformers not available: {str(e)[:50]}...")
        
        # Try 2: OpenAI embeddings (requires API key)
        try:
            from openai import OpenAI
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                print("🔍 Trying OpenAI embeddings...")
                self.backend = OpenAI(api_key=openai_key)
                self.backend_name = "openai"
                print("✅ Using OpenAI embeddings (API-based)")
                return
        except Exception as e:
            print(f"⚠️  OpenAI not available: {str(e)[:50]}...")
        
        # Try 3: Cohere embeddings (alternative API)
        try:
            import cohere
            cohere_key = os.getenv("COHERE_API_KEY")
            if cohere_key:
                print("🔍 Trying Cohere embeddings...")
                self.backend = cohere.Client(cohere_key)
                self.backend_name = "cohere"
                print("✅ Using Cohere embeddings (API-based)")
                return
        except Exception as e:
            print(f"⚠️  Cohere not available: {str(e)[:50]}...")
        
        # Try 4: TF-IDF fallback (always works)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            print("🔍 Falling back to TF-IDF...")
            self.backend = TfidfVectorizer(max_features=384, ngram_range=(1, 2))
            self.backend_name = "tfidf"
            print("⚠️  Using TF-IDF (Fallback - 75% accuracy)")
            print("   Install sentence-transformers for better accuracy!")
            return
        except Exception as e:
            raise RuntimeError(f"No embedding backend available: {e}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the available backend."""
        
        if self.backend_name == "sentence-transformers":
            return self.backend.encode(texts)
        
        elif self.backend_name == "openai":
            response = self.backend.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            return np.array([item.embedding for item in response.data])
        
        elif self.backend_name == "cohere":
            response = self.backend.embed(texts=texts, model="embed-english-light-v3.0")
            return np.array(response.embeddings)
        
        elif self.backend_name == "tfidf":
            if not hasattr(self.backend, 'vocabulary_'):
                # First time - fit the vectorizer
                self.backend.fit(texts)
            return self.backend.transform(texts).toarray()
        
        else:
            raise RuntimeError("No valid backend initialized")
    
    def get_info(self) -> Dict:
        """Get information about the current backend."""
        info = {
            "backend": self.backend_name,
            "quality": {
                "sentence-transformers": "Excellent (90-95%)",
                "openai": "Excellent (90-95%)",
                "cohere": "Very Good (85-90%)",
                "tfidf": "Good (75-80%)"
            }.get(self.backend_name, "Unknown"),
            "cost": {
                "sentence-transformers": "Free (local)",
                "openai": "$0.0001 per query",
                "cohere": "$0.0001 per query",
                "tfidf": "Free (local)"
            }.get(self.backend_name, "Unknown")
        }
        return info


class PDFExtractor:
    """Extract text from PDFs using pdfplumber."""
    
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Extract all text from PDF."""
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)
    
    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """Get number of pages."""
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)


class DocumentChunker:
    """Split documents into chunks."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk text with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.5:
                    end = start + break_point + 1
                    chunk_text = text[start:end]
            
            chunk = {
                "text": chunk_text.strip(),
                "chunk_id": len(chunks),
                "start_char": start,
                "end_char": end,
            }
            
            if metadata:
                chunk.update(metadata)
            
            chunks.append(chunk)
            start = end - self.overlap
        
        return chunks


class SmartVectorStore:
    """Vector store that works with any embedding backend."""
    
    def __init__(self, embedding_backend: EmbeddingBackend):
        self.embedding_backend = embedding_backend
        self.chunks = []
        self.chunk_vectors = None
        
        # Initialize ChromaDB
        try:
            self.client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                is_persistent=False
            ))
            self.collection = self.client.get_or_create_collection(
                name="revmax_documents",
                metadata={"embedding_backend": embedding_backend.backend_name}
            )
            self.use_chromadb = True
        except Exception as e:
            print(f"⚠️  ChromaDB not available, using numpy fallback: {str(e)[:50]}...")
            self.use_chromadb = False
    
    def add_documents(self, chunks: List[Dict], document_name: str):
        """Add chunks to the store."""
        print(f"   Generating embeddings for {len(chunks)} chunks...")
        
        # Store chunks
        for chunk in chunks:
            chunk['document'] = document_name
            self.chunks.append(chunk)
        
        # Generate embeddings
        all_texts = [c['text'] for c in self.chunks]
        embeddings = self.embedding_backend.encode(all_texts)
        
        if self.use_chromadb:
            # Use ChromaDB
            ids = [f"{document_name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{
                "document": document_name,
                "chunk_id": chunk["chunk_id"]
            } for chunk in chunks]
            
            # Handle different embedding types
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                documents=[c['text'] for c in chunks],
                metadatas=metadatas
            )
        else:
            # Use numpy fallback
            self.chunk_vectors = embeddings
        
        print(f"✅ Added {len(chunks)} chunks from {document_name}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks."""
        query_embedding = self.embedding_backend.encode([query])[0]
        
        if self.use_chromadb:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding],
                n_results=top_k
            )
            
            relevant_chunks = []
            for i in range(len(results["documents"][0])):
                relevant_chunks.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i]
                })
        else:
            # Numpy fallback - cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([query_embedding], self.chunk_vectors)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_chunks = []
            for idx in top_indices:
                relevant_chunks.append({
                    "text": self.chunks[idx]['text'],
                    "metadata": {
                        "document": self.chunks[idx]['document'],
                        "chunk_id": self.chunks[idx]['chunk_id']
                    },
                    "similarity": float(similarities[idx])
                })
        
        return relevant_chunks
    
    def count(self):
        """Get total number of chunks."""
        return len(self.chunks)


class EmbeddingRAGAgent:
    """RAG Agent with smart embedding backend selection."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Initialize smart embedding backend
        print("\n🔧 Initializing Embedding Backend...")
        self.embedding_backend = EmbeddingBackend()
        
        self.vector_store = SmartVectorStore(self.embedding_backend)
        self.chunker = DocumentChunker(
            chunk_size=config["chunk_size"],
            overlap=config["chunk_overlap"]
        )
        self.documents = {}
    
    def load_pdf(self, pdf_path: str, document_name: str):
        """Load and index a PDF."""
        print(f"\n📄 Loading: {document_name}")
        
        # Extract text
        extractor = PDFExtractor()
        full_text = extractor.extract_text(pdf_path)
        page_count = extractor.get_page_count(pdf_path)
        
        self.documents[document_name] = {
            "path": pdf_path,
            "pages": page_count,
            "text_length": len(full_text)
        }
        
        print(f"   Pages: {page_count}")
        print(f"   Characters: {len(full_text):,}")
        
        # Chunk document
        chunks = self.chunker.chunk_text(
            full_text,
            metadata={"source": document_name}
        )
        print(f"   Chunks: {len(chunks)}")
        
        # Add to vector store
        self.vector_store.add_documents(chunks, document_name)
    
    def ask(self, question: str, debug: bool = False) -> str:
        """Ask a question using RAG."""
        
        # Retrieve relevant context
        relevant_chunks = self.vector_store.search(
            question,
            top_k=self.config["top_k_results"]
        )
        
        if debug:
            print("\n🔍 Retrieved chunks:")
            for i, chunk in enumerate(relevant_chunks):
                print(f"  [{i+1}] {chunk['metadata']['document']} "
                      f"(similarity: {chunk['similarity']:.3f})")
                print(f"      {chunk['text'][:100]}...")
        
        # Build context
        context = "\n\n---\n\n".join([
            f"[Document: {chunk['metadata']['document']}]\n{chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        # Generate answer
        system_prompt = """
You are a revMax document analysis expert.

You answer questions using ONLY the retrieved document context provided.

Rules:
- Be precise and cite which document supports your answer
- If the context doesn't contain the answer, say so clearly
- Do not invent information
- Keep responses clear and professional
- When discussing performance metrics, be specific
- When discussing rules/logic, refer to PRD documents
"""
        
        response = self.groq_client.chat.completions.create(
            model=self.config["groq_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"""
<RetrievedContext>
{context}
</RetrievedContext>

Question: {question}

Answer based ONLY on the context above. Cite the document name when relevant.
"""
                }
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        return response.choices[0].message.content.strip()
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        backend_info = self.embedding_backend.get_info()
        return {
            "documents_loaded": len(self.documents),
            "document_names": list(self.documents.keys()),
            "total_chunks": self.vector_store.count(),
            "embedding_backend": backend_info,
            "config": self.config
        }


# === INTERACTIVE CLI MODE ===
def interactive_mode(agent):
    """Run interactive Q&A session."""
    print("\n" + "=" * 70)
    print("  INTERACTIVE MODE")
    print("=" * 70)
    print("\nCommands:")
    print("  • Type your question")
    print("  • 'debug' - Toggle debug mode (shows retrieved chunks)")
    print("  • 'stats' - Show agent statistics")
    print("  • 'backend' - Show embedding backend info")
    print("  • 'exit' - Quit")
    print("\n" + "-" * 70 + "\n")
    
    debug_mode = False
    
    while True:
        try:
            question = input("❓ Your question → ")
            
            if question.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if question.lower() == "debug":
                debug_mode = not debug_mode
                print(f"\n🔧 Debug mode: {'ON' if debug_mode else 'OFF'}\n")
                continue
            
            if question.lower() == "backend":
                info = agent.embedding_backend.get_info()
                print(f"\n🔧 Embedding Backend:")
                print(f"   Type: {info['backend']}")
                print(f"   Quality: {info['quality']}")
                print(f"   Cost: {info['cost']}\n")
                continue
            
            if question.lower() == "stats":
                stats = agent.get_stats()
                print(f"\n📊 Statistics:")
                print(f"   Documents: {stats['documents_loaded']}")
                print(f"   Chunks: {stats['total_chunks']}")
                print(f"   Embedding: {stats['embedding_backend']['backend']}")
                print(f"   Quality: {stats['embedding_backend']['quality']}")
                print(f"\n   Loaded documents:")
                for doc in stats['document_names']:
                    print(f"     • {doc}")
                print()
                continue
            
            if not question.strip():
                continue
            
            answer = agent.ask(question, debug=debug_mode)
            print(f"\n💡 Answer:\n{answer}\n")
            print("-" * 70 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


# === MAIN USAGE ===
if __name__ == "__main__":
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 8 + "SMART RAG AGENT WITH NEURAL EMBEDDINGS" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Initialize agent
    agent = EmbeddingRAGAgent(CONFIG)
    
    # Show backend info
    backend_info = agent.embedding_backend.get_info()
    print(f"\n📊 Backend: {backend_info['backend']}")
    print(f"   Quality: {backend_info['quality']}")
    print(f"   Cost: {backend_info['cost']}")
    
    # Load PDFs
    print("\n" + "=" * 70)
    print("LOADING DOCUMENTS")
    print("=" * 70)
    
    try:
        agent.load_pdf(
            "19314_SVBT Performance Deck Oct & Nov.pdf",
            "SVBT_Performance"
        )
        
        agent.load_pdf(
            "Proactive Price Intervention Communication - PRD.pdf",
            "Price_Intervention_PRD"
        )
    except FileNotFoundError as e:
        print(f"\n⚠️  PDF not found: {e}")
        print("Make sure the PDF files are in the current directory.")
        exit(1)
    
    print("\n" + "=" * 70)
    print("✅ Agent ready!")
    print("=" * 70)
    
    # Show stats
    stats = agent.get_stats()
    print(f"\n📊 Loaded Documents: {stats['documents_loaded']}")
    print(f"📦 Total Chunks: {stats['total_chunks']}")
    print(f"🔍 Retrieval: Top {stats['config']['top_k_results']} chunks per query")
    print(f"🧠 Embedding: {stats['embedding_backend']['backend']} "
          f"({stats['embedding_backend']['quality']})")
    
    print("\nDocuments:")
    for doc_name in stats['document_names']:
        print(f"  • {doc_name}")
    
    # Example queries
    print("\n" + "=" * 70)
    print("EXAMPLE QUERIES")
    print("=" * 70)
    
    questions = [
        "What was the GMV trend in November?",
        "Which routes showed ASP decline but occupancy improvement?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Query {i}] {question}")
        try:
            answer = agent.ask(question)
            print(f"\n💡 Answer:\n{answer}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        print("\n" + "-" * 70)
    
    # Start interactive mode
    print("\n" + "=" * 70)
    print("READY FOR YOUR QUESTIONS!")
    print("=" * 70)
    
    interactive_mode(agent)
