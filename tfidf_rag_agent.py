"""
RAG Agent with TF-IDF Embeddings
Using pdfplumber (no DLL dependencies)
Perfect for systems without admin access
"""
 
import os
from groq import Groq
import pdfplumber  # Changed from fitz/PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
 
# === CONFIGURATION ===
os.environ["GROQ_API_KEY"] = "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"
 
CONFIG = {
    "groq_model": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k_results": 20,
}
 
 
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
 
 
class TFIDFVectorStore:
    """Vector store using TF-IDF (no API keys needed)."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=1,
            stop_words='english'
        )
        self.chunks = []
        self.chunk_vectors = None
        self.is_fitted = False
    
    def add_documents(self, chunks: List[Dict], document_name: str):
        """Add chunks to the store."""
        for chunk in chunks:
            chunk['document'] = document_name
            self.chunks.append(chunk)
        
        # Fit vectorizer on all chunks
        all_texts = [c['text'] for c in self.chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(all_texts)
        self.is_fitted = True
        
        print(f"✅ Added {len(chunks)} chunks from {document_name}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks using cosine similarity."""
        if not self.is_fitted:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]['text'],
                "metadata": {
                    "document": self.chunks[idx]['document'],
                    "chunk_id": self.chunks[idx]['chunk_id']
                },
                "similarity": float(similarities[idx])
            })
        
        return results
    
    def count(self):
        """Get total number of chunks."""
        return len(self.chunks)
 
 
class TFIDFRAGAgent:
    """RAG Agent using TF-IDF (no heavy dependencies)."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.vector_store = TFIDFVectorStore()
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
        
        # Generate answer with Groq
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
        return {
            "documents_loaded": len(self.documents),
            "document_names": list(self.documents.keys()),
            "total_chunks": self.vector_store.count(),
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
            
            if question.lower() == "stats":
                stats = agent.get_stats()
                print(f"\n📊 Statistics:")
                print(f"   Documents: {stats['documents_loaded']}")
                print(f"   Chunks: {stats['total_chunks']}")
                print(f"   Chunk size: {stats['config']['chunk_size']} chars")
                print(f"   Top-K retrieval: {stats['config']['top_k_results']}")
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
    print("║" + " " * 10 + "TF-IDF RAG AGENT (No Admin Required)" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\nInitializing agent...")
    
    # Create agent
    agent = TFIDFRAGAgent(CONFIG)
    
    # Load PDFs
    try:
        agent.load_pdf(
            "19314_SVBT Performance Deck Oct & Nov.pdf",
            "SVBT_Performance"
        )
        
        agent.load_pdf(
            "Proactive Price Intervention Communication - PRD.pdf",
            "Price_Intervention_PRD"
        )
        agent.load_pdf(
            "RuleConfig.pdf",
            "RuleConfig_PRD"
        )
        #CGE_Config_1_Bangalore_Khammam
        agent.load_pdf(
            "CGE_Config_1_Bangalore_Khammam.pdf",
            "CGE_Config_1_Bangalore_Khammam_PRD"
        )
        agent.load_pdf(
            "CGE_Config_2_Bangalore_Khammam.pdf",
            "CGE_Config_2_Bangalore_Khammam_PRD"
        )
        agent.load_pdf(
            "CGE_Config_3_Bangalore_Khammam.pdf",
            "CGE_Config_3_Bangalore_Khammam_PRD"
        )
        agent.load_pdf(
            "CGE_Config_4_Bangalore_Khammam.pdf",
            "CGE_Config_4_Bangalore_Khammam_PRD"
        )
        agent.load_pdf(
            "svbt_Bangalore_Khammam.pdf",
            "svbt_Bangalore_Khammam_PRD"
        )
        #Cge And Fare Epoch Configuration Bangalore Khammam
       # agent.load_pdf(
       #     "Cge And Fare Epoch Configuration Bangalore Khammam.pdf",
       #     "Cge And Fare Epoch Configuration Bangalore Khammam_PRD"
       # )
        #Cge And Fare Epoch Explained For Rag
        agent.load_pdf(
            "Cge And Fare Epoch Explained For Rag.pdf",
            "Cge And Fare Epoch Explained For Rag_PRD"
        )
    except FileNotFoundError as e:
        print(f"\n⚠️  PDF not found: {e}")
        print("Make sure the PDF files are in the current directory.")
        print("\nCurrent directory:", os.getcwd())
        exit(1)
    except Exception as e:
        print(f"\n❌ Error loading PDFs: {e}")
        exit(1)
    
    print("\n" + "=" * 70)
    print("✅ Agent ready!")
    print("=" * 70)
    
    # Show stats
    stats = agent.get_stats()
    print(f"\n📊 Loaded Documents: {stats['documents_loaded']}")
    print(f"📦 Total Chunks: {stats['total_chunks']}")
    print(f"🔍 Retrieval: Top {stats['config']['top_k_results']} chunks per query")
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
    
    # Uncomment to start interactive CLI:
    interactive_mode(agent)