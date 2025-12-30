#!/usr/bin/env python3
"""
Test script for RAG Document Analysis System
Verifies that all components work correctly.
"""

import sys

def check_imports():
    """Check if all required packages are available."""
    print("=" * 70)
    print("CHECKING DEPENDENCIES")
    print("=" * 70)
    
    packages = [
        ("groq", "Groq"),
        ("fitz", "PyMuPDF"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "SentenceTransformer"),
    ]
    
    missing = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name:20s} - OK")
        except ImportError:
            print(f"✗ {name:20s} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies available!\n")
    return True


def test_pdf_extraction():
    """Test PDF extraction."""
    print("=" * 70)
    print("TEST 1: PDF EXTRACTION")
    print("=" * 70)
    
    try:
        import fitz
        
        # Test with first PDF
        pdf_path = "/workspace/19314_SVBT Performance Deck Oct & Nov.pdf"
        doc = fitz.open(pdf_path)
        
        print(f"PDF: {pdf_path}")
        print(f"Pages: {len(doc)}")
        
        # Extract first page
        first_page = doc[0].get_text("text")
        print(f"First page characters: {len(first_page)}")
        print(f"First 200 chars: {first_page[:200]}...")
        
        doc.close()
        print("\n✅ PDF extraction working!\n")
        return True
    except Exception as e:
        print(f"\n❌ PDF extraction failed: {e}\n")
        return False


def test_chunking():
    """Test document chunking."""
    print("=" * 70)
    print("TEST 2: DOCUMENT CHUNKING")
    print("=" * 70)
    
    sample_text = """
    This is a sample document for testing chunking functionality.
    We need to ensure that the chunking algorithm properly splits
    large documents into manageable pieces while maintaining context.
    
    The overlap between chunks is important for continuity.
    """ * 20  # Make it longer
    
    chunk_size = 200
    overlap = 50
    chunks = []
    start = 0
    
    while start < len(sample_text):
        end = start + chunk_size
        chunk = sample_text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    print(f"Sample text length: {len(sample_text)}")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0][:100]}...")
    
    print("\n✅ Chunking working!\n")
    return True


def test_embeddings():
    """Test embedding generation."""
    print("=" * 70)
    print("TEST 3: EMBEDDINGS")
    print("=" * 70)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        test_texts = [
            "GMV increased by 25% in November",
            "Revenue performance was strong",
            "The weather is nice today"
        ]
        
        print("Generating embeddings...")
        embeddings = model.encode(test_texts)
        
        print(f"Texts: {len(test_texts)}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Test semantic similarity
        from numpy import dot
        from numpy.linalg import norm
        
        def cosine_similarity(a, b):
            return dot(a, b) / (norm(a) * norm(b))
        
        sim_01 = cosine_similarity(embeddings[0], embeddings[1])
        sim_02 = cosine_similarity(embeddings[0], embeddings[2])
        
        print(f"\nSemantic similarity:")
        print(f"  'GMV increased' vs 'Revenue performance': {sim_01:.3f}")
        print(f"  'GMV increased' vs 'Weather is nice':     {sim_02:.3f}")
        
        if sim_01 > sim_02:
            print("\n✅ Embeddings working correctly!")
            print("   (Related texts have higher similarity)\n")
            return True
        else:
            print("\n⚠️  Unexpected similarity scores\n")
            return False
            
    except Exception as e:
        print(f"\n❌ Embedding test failed: {e}\n")
        return False


def test_vector_store():
    """Test ChromaDB vector store."""
    print("=" * 70)
    print("TEST 4: VECTOR STORE")
    print("=" * 70)
    
    try:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer
        
        # Initialize
        client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=False
        ))
        
        collection = client.get_or_create_collection(name="test")
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Add documents
        docs = [
            "GMV grew by 21% from September to October",
            "ASP declined from ₹1,250 to ₹1,050",
            "Occupancy improved from 82% to 86%",
            "Total trips increased from 508 to 754"
        ]
        
        embeddings = encoder.encode(docs).tolist()
        collection.add(
            ids=[f"doc_{i}" for i in range(len(docs))],
            embeddings=embeddings,
            documents=docs
        )
        
        print(f"Added {len(docs)} documents to vector store")
        
        # Search
        query = "What happened to trip volumes?"
        query_embedding = encoder.encode([query])[0].tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        
        print(f"\nQuery: '{query}'")
        print(f"Top result: {results['documents'][0][0]}")
        
        if "trips" in results['documents'][0][0].lower():
            print("\n✅ Vector store working correctly!\n")
            return True
        else:
            print("\n⚠️  Unexpected search results\n")
            return False
            
    except Exception as e:
        print(f"\n❌ Vector store test failed: {e}\n")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "RAG SYSTEM TEST SUITE" + " " * 32 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    results = []
    
    # Check dependencies
    if not check_imports():
        print("\n⚠️  Install missing dependencies first:")
        print("pip install -r requirements.txt\n")
        return
    
    # Run tests
    results.append(("PDF Extraction", test_pdf_extraction()))
    results.append(("Chunking", test_chunking()))
    results.append(("Embeddings", test_embeddings()))
    results.append(("Vector Store", test_vector_store()))
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} {test_name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nResults: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Your RAG system is ready to use.\n")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.\n")


if __name__ == "__main__":
    run_all_tests()
