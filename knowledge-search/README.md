# Knowledge Search - RAG System

> Personal knowledge retrieval using Rust vector search + Python LLM integration

## Overview

This is a real project I built during Week 14 of my curriculum to learn hands-on fundamentals of RAG systems. It's a functional prototype, not production-ready, but it does one thing well: searches your markdown notes for context and generates answers to your questions.

I chose Python as it handles AI logic (embeddings, RAG pipeline), and Rust to handle vector similarity search for performance.

## Features
- High-performance vector similarity search (Rust)
- Semantic chunking of Markdown notes
- Retrieval-Augmented Generation with OpenAI
- Source file metadata tracking

## Architecture

### Components
1. **VectorStore (Rust, via PyO3)** - Cosine similarity search with BinaryHeap optimization. Exposed to Python through PyO3 FFI.
2. **DocStore (Python)** - Document storage with metadata
3. **EmbeddingGenerator (Python)** - OpenAI text-embedding-3-small integration
4. **RAGPipeline (Python)** - Orchestrates retrieval + generation
5. **ObsidianIngestion (Python)** - Markdown parsing (which includes file chunking) and batch embedding

### Data Flow

User question → embed it (OpenAI) → vector-search (Rust) → retrieve top-k results → generate answer (GPT-4)

## Setup

### Prerequisites
- Rust 1.75+
- Python 3.11+
- OpenAI API key

### Installation
```bash
# Clone repo
git clone https://github.com/LVC1D/intelligent-infrastructure.git
cd intelligent-infrastructure

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Build Rust library
cd knowledge-search
maturin develop --release
cd ..

# Set API key
export OPENAI_API_KEY="your-key-here"
```

## Usage

### 1. Prepare Your Vault

In order for this to work, ensure you have a directory with your relevant markdown notes you want to search. Files can be flat or organized in subdirectories.

### 2. Configure Ingestion
In the `__main__` block, edit `obsidian_ingestion.py` and update the hard-coded path:
```python
stats = ingestion.ingest_directory(
    "/path/to/your/markdown/notes/"  # Change this
)
```

### 3. Run Ingestion
```bash
python obsidian_ingestion.py
```

### 4. Expected Output

Here is an example of an output for a prompt: "How does RAG Pipeline orchestrate the context retrieval?"

```zsh
Chunks: 6
100%|█████████████████████████████████████████| 96/96 [00:07<00:00, 12.37it/s]
{'files_processed': 13, 'chunks_created': 96, 'embeddings_generated': 96}
QUERY: How does RAG Pipeline orchestrate the context retrieval?

=== RETRIEVED CHUNKS ===

Chunk 1:
Implementation Patterns Day 3.md: Context:
 Building RAG pipeline
Focus:
 Testing, debugging, API integration



Chunk 2:
RAG-Systems.md.md: What is RAG?
Retrieval-Augmented Generation
 - A pattern where LLMs answer questions using retrieved context instead of relying solely on training data.
Three Stages:
Retrieval
 - Find relevant documents via vector similarity
Augmentation
 - Add context to the prompt
Generation
 -...

Chunk 3:
Python Essentials.md: Language:
 Python 3.14
Learned:
 Week 14, Day 3
Context:
 Building RAG pipeline



Chunk 4:
Day_3_Session_Log.md: What Works Now
python
rag = RAGPipeline(dimensions=1536)
rag.add_document("Week 1: Built TCP server with Tokio") rag.add_document("Week 6: Cache optimization and SIMD") answer = rag.query("What did I learn about async Rust?") # Returns grounded answer from actual documents

P...

Chunk 5:
Day_3_Session_Log.md: Commit

Day 3: Complete RAG pipeline with DocStore, EmbeddingGenerator, and orchestration

- Built DocStore for text storage with 0-indexed IDs
- Implemented EmbeddingGenerator with OpenAI API integration - Created RAGPipeline orchestrating embed → search → retrieve → LLM
- ...

Chunk 6:
RAG-Systems.md.md: The RAG Flow

1. User Question: "What did I learn about async?"
       ↓
2. Embed question → vector [0.2, 0.8, ...]
       ↓
3. Search vector store → IDs [5, 2]
       ↓
4. Retrieve texts → ["Week 1: Built TCP...", "Week 6: Cache..."]
       ↓
5. Format prompt:
   Context: Week 1:...

=== GENERATED ANSWER ===
The RAG Pipeline orchestrates context retrieval through a series of defined stages:

1. **Embedding the Question**: The user question is embedded into a vector representation.

2. **Searching the Vector Store**: This embedded question vector is then used to search the vector store to find relevant document IDs based on vector similarity.

3. **Retrieving Texts**: Once relevant document IDs are identified, the corresponding texts are retrieved from storage.

4. **Formatting the Prompt**: The retrieved texts are formatted as context alongside the original question.

5. **Generating the Answer**: Finally, the LLM generates an answer using the context provided by the retrieved documents.

This structured approach allows the RAG Pipeline to effectively use both the user's query and relevant past information to generate informed responses.
```

## Testing

### Rust Tests
```bash
cd knowledge-search
cargo tarpaulin
```

### Python Tests
```bash
pytest tests/ -v --cov
```

### Coverage
- Rust: 8/8 tests passing
- Python: 23/23 tests passing, coverage breakdown:
```zsh
docstore.py                                                                                                                               21      0   100%
embeddings.py                                                                                                                         15      4    73%
obsidian_ingestion.py                                                                                                                    91     53    42%
rag_pipeline.py                                                                                                                          30      1    97%
tests/conftest.py                                                                                                                         4      0   100%
tests/test_integration.py                                                                                                               154      0   100%
```

Note: obsidian_ingestion.py shows 42% coverage due to __main__ test code not being covered by pytest.

## Performance

- Vector search: ~670µs for 500 vectors (1536-dim, top-6 results)
- Complexity: O(n log k) via BinaryHeap
- Chunking: H2-level semantic boundaries
- Current scale: 96 chunks from 13 markdown files

## Project Structure
```
intelligent-infrastructure/
├── knowledge-search/      # Rust vector store
│   ├── src/
│   │   ├── lib.rs
│   │   └── python.rs     # PyO3 FFI bindings
├   ├─── rag_pipeline.py        # Python orchestration
├   ├─── obsidian_ingestion.py  # Markdown chunking
├   ├─── docstore.py
├   ├─── embeddings.py
└── tests/
```

## Future Work
- [ ] Add heading metadata to chunks
- [ ] Implement query expansion
- [ ] Web interface for search
- [ ] Hybrid search (keyword + semantic)
- [ ] Addition of Phase 1 RBES techniques (type-state patterns, async runtime integration, unsafe optimizations)

## Learning Notes
Built as part of Week 14 in knowledge engineering curriculum.
Key learnings:

- LLMs enhance the quality of their answers beyond the dataset they are trained on thanks to the RAG: Retrieving the context from the vector database, analyze the chunks - and compose the more precise, faithful and relevant answer
- Building a simple vector search that utilizes the cosine similarity formula via Rust helped me solidify the basic behind-the-scenes of how the embeddings work during the context retrieval of the RAG Pipeline
- **Important**: making the first strategy of chunking and sticking with it regardless is far from ideal - testing and simulations with various queries reveals that critical trap 
