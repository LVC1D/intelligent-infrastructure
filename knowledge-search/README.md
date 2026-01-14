# Knowledge Search - RAG System

> Personal knowledge retrieval using Rust vector search + Python LLM integration

## Overview
[2-3 sentences: What this does, why you built it]

## Features
- High-performance vector similarity search (Rust)
- Semantic chunking of Markdown notes
- Retrieval-Augmented Generation with OpenAI
- Source file metadata tracking

## Architecture

### Components
1. **VectorStore (Rust)** - Cosine similarity search with BinaryHeap optimization
2. **DocStore (Python)** - Document storage with metadata
3. **EmbeddingGenerator (Python)** - OpenAI text-embedding-3-small integration
4. **RAGPipeline (Python)** - Orchestrates retrieval + generation
5. **ObsidianIngestion (Python)** - Markdown parsing and batch embedding

### Data Flow
[Diagram or bullet points showing: Question → Embed → Search → Retrieve → LLM → Answer]

## Setup

### Prerequisites
- Rust 1.75+
- Python 3.11+
- OpenAI API key

### Installation
```bash
# Clone repo
git clone <repo-url>
cd intelligent-infrastructure

# Build Rust library
cd knowledge-search
maturin develop --release

# Install Python dependencies
cd ..
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Basic RAG Query
[Code example: Create pipeline, add docs, query]

### Ingest Obsidian Vault
[Code example: ObsidianIngestion with your vault path]

### Example Queries
[3-4 real queries you've tested with expected output type]

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
- Python: [X tests, Y% coverage]

## Performance

- Vector search: O(n log k) via BinaryHeap
- Chunking: H2-level semantic boundaries
- Current scale: 96 chunks, ~10ms search time

## Project Structure
```
intelligent-infrastructure/
├── knowledge-search/      # Rust vector store
│   ├── src/
│   │   ├── lib.rs
│   │   └── python.rs     # PyO3 FFI bindings
├── rag_pipeline.py        # Python orchestration
├── obsidian_ingestion.py  # Markdown chunking
├── docstore.py
├── embeddings.py
└── tests/
```

## Future Work
- [ ] Add heading metadata to chunks
- [ ] Implement query expansion
- [ ] Web interface for search
- [ ] Hybrid search (keyword + semantic)

## Learning Notes
Built as part of Week 14 in knowledge engineering curriculum.
Key learnings: [2-3 technical insights you gained]
