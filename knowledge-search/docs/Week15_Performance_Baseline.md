# Week 15 Performance Baseline
**Date:** January 16, 2026  
**System:** 91-chunk knowledge base from Obsidian vault

## 1. Rust Vector Search Performance (Synthetic Data)

| Vectors | Search Time (k=6) | Scaling |
|---------|-------------------|---------|
| 500     | 663µs            | 1x      |
| 1,000   | 1.33ms           | 2x      |
| 5,000   | 6.72ms           | 10x     |
| 10,000  | 13.8ms           | 20x     |

**Complexity:** O(n) as expected with BinaryHeap
**Conclusion:** Search scales linearly, performs well

## 2. Python RAG Pipeline (Real Data, 91 chunks)

```zsh
Query 0: How does the RAG pipeline orchestrate the context retrieval?
Embed query: 0.8487882500048727s
Search:      3.0499999411404133e-05s
Retrieve:    3.416993422433734e-06s
LLM gen:     0.7317050830024527s
Total:       1.5805272500001593s
============
Query 1: What programming languages am I learning?
Embed query: 0.3150861670001177s
Search:      3.700000524986535e-05s
Retrieve:    1.0415998985990882e-05s
LLM gen:     0.8794960410014028s
Total:       1.1946296240057563s
============
Query 2: How do I convert a simple for loop in a one-liner comprehensive version in Python?
Embed query: 1.3622555839974666s
Search:      5.191699892748147e-05s
Retrieve:    4.583998816087842e-06s
LLM gen:     4.581620666998788s
Total:       5.943932751993998s
============
Query 3: How does async relate to FFI?
Embed query: 0.4034292500000447s
Search:      5.133400554768741e-05s
Retrieve:    4.874993464909494e-06s
LLM gen:     2.8122279579984024s
Total:       3.2157134169974597s
============
Query 4: How can I visit Mars?
Embed query: 0.29237450000073295s
Search:      5.191699892748147e-05s
Retrieve:    7.5410061981529e-06s
LLM gen:     0.7570282499946188s
Total:       1.0494622080004774s
============
```

## 3. Bottleneck Analysis

**Time Distribution (Average):**
- OpenAI Embedding: ~640ms (26%)
- OpenAI Generation: ~1950ms (74%)
- Rust Search: ~40µs (0.002%)
- Doc Retrieval: ~6µs (0.0002%)

**Key Findings:**
It appears two components contribute the most to the slowdown of the RAG Pipeline:

1. Embedding generation (0.29s - 1.36s): high variance due to network
2. LLM Generation (0.74s - 4.58s): similar but the variance is even greater, potentially due to response delay & length

## 4. Optimization Targets (Days 2-5)

**Target 1: Embedding Cache (Day 2)**
- Implement: In-memory dict cache keyed by query text
- Expected improvement: 0.64s saved per repeated query
- Trade-off: ~6KB memory per unique query
- Reasoning: Simple implementation, measurable impact for repeated queries, teaches caching patterns

**Target 2: Retrieval Quality Evaluation (Day 4)**
- Create evaluation dataset: 20-30 queries with ground truth chunks
- Implement metrics: Precision@6, Recall@6, MRR
- Baseline current system quality
- Reasoning: Can't optimize retrieval without measuring it first, foundational for Days 5+

**Target 3: Query Expansion (Day 3) + Hybrid Search (Day 5)**

*Day 3: LLM-based Query Expansion*
- Implement `expand_query()` that uses GPT-4o-mini to generate related keywords
- Append keywords to original query before embedding
- Expected cost: +500ms latency, few cents per query
- Measure: Latency only (quality measurement needs Day 4 eval dataset)

*Day 5: Hybrid Search (Vector + BM25)*
- Build BM25 index using `rank-bm25` library
- Implement dual search: vector (semantic) + BM25 (keyword)
- Combine scores with weighted fusion
- Measure: Does hybrid improve Precision@6 vs baseline vs expansion-only?

**Day 4 prerequisite:** Evaluation dataset required to measure quality improvements

**Reasoning:** Both techniques are complementary - expansion improves vector search input, hybrid search combines semantic + lexical matching. Ambitious but scoped across 3 days with measurement checkpoints.

**NOT worth optimizing:**
- Rust vector search (40µs → even 10x faster saves nothing)
- Python doc retrieval (6µs → already negligible)
- HNSW or vector databases (search isn't the bottleneck)

## Day 3: Parallel Batch embedding

I have opted in for the `ThreadPoolExecutor` from the `concurrent.futures` library to add concurrency to our batch-embedding logic. Here are the results:

|   Method   |      Time     |    Throughput    |    Speedup   |
| ---------- | ------------- | ---------------- | ------------ |
| Sequential | 1m 55s (115s) | 2.20 chunks/sec  | 11.5x faster |
| Concurrent |      10s      | 23.21 chunks/sec |      N/A     |  

The code (concurrent):
```py 
def _batch_embed_and_add(self, chunks: List[str], source_files: List[str]):
        valid_chunks = [c for c in chunks if c and c.strip()]
        valid_sources = [source_files[i] for i in range(
            len(chunks)) if chunks[i] and chunks[i].strip()]

        if len(valid_chunks) < len(chunks):
            print(f"Filtered out {len(chunks) -
                  len(valid_chunks)} empty chunks")

        n = 20
        split_list = [valid_chunks[i:i + n]
                      for i in range(0, len(valid_chunks), n)]
        split_sources = [valid_sources[i:i + n]
                         for i in range(0, len(valid_sources), n)]

        with tqdm(total=len(valid_chunks)) as pbar:
            with ThreadPoolExecutor() as executor:
                print(f"Starting {len(split_list)} batches across 5 threads")
                all_embeddings = list(executor.map(self.rag.embed_gen.embed_batch, split_list))
                print(f"Completed all batches")

                for batch_embeddings, sources, chunks in zip(all_embeddings, split_sources, split_list):
                    for embedding, source, chunk in zip(batch_embeddings, sources, chunks):
                        self.rag.vec_store.add(embedding)        # Fast: 40µs
                        self.rag.doc_store.add_document(chunk, source)
                        pbar.update(1)
```

