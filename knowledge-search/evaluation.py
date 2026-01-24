from dataclasses import dataclass 
from typing import List
from rag_pipeline import RAGPipeline
from obsidian_ingestion import debug_query_with_ids, ObsidianIngestion


@dataclass
class TestQuery:
    query: str
    relevant_chunk_ids: List[int]
    category: str


GROUND_TRUTH = [
    TestQuery(
        query="How do I create an async TCP server in Rust?",
        relevant_chunk_ids=[211, 164, 206, 161, 155],  # You'll determine these by inspection
        category="async"
    ), 
    TestQuery(
        query="What was covered in the Phase 1 of RBES?",
        relevant_chunk_ids=[229, 73, 119, 64, 1],
        category="rbes"
    ),
    TestQuery(
        query="What's the difference between the GATs and HRTBs?",
        relevant_chunk_ids=[142, 221, 138, 219, 140, 216, 165, 141],  # You'll determine these by inspection
        category="adv_types"
    ), 
    TestQuery(
        query="Can you explain to me how to benchmark and profile a Rust codebase?",
        relevant_chunk_ids=[203, 72, 191, 201, 73],
        category="rust_profiling"
    ),
    TestQuery(
        query="What is the difference between safe and unsafe Rust?",
        relevant_chunk_ids=[153, 145, 148, 146, 166],  # You'll determine these by inspection
        category="unsafe"
    ), 
    TestQuery(
        query="How to unsafely and manually manage memory in Rust?",
        relevant_chunk_ids=[153, 123, 145, 148, 179],
        category="unsafe_memory"
    ),
    TestQuery(
        query="Explain to me how the ring buffer works",
        relevant_chunk_ids=[157, 158],  # You'll determine these by inspection
        category="ring_buffer"
    ), 
    TestQuery(
        query="How does WASM work?",
        relevant_chunk_ids=[130, 131, 126, 134, 132, 135, 127, 128],
        category="wasm"
    ),
    TestQuery(
        query="Tell me everything I have learned so far regarding async Rust. Just a short summary is sufficient.",
        relevant_chunk_ids=[123, 224, 226, 210, 205, 162],  # You'll determine these by inspection
        category="all_rust_async"
    ), 
    TestQuery(
        query="How to build a doubly-linked list?",
        relevant_chunk_ids=[149, 88],
        category="dl_list"
    ),
    TestQuery(
        query="What programming languages are covered in the program I am currently learning?",
        relevant_chunk_ids=[120, 245, 76],  # You'll determine these by inspection
        category="langs"
    ), 
    TestQuery(
        query="How did I improve my RAG Pipeline from Week 14?",
        relevant_chunk_ids=[68, 46, 253, 65, 33, 1, 10, 20],
        category="rag_improve"
    ),
    TestQuery(
        query="Tell me about the FFI between Rust and python",
        relevant_chunk_ids=[34, 37, 103, 43, 54],  # You'll determine these by inspection
        category="ffi"
    ), 
    TestQuery(
        query="Why is vector similarity search important in a RAG Pipeline?",
        relevant_chunk_ids=[96, 117, 107],
        category="vec_sim"
    ),
    TestQuery(
        query="How do I go to Mars?",
        relevant_chunk_ids=[],  # Must be completely irrelevant
        category="mars"
    ), 
]

def evaluate_precision_at_k(
    retrieved_chunk_ids: List[int],
    relevant_chunk_ids: List[int],
    k: int
) -> float:
    """
    Calculate Precision@k
    Returns: float between 0.0 and 1.0
    """
    relevant = 0.0
    if len(relevant_chunk_ids) == 0:
        return relevant

    for id in retrieved_chunk_ids:
        if id in relevant_chunk_ids:
            relevant += 1.0

    return relevant / (k * 1.0)

def evaluate_recall_at_k(
    retrieved_chunk_ids: List[int],
    relevant_chunk_ids: List[int],
    k: int
) -> float:
    """
    Calculate Recall@k
    Returns: float between 0.0 and 1.0
    """
    relevant = 0.0
    if len(relevant_chunk_ids) == 0:
        return relevant

    for id in retrieved_chunk_ids:
        if id in relevant_chunk_ids:
            relevant += 1.0

    return relevant / len(relevant_chunk_ids)

def evaluate_rr(
    retrieved_chunk_ids: List[int],
    relevant_chunk_ids: List[int]
) -> float:
    """
    Calculate Mean Reciprocal Rank for single query
    Returns: float between 0.0 and 1.0
    """
    for id in retrieved_chunk_ids:
        if id in relevant_chunk_ids:
            rank = retrieved_chunk_ids.index(id) + 1
            return 1.0 / rank

    return 0.0


def run_evaluation(
    rag: RAGPipeline,
    test_queries: List[TestQuery],
    k: int = 6
) -> dict:
    """
    Run full evaluation across all test queries
    Returns: dict with average metrics
    """
    # Loop through test queries
    # For each: run RAG query, calculate P@k, R@k, MRR
    # Return averages
    evals = dict.fromkeys(["precision@k", "recall@k", "mrr"])

    all_rr_scores = [] 
    all_prcs_scores = []
    all_recall_scores = []

    for i, tq in enumerate(test_queries):
        query_num = i + 1

        retrieved_ids = debug_query_with_ids(rag, tq.query, k)
        rr = evaluate_rr(retrieved_ids, tq.relevant_chunk_ids)
        prcs_k = evaluate_precision_at_k(retrieved_ids, tq.relevant_chunk_ids, k)
        recall_k = evaluate_recall_at_k(retrieved_ids, tq.relevant_chunk_ids, k)

        print(f"Query {query_num}: {tq.query}")
        print(f"Precision@{k}: {prcs_k}")
        print(f"Recall@{k}: {recall_k}")
        print(f"Individual RR: {rr}")
        print()

        all_rr_scores.append(rr)
        all_prcs_scores.append(prcs_k)
        all_recall_scores.append(recall_k)

    evals["precision@k"] = sum(all_prcs_scores) / len(all_prcs_scores)
    evals["recall@k"] = sum(all_recall_scores) / len(all_recall_scores)
    evals["mrr"] = sum(all_rr_scores) / len(all_rr_scores)
    print("=== === === === ===")
    return evals


if __name__ == "__main__":
    rag = RAGPipeline(dimensions=1536)
    ingestion = ObsidianIngestion(rag)
    ingestion.ingest_directory("/Users/hectorcryo/Documents/Knowledge Engineering Vault/Knowledge-Engineering/")

    evaluations = run_evaluation(rag, GROUND_TRUTH)
    print(evaluations)

