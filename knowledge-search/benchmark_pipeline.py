import cProfile
import pstats
from rag_pipeline import RAGPipeline
import time


def profile_query_components(rag: RAGPipeline, question: str):
    # Time embedding
    start = time.perf_counter()
    query_embedding = rag.embed_gen.embed_text(question)
    embed_time = time.perf_counter() - start
    
    # Time search
    start = time.perf_counter()
    results = rag.vec_store.search(query_embedding, 6)
    search_time = time.perf_counter() - start
    
    # Time document retrieval
    start = time.perf_counter()
    doc_ids = [r.index for r in results]
    docs = rag.doc_store.get_documents(doc_ids)
    retrieval_time = time.perf_counter() - start
    
    # Time LLM generation
    context = "\n\n".join(docs)

    system_prompt = """
    You are a helpful assistant.
    Answer questions based on the provided context.
    """

    user_message = f"""Context:
    {context}

    Question: {question}

    Answer the question based on the context above.
    If the context doesn't contain relevant information, say so.
    """

    start = time.perf_counter()
    response = rag.ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    llm_time = time.perf_counter() - start
    
    return {
        "embed_time": embed_time,
        "search_time": search_time,
        "retrieval_time": retrieval_time,
        "llm_time": llm_time,
        "total": embed_time + search_time + retrieval_time + llm_time
    }

def main():
    rag = RAGPipeline(dimensions=1536)
    query_dict = dict()
    query_dict[0] = "How does the RAG pipeline orchestrate the context retrieval?"
    query_dict[1] = "What programming languages am I learning?"
    query_dict[2] = "How do I convert a simple for loop in a one-liner comprehensive version in Python?"
    query_dict[3] = "How does async relate to FFI?"
    query_dict[4] = "How can I visit Mars?"

    for i in range(5):
        res = profile_query_components(rag, query_dict[i])
        e_t = res["embed_time"]
        s_t = res["search_time"]
        r_t = res["retrieval_time"]
        ll_t = res["llm_time"]
        tt = res["total"]

        print(f"Query {i}: {query_dict[i]}")
        print(f"Embed query: {e_t}s")
        print(f"Search:      {s_t}s")
        print(f"Retrieve:    {r_t}s")
        print(f"LLM gen:     {ll_t}s")
        print(f"Total:       {tt}s")
        print("============")



if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
