from typing import List
from rag_pipeline import RAGPipeline
from obsidian_ingestion import ObsidianIngestion
import asyncio


class Coordinator:
    # consider adding a vault_path arg later
    def __init__(self):
        self.rag = RAGPipeline(dimensions=1536)
        ingestor = ObsidianIngestion(self.rag)
        ingestor.ingest_directory("/users/hectorcryo/Documents/Knowledge Engineering Vault/Knowledge-Engineering/")
    
    async def agent_query(self, question: str) -> str:
        res = await asyncio.to_thread(self.rag.query, question, 6)
        answer = res["answer"]
        return answer

    async def agent_queries(self, questions: List[str]) -> List[str]:
        coroutines = [self.agent_query(q) for q in questions]
        res = await asyncio.gather(*coroutines)
        return res

async def main():
    questions = [
        "How does WASM work?",
        "What did I build throughout the RBES Phase 1?",
        "How to build a ring buffer?",
        "What are some of the reports that you have got?"
    ]

    coord = Coordinator()
    results = await coord.agent_queries(questions)
    print(results)

asyncio.run(main())
