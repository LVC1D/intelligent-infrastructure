from mistletoe.block_token import Heading, CodeFence
from mistletoe.span_token import RawText
from mistletoe import Document
from typing import List
from tqdm import tqdm
from rag_pipeline import RAGPipeline
from pathlib import Path
from concurrent.futures.thread import ThreadPoolExecutor
import threading


def walk_tree(node):
    if isinstance(node, RawText):
        return node.content + "\n"

    text = ""
    if getattr(node, 'children', None):
        if isinstance(node, CodeFence):
            text += node.language + "\n"
        for le in node.children:
            text += walk_tree(le)

    return text


class MarkdownChunker:
    """Chunks markdown files by heading boundaries"""

    def chunk_file(self, filepath: str) -> List[str]:
        # Parse with mistletoe
        # Walk AST
        # Split at headings
        # Return list of text chunks
        with open(filepath, 'r') as f:
            doc = Document(f)

        chunks = list()
        chunk = ""
        for node in doc.children:
            if isinstance(node, Heading) and node.level == 1:
                chunk += walk_tree(node)
                continue

            if isinstance(node, Heading) and node.level == 2:
                if len(chunk.strip()) > 40:
                    chunks.append(chunk)
                chunk = walk_tree(node)
            else:
                chunk += walk_tree(node)
        if len(chunk.strip()) > 40:
            chunks.append(chunk)
        return chunks


class ObsidianIngestion:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline
        self.chunker = MarkdownChunker()

    def ingest_directory(self, vault_path: str) -> dict:
        stuff = Path(vault_path).rglob('*.md')
        res_dict = dict.fromkeys(
            ["files_processed", "chunks_created", "embeddings_generated"], 0)
        all_chunks = list()
        source_files = list()

        for file in stuff:
            path_name = Path(file)
            file_chunks = self.chunker.chunk_file(str(path_name))
            res_dict["chunks_created"] += len(file_chunks)
            res_dict["files_processed"] += 1
            all_chunks.extend(file_chunks)
            source_files.extend([path_name.name] * len(file_chunks))

        res_dict["embeddings_generated"] = len(all_chunks)
        self._batch_embed_and_add(all_chunks, source_files)
        return res_dict

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

    
def debug_query_with_ids(rag: RAGPipeline, query: str, top_k: int = 6) -> list:
    """Helper to see chunk IDs and their content for ground truth creation"""
    embedded_question = rag.embed_gen.embed_text(query)
    results = rag.vec_store.search(embedded_question, top_k)
   
    '''
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}\n")
    '''

    for result in results:
        chunk_id = result.index
        chunk_text = rag.doc_store.get_document(chunk_id)
        
        '''
        print(f"Chunk ID: {chunk_id} | Similarity: {result.similarity:.4f}")
        print(chunk_text)
        print()
        '''

    result_ids = [res.index for res in results]
    return result_ids


if __name__ == "__main__":
    from rag_pipeline import RAGPipeline

    rag = RAGPipeline(dimensions=1536)
    ingestion = ObsidianIngestion(rag)

    # Test with ONE file first
    stats = ingestion.ingest_directory(
        # Just one folder
        "/Users/hectorcryo/Documents/Knowledge Engineering Vault/Knowledge-Engineering/")
    print(stats)

    # Verify it worked
    query = "What are some advanced ways in which I could improve a RAG Pipeline?"
    debug_query_with_ids(rag, query, top_k=12)    
