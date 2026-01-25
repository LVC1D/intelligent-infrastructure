from embeddings import EmbeddingGenerator
from knowledge_search import VectorStore
from docstore import DocStore
import openai


class RAGPipeline:
    def __init__(self, dimensions: int):
        # Initialize all your components
        # VectorStore, DocStore, EmbeddingGenerator, OpenAI client
        self.doc_store = DocStore()
        self.vec_store = VectorStore(dimensions)
        self.embed_gen = EmbeddingGenerator()
        self.ai_client = openai.OpenAI()

    def add_document(self, text: str, source: str) -> int:
        embedding = self.embed_gen.embed_text(text)
        doc_id = self.doc_store.add_document(text, source)
        vec_idx = self.vec_store.add(embedding)
        assert doc_id == vec_idx
        return doc_id

    def query(self, question: str, top_k: int = 20) -> dict:
        embedded_question = self.embed_gen.embed_text(question)
        results = self.vec_store.search(embedded_question, top_k)
        retrieved_docs = [item.index for item in results]
        retrieved_texts = self.doc_store.get_documents(retrieved_docs)

        id_text_pairs = [
            (doc_id, text)
            for doc_id, text in zip(retrieved_docs, retrieved_texts)
        ]

        if not id_text_pairs:
            return {
                "answer": "No relevant information found after filtering.",
                "context": [],
                "chunk_ids": [],
                "query": question
            }
        # ... LLM call stays same but use top_6_texts ...
        top_6_pairs = id_text_pairs[:6]
        top_6_ids = [doc_id for doc_id, _ in top_6_pairs]
        top_6_texts = [text for _, text in top_6_pairs]
        context = "\n\n".join(top_6_texts)

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

        response = self.ai_client.chat.completions.create(
            model="gpt-5-mini",  # Cheap and fast for testing
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        return {
            "answer": response.choices[0].message.content,
            "context": top_6_texts,
            "chunk_ids": top_6_ids,
            "query": question
        }

def is_useful_chunk(chunk_text: str) -> bool:
    """Filter out metadata and sparse chunks"""
     
    # 1. Skip YAML frontmatter and property sections
    if ': ' in chunk_text:
        chunk_text = chunk_text.split(': ', 1)[1]
    
    # Now filters work correctly
    if chunk_text.strip().startswith('---'):
        return False

    # 2. Minimum content threshold (test 80, 100, 150 chars)
    if len(chunk_text.strip()) < 100:  # Increase from 40
        return False

    # 3. Skip mostly-bullets with little prose
    lines = chunk_text.split('\n')
    bullet_lines = sum(1 for l in lines if l.strip().startswith(('-', '*', '+')))
    if bullet_lines / max(len(lines), 1) > 0.7:  # >70% bullets
        return False
     
    # 4. Require some alphabetic content (not just punctuation/numbers)
    alpha_chars = sum(c.isalpha() for c in chunk_text)
    if alpha_chars < 50:  # At least 50 letters
        return False
    
    return True
