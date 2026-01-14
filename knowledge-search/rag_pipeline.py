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

    def add_document(self, text: str) -> int:
        embedding = self.embed_gen.embed_text(text)
        doc_id = self.doc_store.add_document(text)
        vec_idx = self.vec_store.add(embedding)
        assert doc_id == vec_idx
        return doc_id

    def query(self, question: str, top_k: int = 3) -> dict:
        embedded_question = self.embed_gen.embed_text(question)
        results = self.vec_store.search(embedded_question, top_k)

        retrieved_docs = [item.index for item in results]

        retrieved_texts = self.doc_store.get_documents(retrieved_docs)
        answer = ""

        system_prompt = """
        You are a helpful assistant.
        Answer questions based on the provided context.
        """

        # Build context from retrieved documents
        context = "\n\n".join(retrieved_texts)

        user_message = f"""Context:
        {context}

        Question: {question}

        Answer the question based on the context above.
        If the context doesn't contain relevant information, say so.
        """

        response = self.ai_client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap and fast for testing
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        if not retrieved_texts:
            answer = "I don't have any relevant information to answer that question."
        else:
            answer = response.choices[0].message.content

        return {
            "answer": answer,
            "context": retrieved_texts,  # The list of chunk strings
            "query": question
        }
