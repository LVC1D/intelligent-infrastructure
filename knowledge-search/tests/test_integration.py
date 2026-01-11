"""Comprehensive FFI integration tests"""
import pytest
from knowledge_search import VectorStore
from docstore import DocStore
from embeddings import EmbeddingGenerator
from openai import RateLimitError, AuthenticationError, APIConnectionError
from rag_pipeline import RAGPipeline


class TestVectorStoreBasics:
    def test_create_store(self):
        store = VectorStore(dimensions=3)
        assert store is not None

    def test_add_returns_sequential_indices(self):
        store = VectorStore(dimensions=2)
        idx0 = store.add([1.0, 0.0])
        idx1 = store.add([0.0, 1.0])
        idx2 = store.add([0.5, 0.5])
        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

    def test_search_empty_store(self):
        store = VectorStore(dimensions=3)
        results = store.search([1.0, 0.0, 0.0], k=5)
        assert len(results) == 0

    def test_search_k_larger_than_store(self):
        store = VectorStore(dimensions=2)
        store.add([1.0, 0.0])
        store.add([0.0, 1.0])
        results = store.search([1.0, 0.0], k=10)
        assert len(results) == 2  # Only returns what's available


class TestErrorHandling:
    def test_dimension_mismatch_on_add(self):
        store = VectorStore(dimensions=3)
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add([1.0, 2.0])  # Only 2 dims, expected 3

    def test_dimension_mismatch_on_search(self):
        store = VectorStore(dimensions=3)
        store.add([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.search([1.0, 0.0], k=1)  # Query has wrong dims


class TestSimilarityCorrectness:
    def test_identical_vectors_similarity_one(self):
        store = VectorStore(dimensions=3)
        store.add([1.0, 2.0, 3.0])
        results = store.search([1.0, 2.0, 3.0], k=1)
        assert abs(results[0].similarity - 1.0) < 0.0001

    def test_orthogonal_vectors_similarity_zero(self):
        store = VectorStore(dimensions=3)
        store.add([1.0, 0.0, 0.0])
        store.add([0.0, 1.0, 0.0])
        results = store.search([1.0, 0.0, 0.0], k=2)
        # First result should be the identical vector
        assert abs(results[0].similarity - 1.0) < 0.0001
        # Second result should be orthogonal
        assert abs(results[1].similarity - 0.0) < 0.0001

    def test_normalized_vectors(self):
        """Cosine similarity should work with normalized vectors"""
        store = VectorStore(dimensions=2)
        # Add normalized vectors
        store.add([1.0, 0.0])
        store.add([0.0, 1.0])
        store.add([0.7071, 0.7071])  # 45 degrees

        # Search with 45-degree vector
        results = store.search([0.7071, 0.7071], k=3)

        # Should match itself perfectly
        assert results[0].index == 2
        assert abs(results[0].similarity - 1.0) < 0.001


class TestResultOrdering:
    def test_results_sorted_by_similarity(self):
        store = VectorStore(dimensions=2)
        store.add([1.0, 0.0])    # Perfect match to query
        store.add([0.5, 0.5])    # Partial match
        store.add([0.0, 1.0])    # Orthogonal

        results = store.search([1.0, 0.0], k=3)

        # Results must be sorted descending by similarity
        for i in range(len(results) - 1):
            assert results[i].similarity >= results[i+1].similarity


class TestDocStore:
    def test_docstore_initialized(self):
        store = DocStore()
        store.add_document("prompt one")
        store.add_document("prompt two")
        assert len(store.store) == 2

    def test_docstore_instances_isolated(self):
        """Each DocStore should have independent storage"""
        store1 = DocStore()
        store2 = DocStore()

        store1.add_document("store1 doc")
        store2.add_document("store2 doc")

        assert len(store1.store) == 1
        assert len(store2.store) == 1
        assert store1.get_document(0) == "store1 doc"
        assert store2.get_document(0) == "store2 doc"

    def test_get_nonexistent_document(self):
        store = DocStore()
        assert store.get_document(1) is None

    def test_get_documents_with_mixed_ids(self):
        store = DocStore()

        store.add_document("prompt one")
        store.add_document("prompt two")
        store.add_document("prompt three")
        store.add_document("prompt four")

        results = store.get_documents([1, 3, 5, 7])
        assert len(results) == 2

    def test_add_returns_correct_ids(self):
        store = DocStore()

        store.add_document("prompt one")
        store.add_document("prompt two")
        store.add_document("prompt three")
        store.add_document("prompt four")

        assert list(store.store.keys()) == [0, 1, 2, 3]


class TestEmbeddingGenerator:
    def test_embed_gen(self):
        embed_gen = EmbeddingGenerator()
        embed_vector = embed_gen.embed_text("This is a test")
        assert len(embed_vector) == 1536  # Exact dimension

    def test_invalid_api_key_raises_auth_error(self):
        """Test that invalid key raises AuthenticationError"""
        embed_gen = EmbeddingGenerator()
        embed_gen.client.api_key = "invalid-key"

        with pytest.raises(AuthenticationError):
            embed_gen.embed_text("test")

    def test_empty_string_raises_value_error(self):
        """Test that empty string is rejected"""
        embed_gen = EmbeddingGenerator()

        # Use ValueError, not Exception
        with pytest.raises(ValueError, match="Text cannot be empty or whitespace"):
            embed_gen.embed_text("")


class TestRAGPipeline:
    def test_rag_add_document(self):
        rag = RAGPipeline(dimensions=1536)
        doc_id = rag.add_document("Test document about Rust")
        assert doc_id == 0

        doc_id2 = rag.add_document("Another document")
        assert doc_id2 == 1

    def test_debug_retrieval(self):
        rag = RAGPipeline(dimensions=1536)
        rag.add_document("Week 1: Built TCP server")

        # See what search returns
        results = rag.vec_store.search(rag.embed_gen.embed_text("Week 1"), k=1)
        print(f"Search results: {results}")
        print(f"Type of results: {type(results)}")
        print(f"First result: {results[0]}")
        print(f"Type of first result: {type(results[0])}")

        # Try to get documents
        retrieved = rag.doc_store.get_documents(results)
        print(f"Retrieved texts: {retrieved}")

    def test_rag_retrieval(self):
        rag = RAGPipeline(dimensions=1536)
        rag.add_document("Async Rust with Tokio")
        rag.add_document("Baking cookies at 350°F")
        rag.add_document("Rust performance optimization")

        answer = rag.query("What did I learn about Rust?", top_k=2)

        # Answer should reference Rust docs, not cookies
        assert "Rust" in answer or "async" in answer or "performance" in answer
        assert "cookies" not in answer.lower()

    def test_rag_with_real_questions(self):
        rag = RAGPipeline(dimensions=1536)

        # Add some RBES content
        rag.add_document(
            "Week 1: Built TCP server with Tokio, handled 500 connections")
        rag.add_document("Week 6: Learned cache optimization and SIMD")
        rag.add_document("Week 8: Implemented custom allocators")

        answer = rag.query("What did I build in Week 1?")
        print(f"Answer: {answer}")

        # Should mention TCP or Tokio or connections
        assert any(word in answer.lower()
                   for word in ["tcp", "tokio", "server", "connection"])
