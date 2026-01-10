"""Comprehensive FFI integration tests"""
import pytest
from knowledge_search import VectorStore


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
