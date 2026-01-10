#!/usr/bin/env python3
"""Quick smoke test for Rust FFI boundary"""

from knowledge_search import VectorStore


def test_basic_operations():
    """Test that basic VectorStore operations work from Python"""
    print("Testing VectorStore FFI...")

    # Create store
    store = VectorStore(dimensions=3)
    print("✓ Created VectorStore with dimensions=3")

    # Add vectors
    idx0 = store.add([1.0, 0.0, 0.0])
    idx1 = store.add([0.0, 1.0, 0.0])
    idx2 = store.add([0.0, 0.0, 1.0])
    print(f"✓ Added 3 vectors, got indices: {idx0}, {idx1}, {idx2}")

    # Search
    results = store.search([1.0, 0.0, 0.0], k=2)
    print(f"✓ Search returned {len(results)} results")

    # Verify results structure
    for r in results:
        print(f"  - index={r.index}, similarity={r.similarity:.4f}")

    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0].similarity > results[1].similarity, "Results not sorted by similarity"
    print("✓ Results correctly sorted")

    print("\nAll basic tests passed! ✓")


def test_error_handling():
    """Test that dimension mismatches raise proper exceptions"""
    print("\nTesting error handling...")

    store = VectorStore(dimensions=3)

    try:
        store.add([1.0, 2.0])  # Wrong dimensions
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Dimension mismatch raised ValueError: {e}")

    print("Error handling works! ✓")


def test_similarity_correctness():
    """Test that cosine similarity is actually correct"""
    print("\nTesting similarity correctness...")

    store = VectorStore(dimensions=3)

    # Add identical vector
    store.add([1.0, 0.0, 0.0])

    # Search with same vector - should get similarity = 1.0
    results = store.search([1.0, 0.0, 0.0], k=1)

    similarity = results[0].similarity
    assert abs(
        similarity - 1.0) < 0.0001, f"Expected similarity ~1.0, got {similarity}"
    print(f"✓ Identical vectors have similarity={similarity:.6f}")

    # Add orthogonal vector
    store.add([0.0, 1.0, 0.0])

    # Search - orthogonal vectors should have similarity = 0.0
    results = store.search([1.0, 0.0, 0.0], k=2)
    orthogonal_sim = results[1].similarity
    assert abs(orthogonal_sim) < 0.0001, f"Expected orthogonal similarity ~0.0, got {
        orthogonal_sim}"
    print(f"✓ Orthogonal vectors have similarity={orthogonal_sim:.6f}")

    print("Similarity math is correct! ✓")


if __name__ == "__main__":
    test_basic_operations()
    test_error_handling()
    test_similarity_correctness()
    print("\n" + "="*50)
    print("ALL FFI TESTS PASSED ✓")
    print("="*50)
