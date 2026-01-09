pub struct VectorStore {
    dimensions: usize,
    data: Vec<f32>,
    norms: Vec<f32>,
    count: usize,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct SearchResult {
    pub index: usize,
    pub similarity: f32,
}

impl VectorStore {
    /// Create a new vector store for vectors of given dimensions
    #[must_use]
    #[allow(clippy::must_use_candidate)]
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            data: Vec::new(),
            norms: Vec::new(),
            count: 0,
        }
    }

    /// Add a vector to the store. Returns its index.
    /// Panics if vector length != dimensions
    #[must_use]
    #[allow(clippy::must_use_candidate)]
    #[allow(clippy::missing_panics_doc)]
    pub fn add(&mut self, vector: &[f32]) -> usize {
        assert!(
            vector.len() == self.dimensions,
            "The passed-in vector's length doens't correspond to the dimensions size"
        );
        let idx = self.count;

        for &x in vector {
            self.data.push(x);
        }
        self.norms.push(compute_norm(vector));
        self.count += 1;

        idx
    }

    /// Search for top-k most similar vectors to query
    /// Returns results sorted by similarity (highest first)
    #[must_use]
    #[allow(clippy::must_use_candidate)]
    #[allow(clippy::missing_panics_doc)]
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        let query_norm = compute_norm(query);
        let mut res: Vec<SearchResult> = vec![];

        for i in 0..self.count {
            let cos_sim = cosine_similarity(
                query,
                query_norm,
                &self.data[i * self.dimensions..(i + 1) * self.dimensions],
                self.norms[i],
            );

            res.push(SearchResult {
                index: i,
                similarity: cos_sim,
            });
        }

        res.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        res.truncate(k);

        res
    }
}

// Helper functions you'll need (private):
fn compute_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn compute_norm(v: &[f32]) -> f32 {
    let res: f32 = v.iter().fold(0.0, |acc, &x| acc + x.powi(2));
    res.sqrt()
}

fn cosine_similarity(a: &[f32], a_norm: f32, b: &[f32], b_norm: f32) -> f32 {
    compute_dot(a, b) / (a_norm * b_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_new_store() {
        // Create store, verify dimensions
        let store = VectorStore::new(384);
        assert_eq!(store.dimensions, 384);
    }

    #[test]
    fn test_add_vector() {
        // Add a vector, check it returns index 0
        // Add another, check it returns index 1
        let mut store = VectorStore::new(3);

        assert_eq!(store.add(&[3.5, 4.7, 6.8]), 0);
        assert_eq!(store.add(&[3.5, 4.7, 6.8]), 1);
    }

    #[test]
    #[should_panic(
        expected = "The passed-in vector's length doens't correspond to the dimensions size"
    )]
    fn test_add_wrong_dimensions() {
        // Try to add vector with wrong dimensions
        let mut store = VectorStore::new(4);

        assert_eq!(store.add(&[3.5, 4.7, 6.8]), 0);
    }

    #[test]
    fn test_search_single_vector() {
        // Add one vector, search for itself
        // Should return similarity = 1.0
        let mut store = VectorStore::new(3);
        let vec = vec![3.5, 4.7, 6.8];

        let _ = store.add(&vec);
        let res = store.search(&vec, 1); // ← Search with the vector itself

        assert_eq!(res.len(), 1);
        assert_eq!(res[0].index, 0);
        assert_relative_eq!(res[0].similarity, 1.0);
    }

    #[test]
    fn test_search_multiple_vectors() {
        // Add 3 vectors, search, verify ranking
        let mut store = VectorStore::new(3);

        let _ = store.add(&[3.5, 4.7, 6.8]);
        let _ = store.add(&[4.5, 4.1, 1.8]);
        let _ = store.add(&[1.5, 2.7, 1.7]);

        let res = store.search(&[4.2, 3.5, 1.0], 3);
        assert_relative_eq!(
            res[0].similarity,
            cosine_similarity(
                &[4.2, 3.5, 1.0],
                compute_norm(&[4.2, 3.5, 1.0]),
                &[4.5, 4.1, 1.8],
                compute_norm(&[4.5, 4.1, 1.8])
            )
        );
        assert_relative_eq!(
            res[2].similarity,
            cosine_similarity(
                &[4.2, 3.5, 1.0],
                compute_norm(&[4.2, 3.5, 1.0]),
                &[3.5, 4.7, 6.8],
                compute_norm(&[3.5, 4.7, 6.8])
            )
        );
    }

    #[test]
    fn test_search_top_k() {
        // Add 5 vectors, search with k=2
        // Verify only 2 results returned
        let mut store = VectorStore::new(3);

        let _ = store.add(&[3.5, 4.7, 6.8]);
        let _ = store.add(&[4.5, 4.1, 1.8]);
        let _ = store.add(&[1.5, 2.7, 1.7]);
        let _ = store.add(&[3.7, 9.7, 12.7]);
        let _ = store.add(&[8.5, 4.9, 4.0]);
        let res = store.search(&[4.2, 3.5, 1.0], 2);

        assert_eq!(res.len(), 2);
    }

    #[test]
    fn test_cosine_similarity_parallel_vectors() {
        // Test [1,0] and [2,0] → should be 1.0
        let cos_sim = cosine_similarity(
            &[1.0, 0.0],
            compute_norm(&[1.0, 0.0]),
            &[2.0, 0.0],
            compute_norm(&[2.0, 0.0]),
        );

        assert_relative_eq!(cos_sim, 1.0);
    }

    #[test]
    fn test_cosine_similarity_perpendicular() {
        // Test [1,0] and [0,1] → should be 0.0
        let cos_sim = cosine_similarity(
            &[1.0, 0.0],
            compute_norm(&[1.0, 0.0]),
            &[0.0, 1.0],
            compute_norm(&[0.0, 1.0]),
        );

        assert_relative_eq!(cos_sim, 0.0);
    }
}
