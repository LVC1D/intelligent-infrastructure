use criterion::{Criterion, criterion_group, criterion_main};
use knowledge_search::VectorStore;
use std::hint::black_box;

fn benchmark_vector_search(c: &mut Criterion) {
    let mut store = VectorStore::new(1536);

    // Create and add 500 vectors
    for i in 0..500 {
        let vec: Vec<f32> = (0..1536).map(|j| ((i + j) as f32 * 0.1).sin()).collect();
        let _ = store.add(&vec);
    }

    // Create query vector
    let query_vec: Vec<f32> = (0..1536).map(|j| (j as f32 * 0.2).cos()).collect();

    c.bench_function("vector search: BinaryHeap", |b| {
        b.iter(|| black_box(store.search(&query_vec, 6)));
    });
}

criterion_group!(benches, benchmark_vector_search);
criterion_main!(benches);
