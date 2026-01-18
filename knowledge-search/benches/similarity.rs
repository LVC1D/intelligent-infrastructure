use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use knowledge_search::VectorStore;
use std::hint::black_box;

fn benchmark_search_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search_scaling");
    for size in [500, 1000, 5000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut store = VectorStore::new(1536);

            // Create and add 500 vectors
            for i in 0..size {
                let vec: Vec<f32> = (0..1536).map(|j| ((i + j) as f32 * 0.1).sin()).collect();
                let _ = store.add(&vec);
            }

            // Create query vector
            let query_vec: Vec<f32> = (0..1536).map(|j| (j as f32 * 0.2).cos()).collect();

            b.iter(|| black_box(store.search(&query_vec, 6)));
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_search_scaling);
criterion_main!(benches);
