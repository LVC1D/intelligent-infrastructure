use super::VectorStore;
use pyo3::{exceptions, prelude::*};

#[pyclass(name = "VectorStore")]
struct PyVectorStore {
    inner: VectorStore,
}

#[pymethods]
impl PyVectorStore {
    #[new]
    fn new(dimensions: usize) -> Self {
        Self {
            inner: VectorStore::new(dimensions),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn add(&mut self, vector: Vec<f32>) -> PyResult<usize> {
        self.inner
            .add(&vector)
            .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))
    }

    fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<PySearchResult>> {
        let results = self
            .inner
            .search(&query, k)
            .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|r| PySearchResult {
                index: r.index,
                similarity: r.similarity,
            })
            .collect())
    }
}

#[pyclass]
#[derive(Clone)]
struct PySearchResult {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    similarity: f32,
}

#[pymodule]
fn knowledge_search(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVectorStore>()?;
    Ok(())
}
