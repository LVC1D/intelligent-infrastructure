use pyo3::{
    PyAny, PyResult, Python,
    types::{PyAnyMethods, PyModule},
};

use std::ffi::CString;
use tokio::{spawn, task::spawn_blocking};

#[tokio::main]
async fn main() -> PyResult<()> {
    let first =
        spawn(async { agent_query(1, "What did we cover in RBES Phase 1?".to_string()).await });

    let second =
        spawn(async { agent_query(1, "What Python concepts did I cover?".to_string()).await });
    let third = spawn(async { agent_query(1, "How to build a RAG pipeline?".to_string()).await });

    let (one, two, three) = tokio::join!(first, second, third);
    let one = one.unwrap();
    let two = two.unwrap();
    let three = three.unwrap();

    println!("{one}");
    println!("{two}");
    println!("{three}");

    Ok(())
}

async fn agent_query(agent_id: usize, question: String) -> String {
    let res = spawn_blocking(move || -> PyResult<_> {
        Python::attach(|py| -> PyResult<_> {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            path.call_method1(
                "append",
                ("/Users/hectorcryo/dev/intelligent-infrastructure/knowledge-search",),
            )?;

            path.call_method1("append", ("/Users/hectorcryo/dev/intelligent-infrastructure/.venv/lib/python3.14/site-packages",))?;

            // Then your file paths
            let rag_path =
                "/Users/hectorcryo/dev/intelligent-infrastructure/knowledge-search/rag_pipeline.py";
            let obs_path = "/Users/hectorcryo/dev/intelligent-infrastructure/knowledge-search/obsidian_ingestion.py";
            let database_source =
                "/Users/hectorcryo/Documents/Knowledge Engineering Vault/Knowledge-Engineering/";

            let rag_code = std::fs::read_to_string(rag_path)?;
            let rag_module = PyModule::from_code(
                py,
                CString::new(rag_code)?.as_c_str(),
                CString::new(rag_path)?.as_c_str(),
                c"rag_pipeline",
            )?;

            let obs_code = std::fs::read_to_string(obs_path)?;
            let obs_module = PyModule::from_code(
                py,
                CString::new(obs_code)?.as_c_str(),
                CString::new(obs_path)?.as_c_str(),
                c"obsidian_ingestion",
            )?;

            let rag_class_inst = rag_module.getattr("RAGPipeline")?.call1((1536,))?;

            let obs_class = obs_module
                .getattr("ObsidianIngestion")?
                .call1((&rag_class_inst,))?;

            let _ = obs_class.call_method1("ingest_directory", (database_source,))?;

            let result = rag_class_inst
                .call_method1("query", (question,))?
                .get_item("answer")?;

            let result_str: String = result.extract()?;

            Ok(result_str)
        })
    }).await.unwrap();

    res.unwrap()
}
