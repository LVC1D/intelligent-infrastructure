Session Log: Day 5 - RAG Quality & Testing
Date: January 13, 2026
Time: Jan 13th 20:30 / Jan 14th 12:30 PM (excluding sleep and day job hours - total of about 4 - 4.5 hours)

Starting Status

Current build: Day 5 - RAG Quality & Testing
Last completed: Day 4 - Obsidian vault ingestion working (9 files, 120 chunks)
Blockers from yesterday: None, but identified need for quality testing
Today's target: Test RAG quality systematically, add metadata tracking, identify failure modes


Build Log
Time: About 4 hours
What you built:

[List the components you modified: MarkdownChunker, DocStore, ObsidianIngestion, VectorStore]
I modified the following components for an improved context retrieval and an improved answer relevancy in the answer generation:

- MarkdownChunker: I have opted in for chunking the files specifically By Heading 2's 
- ObsidianIngestion: Now considers source file names to add metadata for each chunk to analyze the top-k picked chunks for generating answers
- DocStore: Now accepts the source file names prepended to the document's content
- VectorStore: Instead of sorting the entire embedding vectors, we utilize Rust's `BinaryHeap` as min-heap (the lowest of the top-k at the bottom / end of the list)

What worked:

The above-mentioned amendments have shown the corresponding results as expected
In particular, in regards to the BinaryHeap-to-Vector sorting, did the job correctly because I explicitly took advantage of the `SearchResult`'s `Ord` trait implementation on it to sort the drained vector.

What didn't work:

- Chunking notes by headings only (I have a fair share of "naked" headings that have no subsequent non-heading content, which contributed the most to "context waste" - as in producing garbage chunks that would occupy the top-k results purely based on the semantics of the content ion relation to the prompt
- Claude's approach with `into_sorted_vec().into_iter().map()` on a BinaryHeap - this does not sort the SearchResults correctly, if not at all, hence a tiny time complexity overhead from my explicit sorting (k elements instead of `n` elements) is not only negligible, but virtually the way to go about it 
- usage of small `k` value in our RAG pipeline simulations - though relatively fast throughput, but the chunk size outputted for context retrieval is too narrow and has higher chance of missing out on "juicier" chunks for consideration

Problem-solving:

1. Before starting Day 5, I had no idea at all what the issue with the original chunking was. It gave me the chunks, it gave SOME answer - I thought that was fine.

2. Today's session, however, revealed the core issue with it - which yielded me the idea to chunk out a little more granularly.

3. As mentioned regarding the BinaryHeap sorting, I was mentally wrestling with Claude regarding why its solution was flawed - until I figured out a more explicit approach.

4. (unrelated to this session, but still critical) - prompt engineering and its particularities matters across the entire board of everything AI-related. Even with the inital context provided, even when you optimize the logic to ensure the faithfulness factor remaining as high as possible, it fades off gradually the further you progress with the conversation.

Specific reminders of it, or extra steps to instruct the model to recall / keep in mind again can (and should) re-bump the contextual precision, answer relevancy AND in turn - faithfulness of the response / thread.

What Shipped Today

What was shipped today is a major refinement of the existing basic RAG Pipeline that uses a cheap and efficient GPT model (40-mini) and the "text-embeddings-3-small" embedding formula that can accurately and efficiently crawl my very own Obsidian notes and generate appropriate answers based on the query I feed it.

Struggles / Learnings
What was hard: The management of source file names when modifying the `ingest_directory()` method of the ObsidianIngestion class; the overall design layout in code (if I were to have done it completely on my own - as of now); How to handle data insertions / sorting in the BinaryHeap (had to look up the Reverse() wrapping / unwrapping) 

What you learned: Chunking approach - as well as the decision on what `k` factor should be - is critical in deciding how accurate and comprehensive I want my RAG Pipeline I want to work continuously.

RBES skill applied: Primarily it was performance optimization through the debugging and testing. A lot of RBES skills was not (yet) applied, though I would love to at some point (i.e.: Type systems and async)

Confidence rating: [4.3/5] for [RAG quality debugging / Rust BinaryHeap]
