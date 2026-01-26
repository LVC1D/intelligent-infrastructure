# Week 16: Agent Architecture Mental Model

## What is an Agent?

It's a runnable executor that runs in a loop that is responsible to execute given tasks, that are communicated to it via either the human-ingested prompt or from another agent. Such communications are also powered by the LLM reasoning to support either the programmable or natural conversation between the parties (agent to agent, or agent to human)

## How Do Agents Communicate?
- Pattern from AutoGen: Through conversation programming: 2-step paradigm, one - computing the received responses and translating into something actionable (or generating a response to pass onto another agent), and two - control flow: pre-defined (or dynamically-adjusted) conditions upon which the conversation flows throughout the scope of the task(s)
- Pattern from CrewAI: Individual Agent class with its fully-defined essential attributes and behavior methods to manage the context, tools, and the task(s) at hand to execute over a certain time; and the Agent Executor (the crew orchestrator) - oversees and manages the data produced / passed by the agents, logs the actions / events emitted, keep record of the memory the agents work with and more
- My Rust approach will use: The essential knowledge I obtained form RBES Phase 1, namely: Tokio-centered async runtime, type-safety for GATs / HRTBs and Phantom Types, some Unsafe Rust (i.e.: some FFI if I need to communicate with Python, or for direct memory management), but the essential ideas for this are going to be inspired from the above two points

## How Do They Prevent Deadlocks?
- What I learned from CrewAI: From what I am observing, they have several async def methods - in which I pretty much see an extensive use of if / else statements and try / except statements
- My approach with Tokio: I will use message-=passing via channels, Arc / Mutex - as these directly have a benefit for safely passing tasks between threads and are exactly what will be used in Tokio - especially if paired with Phantom Types to ensure type safety at compile, so that threads do not need to perform more work that what they actually need

## How Does Tool Calling Work?
- Interface pattern: ...
- How my RAG system becomes a tool: ...

## Trait Signatures (Pseudocode)
```rust
#[async_trait]
trait Agent {
    // What methods?
    async fn execute_task(&self, task: Task, tools: &[Box<dyn Tool>]) -> Result<AgentOutput>;
    pub fn get_role(&self) -> &str;
}

trait Tool {
    async fn execute(&self, input: &str) -> Result<String>;
    fn name(&self) -> &str;
    fn description(&self) -> &str;
}

struct Coordinator {
    tools: Vec<Box<dyn Tool>>,
    messages: mpsc::Receiver<AgentMessage>,
    task_queue: HashMap<TaskId, JoinHandle<AgentOutput>>,
    next_task_id: usize,
}
```

## Questions for Monday's Mental Model Check
- How does Tool calling work?
- Besides what I discovered on the CrewAI source code, how do CrewAI and Autogen actually orchestrate the asyncronicity to prevent deadlocks? By relying on the LLM's API wrappers that handle async logic, or did I miss something?
