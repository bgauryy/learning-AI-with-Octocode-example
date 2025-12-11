# 🤖 AI Agentic Development Best Practices
## Comprehensive Guide from Top GitHub Repositories

**Compiled from:** Oracle AI Developer Hub, OpenAI Agents SDK, Microsoft AutoGen, CrewAI, Mem0, LlamaIndex

---

## Table of Contents

1. [Overview & Core Concepts](#1-overview--core-concepts)
2. [Memory Engineering](#2-memory-engineering)
3. [Context Engineering](#3-context-engineering)
4. [Agent Architectures](#4-agent-architectures)
5. [Multi-Agent Patterns](#5-multi-agent-patterns)
6. [RAG Best Practices](#6-rag-best-practices)
7. [Tool Design Patterns](#7-tool-design-patterns)
8. [Session & State Management](#8-session--state-management)
9. [Orchestration Patterns](#9-orchestration-patterns)
10. [Production Considerations](#10-production-considerations)
11. [References](#11-references)

---

## 1. Overview & Core Concepts

### What is an AI Agent?

An AI agent is a system that can perceive its environment, reason about tasks, act through tools, remember across sessions, and learn from interactions.

### The Three Pillars of Agentic AI

| Pillar | Description | Source |
|--------|-------------|--------|
| **Memory Engineering** | Designing how agents store, organize, and retrieve information | Oracle AI Hub [1] |
| **Context Engineering** | Optimizing what goes into the LLM context window | Anthropic/Oracle [1] |
| **Prompt Engineering** | Crafting instructions that guide agent behavior | All frameworks |

### Core Agent Loop (OpenAI Agents SDK)

1. Call LLM with model settings and message history
2. LLM returns response (may include tool calls)
3. If final output → return and end
4. If handoff → switch to new agent, repeat
5. If tool calls → execute tools, append results, repeat

> **Source:** OpenAI Agents SDK [2]

---

## 2. Memory Engineering

### Definition

> **Agent Memory** is the exocortex that augments an LLM—capturing, encoding, storing, linking, and retrieving information beyond the model's parametric and contextual limits.

> **Memory Engineering** is the scaffolding and control harness that we design to move information optimally and efficiently into, through, and across all components of an AI system.

> **Source:** Oracle AI Developer Hub [1]

### The 6 Types of Agent Memory

| Memory Type | Human Analogy | Purpose | Storage |
|-------------|---------------|---------|---------|
| **Conversational** | Short-term memory | Chat history per thread | SQL Table |
| **Knowledge Base** | Long-term semantic memory | Facts, documents, search results | Vector Store |
| **Workflow** | Procedural memory | Learned action patterns | Vector Store |
| **Toolbox** | Skill memory | Dynamic tool definitions | Vector Store |
| **Entity** | Episodic memory | People, places, systems | Vector Store |
| **Summary** | Compressed memory | Condensed context | Vector Store |

> **Source:** Oracle AI Developer Hub [1]

### Multi-Level Memory (Mem0)

Mem0 provides three levels of memory persistence:

| Level | Description |
|-------|-------------|
| **User Memory** | Persistent preferences per user |
| **Session Memory** | Context within a conversation |
| **Agent Memory** | State for the agent itself |

**Key Benefits:**
- +26% accuracy over OpenAI Memory on LOCOMO benchmark
- 91% faster responses than full-context
- 90% fewer tokens than full-context

> **Source:** Mem0 [5]

### Programmatic vs Agentic Operations

| Operation | Programmatic | Agentic |
|-----------|:------------:|:-------:|
| Memory reads | ✅ | ❌ |
| Memory writes | ✅ | ❌ |
| Tool execution | ❌ | ✅ |
| Summary expansion | ❌ | ✅ |

**Rationale:**
- Memory reads are automatic because the agent can't know what it doesn't know
- Memory writes are automatic for reliability and completeness
- Tool calls require judgment about when and what to search

> **Source:** Oracle AI Developer Hub [1]

---

## 3. Context Engineering

### Definition

> **Context engineering** refers to the set of strategies for curating and maintaining the optimal set of tokens during LLM inference, including all the other information that may land there outside of the prompts.

> **Source:** Anthropic (cited in Oracle AI Hub [1])

### Key Techniques

#### 1. Context Window Monitoring
- Estimate tokens (~4 chars per token)
- Track usage percentage
- Trigger actions at thresholds (typically 80%)

#### 2. Auto-Summarization
- Compress long content when approaching limits
- Store full content with summary ID for later retrieval
- Replace inline content with references

#### 3. Just-In-Time (JIT) Retrieval
- Fetch information only when needed
- Store compact summaries
- Expand on demand via tool calls

### Context Management Flow

```
Context built → Check usage % → If >80%: Summarize & offload → Store with ID
                                         ↓
Agent sees: [Summary ID: abc123] Brief description 
            ← Agent can call expand_summary("abc123") if needed
```

> **Source:** Oracle AI Developer Hub [1]

---

## 4. Agent Architectures

### OpenAI Agents SDK Architecture

**Core Concepts:**
1. **Agents**: LLMs configured with instructions, tools, guardrails, and handoffs
2. **Handoffs**: Specialized tool calls for transferring control between agents
3. **Guardrails**: Configurable safety checks for input/output validation
4. **Sessions**: Automatic conversation history management
5. **Tracing**: Built-in tracking of agent runs

> **Source:** OpenAI Agents SDK [2]

### Microsoft AutoGen Layered Architecture

| Layer | Purpose |
|-------|---------|
| **Core API** | Message passing, event-driven agents, distributed runtime |
| **AgentChat API** | Simpler API for rapid prototyping, common patterns |
| **Extensions API** | First/third-party extensions for LLM clients, code execution |

**Key Features:**
- Cross-language support (.NET and Python)
- Local and distributed runtime
- Built-in Studio for no-code GUI

> **Source:** Microsoft AutoGen [3]

### CrewAI Dual Architecture

**1. Crews** - Autonomous agent teams:
- Natural, autonomous decision-making
- Dynamic task delegation
- Specialized roles with goals
- Flexible problem-solving

**2. Flows** - Event-driven workflows:
- Fine-grained control over execution
- Secure state management
- Clean integration with production code
- Conditional branching

> **Source:** CrewAI [4]

---

## 5. Multi-Agent Patterns

### Pattern 1: Handoff-Based Routing (OpenAI)

Agents can transfer control to other agents based on context:
- Language detection → route to language-specific agent
- Domain detection → route to specialist agent
- Escalation → route to human or advanced agent

> **Source:** OpenAI Agents SDK [2]

### Pattern 2: Role-Based Crews (CrewAI)

Define agents by role, goal, and backstory:
- **Researcher**: Uncover developments in topic
- **Analyst**: Create detailed reports from findings
- **Specialist**: Domain-specific expertise

Configure with YAML for clean separation:
- `agents.yaml`: Agent definitions
- `tasks.yaml`: Task definitions
- `crew.py`: Logic and tools

> **Source:** CrewAI [4]

### Pattern 3: AgentTool Orchestration (AutoGen)

Use agents as tools for other agents:
- Math expert agent as tool
- Chemistry expert agent as tool
- Orchestrator agent decides which to call

> **Source:** Microsoft AutoGen [3]

### Pattern 4: Hierarchical Process

- Automatically assign manager to coordinate
- Planning and execution through delegation
- Validation of results before proceeding

> **Source:** CrewAI [4]

### Orchestration Workflow Example

```
Orchestrator Agent
    ├── translate_to_research_papers (Agent as Tool)
    │       └── get_research_papers (Function Tool)
    └── translate_to_research_conversations (Agent as Tool)
            └── get_past_research_conversations (Function Tool)
                    ↓
            Synthesizer Agent
                    ↓
            Final Output
```

> **Source:** Oracle AI Developer Hub [1]

---

## 6. RAG Best Practices

### LlamaIndex Data Framework

**Core Capabilities:**
- **Data connectors**: Ingest APIs, PDFs, docs, SQL
- **Data structuring**: Indices, graphs for LLM use
- **Retrieval interface**: Knowledge-augmented output

> **Source:** LlamaIndex [6]

### Three Retrieval Mechanisms

| Method | Description | Best For |
|--------|-------------|----------|
| **Keyword Search** | Full-text using text indices | Exact term matching |
| **Vector Search** | Semantic similarity via embeddings | Meaning-based queries |
| **Hybrid Search** | Combines both approaches | Production systems |

### Hybrid Retrieval Strategies

**Pre-filtering**: Keyword first, then vector ranking
- Fast, keyword-strict
- Best for narrow searches

**Post-filtering**: Vector first, then keyword filter
- Semantically rich
- Best for exploratory queries

**Reciprocal Rank Fusion (RRF)**:
```
RRF_score = 1/(k + rank_vector) + 1/(k + rank_text)
```
- Most robust for production
- Combines strengths of both

> **Source:** Oracle AI Developer Hub [1]

### Embedding Best Practice

Use instruction prefixes for models like Nomic:
- **Documents**: `"search_document: <text>"`
- **Queries**: `"search_query: <text>"`

This aligns embedding spaces for better retrieval accuracy.

> **Source:** Oracle AI Developer Hub [1]

---

## 7. Tool Design Patterns

### Function Tool Pattern (OpenAI)

Define tools with the `@function_tool` decorator:
- Clear docstrings for LLM understanding
- Type hints for parameter validation
- Return structured responses

> **Source:** OpenAI Agents SDK [2]

### Semantic Tool Retrieval

**Problem**: Too many tools causes context bloat and poor selection.

**Solution**: Store tools in vector DB, retrieve by query similarity.

**Implementation:**
1. Store tool name + description + signature as embedding
2. Augment docstrings with LLM for better descriptions
3. Generate synthetic queries that would use each tool
4. At runtime, retrieve only relevant tools (3-5)

> **Source:** Oracle AI Developer Hub [1]

### Tool Augmentation

Use LLM to improve tool discoverability:
1. **Docstring augmentation**: Enhance descriptions
2. **Synthetic query generation**: Create example queries
3. **Rich embedding**: Combine all for better retrieval

> **Source:** Oracle AI Developer Hub [1]

---

## 8. Session & State Management

### OpenAI Sessions

**Built-in options:**
- `SQLiteSession`: File-based or in-memory database
- `RedisSession`: Scalable, distributed deployments
- Custom implementations via `Session` protocol

**Key Methods:**
- `get_items()`: Retrieve conversation history
- `add_items()`: Store new items
- `pop_item()`: Remove and return recent items
- `clear_session()`: Clear all items

> **Source:** OpenAI Agents SDK [2]

### CrewAI State Management

**Flows provide:**
- Structured state with Pydantic models
- Event-driven state transitions
- Conditional routing based on state

**State Flow:**
```
fetch_data() → state.sentiment = "analyzing"
    ↓
analyze_with_crew() → state.confidence = 0.85
    ↓
router determines next step based on confidence
```

> **Source:** CrewAI [4]

### Oracle Session Pattern

**Design Choice**: Mark vs Delete

| Approach | Pros | Cons |
|----------|------|------|
| Delete summarized | Simple, immediate savings | Permanent data loss |
| Mark as summarized | Preserves history, reversible | More complex queries |

**Recommendation**: Mark instead of delete for auditing and reversibility.

> **Source:** Oracle AI Developer Hub [1]

---

## 9. Orchestration Patterns

### Sequential Process

Tasks executed in order:
```
Task 1 → Task 2 → Task 3 → Output
```
- Simplest pattern
- Each task depends on previous output

> **Source:** CrewAI [4]

### Hierarchical Process

Manager coordinates agents:
```
Manager Agent
    ├── Delegate to Agent A
    ├── Validate Result A
    ├── Delegate to Agent B
    ├── Validate Result B
    └── Synthesize Final Output
```

> **Source:** CrewAI [4]

### Event-Driven Flow (CrewAI)

Use decorators for flow control:
- `@start()`: Entry point
- `@listen(step)`: Triggered after step completes
- `@router(step)`: Conditional branching
- `or_()`, `and_()`: Combine conditions

> **Source:** CrewAI [4]

### Human-in-the-Loop

**Temporal Integration (AutoGen)**:
- Durable, long-running workflows
- Pause for human input
- Resume after approval

> **Source:** Microsoft AutoGen [3]

---

## 10. Production Considerations

### Guardrails (OpenAI)

- Input validation before processing
- Output validation before returning
- Configurable safety checks

> **Source:** OpenAI Agents SDK [2]

### Tracing & Observability

**OpenAI Agents SDK supports:**
- Custom spans
- External destinations: Logfire, AgentOps, Braintrust, Scorecard, Keywords AI

**CrewAI AOP provides:**
- Real-time monitoring
- Metrics, logs, and traces
- Unified control plane

> **Sources:** OpenAI Agents SDK [2], CrewAI [4]

### Scalability

**CrewAI**: 5.76x faster than LangGraph in certain tasks

**Mem0**: 
- 91% faster than full-context
- 90% fewer tokens

> **Sources:** CrewAI [4], Mem0 [5]

### Framework Independence

**CrewAI advantage**: Built from scratch, independent of LangChain
- Faster execution
- Lighter resource demands
- More customization control

> **Source:** CrewAI [4]

---

## 11. References

### Primary Sources

| # | Repository | Description | URL |
|---|------------|-------------|-----|
| [1] | **Oracle AI Developer Hub** | Memory & context engineering, 6 memory types, RAG patterns | https://github.com/oracle-devrel/oracle-ai-developer-hub |
| [2] | **OpenAI Agents SDK** | Official OpenAI agent framework, handoffs, sessions, guardrails | https://github.com/openai/openai-agents-python |
| [3] | **Microsoft AutoGen** | Multi-agent framework, layered architecture, cross-language | https://github.com/microsoft/autogen |
| [4] | **CrewAI** | Role-based agents, crews, flows, event-driven orchestration | https://github.com/crewAIInc/crewAI |
| [5] | **Mem0** | Dedicated memory layer, multi-level memory, personalization | https://github.com/mem0ai/mem0 |
| [6] | **LlamaIndex** | Data framework for RAG, data connectors, indices | https://github.com/run-llama/llama_index |

### Key Notebooks & Documentation

| Resource | Source | Link |
|----------|--------|------|
| Memory & Context Engineering Notebook | Oracle | [View](https://github.com/oracle-devrel/oracle-ai-developer-hub/blob/main/notebooks/memory_context_engineering_agents.ipynb) |
| RAG Agents Zero to Hero | Oracle | [View](https://github.com/oracle-devrel/oracle-ai-developer-hub/blob/main/notebooks/oracle_rag_agents_zero_to_hero.ipynb) |
| OpenAI Agents Examples | OpenAI | [View](https://github.com/openai/openai-agents-python/tree/main/examples) |
| CrewAI Examples | CrewAI | [View](https://github.com/crewAIInc/crewAI-examples) |
| AutoGen Documentation | Microsoft | [View](https://microsoft.github.io/autogen/) |
| LlamaIndex Documentation | LlamaIndex | [View](https://docs.llamaindex.ai/) |
| Mem0 Documentation | Mem0 | [View](https://docs.mem0.ai/) |

### Research Papers

| Paper | Citation |
|-------|----------|
| Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory | arXiv:2504.19413 (2025) |

### Learning Resources

| Course | Provider | Link |
|--------|----------|------|
| Multi AI Agent Systems with CrewAI | DeepLearning.AI | [View](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) |
| Practical Multi AI Agents | DeepLearning.AI | [View](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/) |

---

## Quick Reference Summary

### Memory Types (6)
1. Conversational (SQL) → Chat history
2. Knowledge Base (Vector) → Documents/facts
3. Workflow (Vector) → Action patterns
4. Toolbox (Vector) → Tool definitions
5. Entity (Vector) → People/places/systems
6. Summary (Vector) → Compressed context

### Key Patterns
- **Programmatic reads, agentic actions**
- **Hybrid retrieval** (keyword + vector + RRF)
- **Semantic tool retrieval** (vector DB for tools)
- **JIT memory expansion** (summaries + on-demand)
- **Role-based agents** (crews with specialization)
- **Event-driven flows** (decorators for control)

### Production Checklist
- [ ] Multiple memory types implemented
- [ ] Context window monitoring at 80%
- [ ] Guardrails for input/output validation
- [ ] Tracing and observability enabled
- [ ] Session persistence configured
- [ ] Hybrid retrieval for RAG
- [ ] Tool count limited (3-5 per call)

---

*Document compiled from public GitHub repositories. All code and concepts belong to their respective owners. Last updated: December 2025.*

