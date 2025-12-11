# 🔍 Learning AI Development with Octocode Research

This project demonstrates how to use **Octocode MCP** to research and learn AI development best practices directly from GitHub repositories.

---

## 🎥 Video Tutorial

**Full Research Walkthrough**: [Learning AI with Octocode research](https://www.youtube.com/watch?v=r-GpBDDnmyk)

Watch the complete research process in action, covering:
- How to use Octocode MCP for AI development research
- Researching the Oracle AI Developer Hub repository
- Extracting best practices and patterns
- Building working implementations from research

---

## 📚 What is Octocode Research?

Octocode is an MCP (Model Context Protocol) tool that lets you **research GitHub repositories** using natural language. Instead of manually browsing code, you ask questions and get structured answers with code references.

---

## 🚀 The Research Process (6 Steps)

### Step 1: Start with a Goal
```
/octocode/research

I want to learn AI best practices from this repo:
https://github.com/oracle-devrel/oracle-ai-developer-hub
```

### Step 2: Explore Repository Structure
Octocode uses `githubViewRepoStructure` to map the repository:
```
📂 oracle-ai-developer-hub/
├── 📂 apps/              → Full applications
├── 📂 notebooks/         → Jupyter tutorials
│   ├── memory_context_engineering_agents.ipynb
│   ├── oracle_rag_agents_zero_to_hero.ipynb
│   └── oracle_rag_with_evals.ipynb
└── README.md
```

### Step 3: Get File Contents
Octocode uses `githubGetFileContent` to read relevant files:
- Fetches notebook content
- Extracts code patterns
- Identifies best practices

### Step 4: Search for Patterns
Octocode uses `githubSearchCode` to find specific implementations:
- Memory management patterns
- Agent orchestration code
- RAG pipeline examples

### Step 5: Generate Research Output
Creates structured documentation:
- Best practices summary
- Code examples
- Learning resources
- Working implementations

### Step 6: Create Working Example
I also created a working example using Python from the research:
- `multi_agent_session_example.py` - A fully functional 3-agent system implementation
- Demonstrates orchestrator, research agent, and analyst agent patterns
- Includes memory management and agent coordination

---

## 📁 Project Outputs

| File | Description |
|------|-------------|
| `multi_agent_session_example.py` | Output from Octocode research - Working 3-agent system implementation |
| `AI_Agentic_Development_Best_Practices.md` | Output of full research - Comprehensive best practices document |
| `.octocode/research/agentic-ai-best-practices/research.md` | Octocode raw research output - Unprocessed research data |

---

## 🧠 Key Concepts Learned

### 6 Types of Agent Memory
| Memory Type | Purpose |
|-------------|---------|
| **Conversational** | Chat history per thread |
| **Knowledge Base** | Facts and documents |
| **Workflow** | Learned action patterns |
| **Toolbox** | Available tools |
| **Entity** | People, places, concepts |
| **Summary** | Compressed context |

### Agent Architecture Pattern
```
User Query → Orchestrator → [Research Agent, Analyst Agent] → Synthesizer → Response
```

---

## 🛠️ Octocode Tools Used

| Tool | Purpose |
|------|---------|
| `githubViewRepoStructure` | Explore repository layout |
| `githubGetFileContent` | Read specific files |
| `githubSearchCode` | Find code patterns |
| `githubSearchRepositories` | Discover related repos |
| `packageSearch` | Find NPM/Python packages |

---

## 📖 How to Reproduce This Research

1. **Install Octocode MCP** in your IDE
2. **Use the `/octocode/research` prompt** with your topic
3. **Ask specific questions** about the codebase
4. **Generate documentation** and working examples

---

## 🔗 Resources Found

- [Hugging Face Agents Course](https://github.com/huggingface/agents-course) (24k ⭐)
- [GenAI Agents Tutorials](https://github.com/NirDiamant/GenAI_Agents) (18k ⭐)
- [LangChain](https://github.com/langchain-ai/langchain) (121k ⭐)
- [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) (67k ⭐)

---

Created with [Octocode MCP](https://octocode.ai) 🔍🐙
