# 🔍 Learning AI Development with Octocode Research

This project demonstrates how to use **Octocode MCP** to research and learn AI development best practices directly from GitHub repositories. It starts with research on https://github.com/oracle-devrel/oracle-ai-developer-hub, creating agents (from research context), and then researching for more resources.

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

## 🚀 The Research Process (5 Steps)

### Step 1: Start with a Goal
```
/octocode/research

I want to learn AI best practices from this repo:
https://github.com/oracle-devrel/oracle-ai-developer-hub
```

### Step 2: Octocode Research Tools

Octocode Research...🔍🐙

### Step 3: Generate Research Output
Creates structured documentation:
- Best practices summary
- Code examples
- Learning resources
- Working implementations

### Step 4: Create Working Example
From the research context, I was able to create agents easily using simple prompts. I created a working example using Python:
- `multi_agent_session_example.py` - A fully functional 3-agent system implementation
- Demonstrates orchestrator, research agent, and analyst agent patterns
- Includes memory management and agent coordination

### Step 5: Expand Research
I asked Octocode to find more resources to build a more comprehensive context for agent generation (see the final generated file: [.octocode/research/agentic-ai-best-practices/research.md](.octocode/research/agentic-ai-best-practices/research.md)).

**Tip:** You can use this file to ask Octocode to go even deeper...

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

## ⚠️ Disclaimer

When using Octocode, it creates a `.octocode` folder to store research context and automatically adds it to your `.gitignore` file to keep your repository clean.

---

Created with [Octocode MCP](https://octocode.ai) 🔍🐙
