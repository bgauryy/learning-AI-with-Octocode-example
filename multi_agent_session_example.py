"""
Multi-Agent System with Shared Session Management
==================================================

This example demonstrates best practices for building a 3-agent system where
all agents communicate and share context through a centralized session/memory layer.

Architecture Overview:
┌─────────────────────────────────────────────────────────────────────┐
│                        Shared Memory Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Conversation │  │  Knowledge   │  │   Entity     │              │
│  │   Memory     │  │    Base      │  │   Memory     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ Read/Write
          ┌───────────────────┼───────────────────┐
          │                   │                   │
    ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐
    │ Research  │       │  Analyst  │       │ Response  │
    │   Agent   │◄─────►│   Agent   │◄─────►│   Agent   │
    └───────────┘       └───────────┘       └───────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  Orchestrator   │
                    │     Agent       │
                    └─────────────────┘

Key Best Practices Implemented:
1. Centralized Memory Layer - All agents read/write to shared memory
2. Session-based Context - Thread ID tracks conversation across agents
3. Programmatic Memory Operations - Memory reads happen automatically
4. Entity Extraction - Important entities are extracted and stored
5. Context Summarization - Long contexts are compressed automatically
6. Agent Orchestration - Orchestrator coordinates specialized agents
"""

import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio


# =============================================================================
# SECTION 1: MEMORY TYPES AND DATA MODELS
# =============================================================================

class MemoryType(Enum):
    """Types of memory in the system."""
    CONVERSATIONAL = "conversational"  # Chat history per thread
    KNOWLEDGE = "knowledge"            # Facts and documents
    ENTITY = "entity"                  # People, places, concepts
    WORKFLOW = "workflow"              # Learned action patterns
    SUMMARY = "summary"                # Compressed contexts


@dataclass
class MemoryEntry:
    """
    A single entry in memory.
    
    Attributes:
        id: Unique identifier for this memory
        thread_id: Session/conversation thread this belongs to
        memory_type: Type of memory (conversational, knowledge, etc.)
        content: The actual content/text
        metadata: Additional structured data
        timestamp: When this was created
        agent_source: Which agent created this memory
    """
    id: str
    thread_id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    agent_source: str = "system"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "agent_source": self.agent_source
        }


@dataclass 
class AgentMessage:
    """
    A message passed between agents.
    
    This structure allows agents to communicate findings, requests,
    and context to each other through the orchestrator.
    """
    from_agent: str
    to_agent: str
    content: str
    message_type: str = "info"  # info, request, response, handoff
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SECTION 2: SHARED MEMORY LAYER
# =============================================================================

class SharedMemoryLayer:
    """
    Centralized memory layer that all agents share.
    
    This is the KEY to multi-agent context sharing:
    - All agents read from the same memory
    - All agents write to the same memory
    - Memory is organized by thread_id for session isolation
    - Different memory types serve different purposes
    
    Best Practices:
    1. Memory reads are PROGRAMMATIC (automatic at each agent turn)
    2. Memory writes are PROGRAMMATIC (automatic after agent response)
    3. Entity extraction happens automatically
    4. Context summarization triggers at threshold
    """
    
    def __init__(self, context_threshold: float = 0.8):
        """
        Initialize the shared memory layer.
        
        Args:
            context_threshold: Percentage of context window that triggers summarization
        """
        # In production, these would be database tables/vector stores
        self._conversational: List[MemoryEntry] = []
        self._knowledge: List[MemoryEntry] = []
        self._entities: List[MemoryEntry] = []
        self._workflows: List[MemoryEntry] = []
        self._summaries: List[MemoryEntry] = []
        
        self.context_threshold = context_threshold
        self.max_context_tokens = 128000  # GPT-4 context window
        
    # -------------------------------------------------------------------------
    # CONVERSATIONAL MEMORY: Chat history per thread
    # -------------------------------------------------------------------------
    
    def write_conversation(
        self, 
        thread_id: str, 
        role: str, 
        content: str, 
        agent_source: str = "system"
    ) -> str:
        """
        Store a message in conversation history.
        
        Args:
            thread_id: The session/thread identifier
            role: 'user', 'assistant', or agent name
            content: The message content
            agent_source: Which agent created this
            
        Returns:
            The ID of the created memory entry
        """
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            memory_type=MemoryType.CONVERSATIONAL,
            content=content,
            metadata={"role": role},
            agent_source=agent_source
        )
        self._conversational.append(entry)
        return entry.id
    
    def read_conversation(
        self, 
        thread_id: str, 
        limit: int = 20,
        exclude_summarized: bool = True
    ) -> List[MemoryEntry]:
        """
        Read conversation history for a thread.
        
        This is called AUTOMATICALLY at the start of each agent turn
        to provide context. The agent doesn't decide whether to "remember" -
        it always gets the relevant history.
        
        Args:
            thread_id: The session/thread identifier
            limit: Maximum messages to return
            exclude_summarized: Skip messages that have been summarized
            
        Returns:
            List of conversation entries, oldest first
        """
        entries = [
            e for e in self._conversational 
            if e.thread_id == thread_id
            and (not exclude_summarized or not e.metadata.get("summarized"))
        ]
        # Sort by timestamp, return most recent
        entries.sort(key=lambda x: x.timestamp)
        return entries[-limit:]
    
    # -------------------------------------------------------------------------
    # KNOWLEDGE BASE MEMORY: Facts and documents
    # -------------------------------------------------------------------------
    
    def write_knowledge(
        self, 
        thread_id: str, 
        content: str, 
        source: str,
        agent_source: str = "system"
    ) -> str:
        """
        Store knowledge/facts discovered during the session.
        
        This is used when:
        - Research agent finds relevant documents
        - External API returns data
        - User provides reference material
        
        Args:
            thread_id: The session identifier
            content: The knowledge content
            source: Where this knowledge came from
            agent_source: Which agent added this
        """
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            memory_type=MemoryType.KNOWLEDGE,
            content=content,
            metadata={"source": source},
            agent_source=agent_source
        )
        self._knowledge.append(entry)
        return entry.id
    
    def read_knowledge(
        self, 
        thread_id: str, 
        query: str = None, 
        k: int = 5
    ) -> List[MemoryEntry]:
        """
        Search knowledge base for relevant content.
        
        In production, this would use vector similarity search.
        Here we do simple keyword matching for demonstration.
        
        Args:
            thread_id: The session identifier
            query: Search query (optional)
            k: Number of results to return
        """
        entries = [e for e in self._knowledge if e.thread_id == thread_id]
        
        if query:
            # Simple relevance scoring (in production: vector similarity)
            query_lower = query.lower()
            scored = [
                (e, sum(1 for word in query_lower.split() if word in e.content.lower()))
                for e in entries
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [e for e, score in scored[:k] if score > 0]
        
        return entries[:k]
    
    # -------------------------------------------------------------------------
    # ENTITY MEMORY: People, places, concepts mentioned
    # -------------------------------------------------------------------------
    
    def write_entity(
        self, 
        thread_id: str, 
        name: str, 
        entity_type: str, 
        description: str,
        agent_source: str = "system"
    ) -> str:
        """
        Store an extracted entity.
        
        Entities are automatically extracted from conversations and
        stored for quick reference. This helps agents maintain context
        about who/what is being discussed.
        
        Args:
            thread_id: The session identifier
            name: Entity name (e.g., "John Smith")
            entity_type: Type (PERSON, ORGANIZATION, CONCEPT, etc.)
            description: Brief description or context
        """
        # Check if entity already exists for this thread
        existing = [
            e for e in self._entities 
            if e.thread_id == thread_id 
            and e.metadata.get("name") == name
        ]
        
        if existing:
            # Update existing entity
            existing[0].content = description
            existing[0].timestamp = datetime.now()
            return existing[0].id
        
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            memory_type=MemoryType.ENTITY,
            content=f"{name} ({entity_type}): {description}",
            metadata={"name": name, "type": entity_type, "description": description},
            agent_source=agent_source
        )
        self._entities.append(entry)
        return entry.id
    
    def read_entities(self, thread_id: str, query: str = None) -> List[MemoryEntry]:
        """
        Read entities relevant to the current context.
        
        This helps agents answer questions like "Who were we talking about?"
        without needing to scan the entire conversation.
        """
        entries = [e for e in self._entities if e.thread_id == thread_id]
        
        if query:
            query_lower = query.lower()
            entries = [
                e for e in entries 
                if query_lower in e.metadata.get("name", "").lower()
                or query_lower in e.content.lower()
            ]
        
        return entries
    
    # -------------------------------------------------------------------------
    # SUMMARY MEMORY: Compressed context
    # -------------------------------------------------------------------------
    
    def write_summary(
        self, 
        thread_id: str, 
        summary: str, 
        full_content: str,
        description: str
    ) -> str:
        """
        Store a compressed summary of conversation/context.
        
        When context gets too long, we summarize older parts and store
        both the summary and original. The agent sees the compact summary
        but can request the full content if needed.
        """
        entry = MemoryEntry(
            id=str(uuid.uuid4())[:8],  # Short ID for easy reference
            thread_id=thread_id,
            memory_type=MemoryType.SUMMARY,
            content=summary,
            metadata={
                "full_content": full_content,
                "description": description
            }
        )
        self._summaries.append(entry)
        return entry.id
    
    def read_summaries(self, thread_id: str) -> List[MemoryEntry]:
        """Get available summaries for a thread."""
        return [e for e in self._summaries if e.thread_id == thread_id]
    
    def expand_summary(self, summary_id: str) -> Optional[str]:
        """
        Get full content for a summary (just-in-time retrieval).
        
        This is an AGENTIC operation - the agent decides when it needs
        more detail and requests expansion.
        """
        for entry in self._summaries:
            if entry.id == summary_id:
                return entry.metadata.get("full_content", entry.content)
        return None
    
    # -------------------------------------------------------------------------
    # CONTEXT BUILDING: Assembles context for agents
    # -------------------------------------------------------------------------
    
    def build_agent_context(
        self, 
        thread_id: str, 
        current_query: str,
        agent_name: str
    ) -> Dict[str, Any]:
        """
        Build complete context for an agent.
        
        This is the PROGRAMMATIC part - called automatically before
        each agent processes a request. The agent doesn't choose what
        to remember; it receives all relevant context.
        
        Args:
            thread_id: The session identifier
            current_query: The current user query
            agent_name: Which agent is receiving context
            
        Returns:
            Dictionary with all context sections
        """
        context = {
            "thread_id": thread_id,
            "current_query": current_query,
            "agent": agent_name,
            "conversation_history": [],
            "relevant_knowledge": [],
            "known_entities": [],
            "available_summaries": [],
            "context_usage": {}
        }
        
        # 1. Get conversation history
        conversations = self.read_conversation(thread_id)
        context["conversation_history"] = [
            {"role": e.metadata.get("role"), "content": e.content, "agent": e.agent_source}
            for e in conversations
        ]
        
        # 2. Get relevant knowledge
        knowledge = self.read_knowledge(thread_id, current_query)
        context["relevant_knowledge"] = [
            {"content": e.content, "source": e.metadata.get("source")}
            for e in knowledge
        ]
        
        # 3. Get known entities
        entities = self.read_entities(thread_id, current_query)
        context["known_entities"] = [
            {"name": e.metadata.get("name"), "type": e.metadata.get("type"), 
             "description": e.metadata.get("description")}
            for e in entities
        ]
        
        # 4. Get available summaries
        summaries = self.read_summaries(thread_id)
        context["available_summaries"] = [
            {"id": e.id, "description": e.metadata.get("description")}
            for e in summaries
        ]
        
        # 5. Calculate context usage
        total_chars = sum(len(str(v)) for v in context.values())
        estimated_tokens = total_chars // 4
        context["context_usage"] = {
            "estimated_tokens": estimated_tokens,
            "max_tokens": self.max_context_tokens,
            "usage_percent": round((estimated_tokens / self.max_context_tokens) * 100, 1)
        }
        
        return context
    
    def format_context_for_prompt(self, context: Dict[str, Any]) -> str:
        """
        Format context dict into a prompt-friendly string.
        
        This creates a structured context that the LLM can easily parse.
        """
        sections = []
        
        # Conversation history
        if context["conversation_history"]:
            history = "\n".join([
                f"[{msg['role']}] ({msg['agent']}): {msg['content'][:200]}..."
                if len(msg['content']) > 200 else f"[{msg['role']}] ({msg['agent']}): {msg['content']}"
                for msg in context["conversation_history"][-10:]  # Last 10 messages
            ])
            sections.append(f"## Conversation History\n{history}")
        
        # Relevant knowledge
        if context["relevant_knowledge"]:
            knowledge = "\n".join([
                f"- [{k['source']}]: {k['content'][:150]}..."
                for k in context["relevant_knowledge"][:3]
            ])
            sections.append(f"## Relevant Knowledge\n{knowledge}")
        
        # Known entities
        if context["known_entities"]:
            entities = "\n".join([
                f"- {e['name']} ({e['type']}): {e['description']}"
                for e in context["known_entities"][:5]
            ])
            sections.append(f"## Known Entities\n{entities}")
        
        # Available summaries
        if context["available_summaries"]:
            summaries = "\n".join([
                f"- [Summary ID: {s['id']}] {s['description']}"
                for s in context["available_summaries"]
            ])
            sections.append(f"## Available Summaries (use expand_summary to get details)\n{summaries}")
        
        # Context usage warning
        if context["context_usage"]["usage_percent"] > 70:
            sections.append(
                f"⚠️ Context Usage: {context['context_usage']['usage_percent']}% "
                f"({context['context_usage']['estimated_tokens']}/{context['context_usage']['max_tokens']} tokens)"
            )
        
        return "\n\n".join(sections)


# =============================================================================
# SECTION 3: BASE AGENT CLASS
# =============================================================================

class BaseAgent:
    """
    Base class for all agents in the system.
    
    Each agent:
    1. Has a specific role and capabilities
    2. Reads from shared memory automatically
    3. Writes to shared memory automatically
    4. Can communicate with other agents via the orchestrator
    
    Best Practice: Agents are STATELESS - all state is in SharedMemoryLayer
    """
    
    def __init__(
        self, 
        name: str, 
        role: str, 
        memory: SharedMemoryLayer,
        instructions: str
    ):
        """
        Initialize an agent.
        
        Args:
            name: Unique agent identifier
            role: Agent's role description
            memory: Shared memory layer instance
            instructions: System instructions for this agent
        """
        self.name = name
        self.role = role
        self.memory = memory
        self.instructions = instructions
    
    async def process(
        self, 
        thread_id: str, 
        query: str, 
        inter_agent_context: List[AgentMessage] = None
    ) -> tuple[str, List[AgentMessage]]:
        """
        Process a query and return response.
        
        This is the main agent loop:
        1. AUTOMATICALLY read context from shared memory
        2. Process query with context
        3. Generate response
        4. AUTOMATICALLY write to shared memory
        5. Return response and any messages for other agents
        
        Args:
            thread_id: Session identifier
            query: User query or orchestrator request
            inter_agent_context: Messages from other agents
            
        Returns:
            Tuple of (response_text, messages_for_other_agents)
        """
        # STEP 1: Build context (PROGRAMMATIC - always happens)
        context = self.memory.build_agent_context(thread_id, query, self.name)
        formatted_context = self.memory.format_context_for_prompt(context)
        
        # STEP 2: Add inter-agent messages to context
        if inter_agent_context:
            agent_messages = "\n".join([
                f"[From {msg.from_agent}]: {msg.content}"
                for msg in inter_agent_context
                if msg.to_agent == self.name or msg.to_agent == "all"
            ])
            if agent_messages:
                formatted_context += f"\n\n## Messages from Other Agents\n{agent_messages}"
        
        # STEP 3: Process (this would call the LLM in production)
        response, outgoing_messages = await self._generate_response(
            query, formatted_context, context
        )
        
        # STEP 4: Write response to memory (PROGRAMMATIC - always happens)
        self.memory.write_conversation(
            thread_id=thread_id,
            role="assistant",
            content=response,
            agent_source=self.name
        )
        
        # STEP 5: Extract and store entities (PROGRAMMATIC)
        await self._extract_and_store_entities(thread_id, response)
        
        return response, outgoing_messages
    
    async def _generate_response(
        self, 
        query: str, 
        context: str,
        raw_context: Dict
    ) -> tuple[str, List[AgentMessage]]:
        """
        Generate response using LLM.
        
        Override this in subclasses for specific agent behavior.
        In production, this calls the actual LLM API.
        """
        # Placeholder - subclasses implement specific logic
        raise NotImplementedError("Subclasses must implement _generate_response")
    
    async def _extract_and_store_entities(self, thread_id: str, text: str):
        """
        Extract entities from text and store in memory.
        
        This is a simplified version - in production, use an LLM
        for entity extraction.
        """
        # Simple pattern matching for demonstration
        # In production: Use LLM with structured output
        entity_patterns = [
            ("PERSON", ["user", "customer", "client"]),
            ("TOPIC", ["project", "system", "feature"]),
            ("ORGANIZATION", ["company", "team", "department"])
        ]
        
        text_lower = text.lower()
        for entity_type, keywords in entity_patterns:
            for keyword in keywords:
                if keyword in text_lower:
                    self.memory.write_entity(
                        thread_id=thread_id,
                        name=keyword.title(),
                        entity_type=entity_type,
                        description=f"Mentioned in context by {self.name}",
                        agent_source=self.name
                    )


# =============================================================================
# SECTION 4: SPECIALIZED AGENTS
# =============================================================================

class ResearchAgent(BaseAgent):
    """
    Agent specialized in finding and gathering information.
    
    Responsibilities:
    - Search knowledge bases
    - Find relevant documents
    - Gather facts and data
    """
    
    def __init__(self, memory: SharedMemoryLayer):
        super().__init__(
            name="ResearchAgent",
            role="Information Gatherer",
            memory=memory,
            instructions="""You are a Research Agent specialized in finding information.
            
Your responsibilities:
1. Search available knowledge bases for relevant information
2. Identify what additional information is needed
3. Summarize findings for other agents
4. Flag when you cannot find requested information

Always cite your sources when providing information."""
        )
    
    async def _generate_response(
        self, 
        query: str, 
        context: str,
        raw_context: Dict
    ) -> tuple[str, List[AgentMessage]]:
        """
        Research agent processing logic.
        
        In production, this would:
        1. Call LLM with research-focused prompt
        2. Potentially call external search APIs
        3. Store findings in knowledge base
        """
        # Simulated research response
        response = f"""## Research Findings for: {query}

Based on the available context and knowledge base:

**Relevant Information Found:**
- Analyzed {len(raw_context.get('relevant_knowledge', []))} knowledge entries
- Identified {len(raw_context.get('known_entities', []))} relevant entities

**Key Points:**
1. The query relates to topics discussed in the conversation
2. Additional context may be available in the summaries
3. I recommend the Analyst Agent review these findings

**Sources:** Internal knowledge base, conversation history"""

        # Create message for Analyst Agent
        outgoing_messages = [
            AgentMessage(
                from_agent=self.name,
                to_agent="AnalystAgent",
                content=f"Research completed for query: {query}. Found {len(raw_context.get('relevant_knowledge', []))} relevant items.",
                message_type="info"
            )
        ]
        
        # Store findings in knowledge base
        self.memory.write_knowledge(
            thread_id=raw_context["thread_id"],
            content=response,
            source="ResearchAgent analysis",
            agent_source=self.name
        )
        
        return response, outgoing_messages


class AnalystAgent(BaseAgent):
    """
    Agent specialized in analysis and synthesis.
    
    Responsibilities:
    - Analyze information from Research Agent
    - Identify patterns and insights
    - Draw conclusions
    - Prepare structured analysis
    """
    
    def __init__(self, memory: SharedMemoryLayer):
        super().__init__(
            name="AnalystAgent",
            role="Information Analyst",
            memory=memory,
            instructions="""You are an Analyst Agent specialized in synthesizing information.
            
Your responsibilities:
1. Analyze information gathered by the Research Agent
2. Identify patterns, trends, and insights
3. Draw logical conclusions from available data
4. Prepare clear, structured analysis reports

Always support your conclusions with evidence from the context."""
        )
    
    async def _generate_response(
        self, 
        query: str, 
        context: str,
        raw_context: Dict
    ) -> tuple[str, List[AgentMessage]]:
        """
        Analyst agent processing logic.
        """
        # Check for messages from Research Agent
        research_findings = [
            k for k in raw_context.get('relevant_knowledge', [])
            if 'ResearchAgent' in k.get('source', '')
        ]
        
        response = f"""## Analysis Report

**Query Under Analysis:** {query}

**Information Sources:**
- Conversation context: {len(raw_context.get('conversation_history', []))} messages
- Knowledge items: {len(raw_context.get('relevant_knowledge', []))} entries
- Research findings: {len(research_findings)} items

**Analysis:**
Based on the available information, here are the key insights:

1. **Context Understanding:** The query appears in the context of ongoing discussion
2. **Pattern Identification:** Multiple related entities have been mentioned
3. **Synthesis:** The information converges on several key themes

**Confidence Level:** Medium (based on available data)

**Recommendations:**
- The Response Agent should use these findings to formulate a comprehensive answer
- Additional research may be beneficial for completeness"""

        # Message for Response Agent
        outgoing_messages = [
            AgentMessage(
                from_agent=self.name,
                to_agent="ResponseAgent",
                content=f"Analysis complete. Confidence: Medium. Key themes identified from {len(research_findings)} research items.",
                message_type="info"
            )
        ]
        
        return response, outgoing_messages


class ResponseAgent(BaseAgent):
    """
    Agent specialized in formulating final responses.
    
    Responsibilities:
    - Synthesize research and analysis into coherent response
    - Ensure response addresses user's original query
    - Format response appropriately
    - Handle follow-up questions
    """
    
    def __init__(self, memory: SharedMemoryLayer):
        super().__init__(
            name="ResponseAgent",
            role="Response Synthesizer",
            memory=memory,
            instructions="""You are a Response Agent specialized in creating final responses.
            
Your responsibilities:
1. Synthesize information from Research and Analyst agents
2. Create clear, helpful responses to user queries
3. Ensure responses are complete and accurate
4. Handle follow-up questions with full context awareness

Always maintain a helpful, professional tone."""
        )
    
    async def _generate_response(
        self, 
        query: str, 
        context: str,
        raw_context: Dict
    ) -> tuple[str, List[AgentMessage]]:
        """
        Response agent processing logic.
        """
        # Build final response using all context
        entity_summary = ", ".join([
            e['name'] for e in raw_context.get('known_entities', [])[:3]
        ]) or "No specific entities"
        
        response = f"""Thank you for your question: "{query}"

Based on our team's analysis:

**Summary:**
Our Research Agent gathered relevant information and our Analyst Agent identified key patterns. Here's what we found:

**Key Points:**
1. Your query has been analyzed in the context of our ongoing conversation
2. Relevant entities in this discussion: {entity_summary}
3. We've considered {len(raw_context.get('relevant_knowledge', []))} knowledge sources

**Recommendation:**
Based on the analysis, we recommend [specific action based on context].

**Follow-up:**
Feel free to ask any clarifying questions. I have access to our full conversation history and can expand on any point.

---
*Response synthesized from multi-agent analysis*"""

        # No outgoing messages - final response
        return response, []


# =============================================================================
# SECTION 5: ORCHESTRATOR
# =============================================================================

class Orchestrator:
    """
    Coordinates the multi-agent system.
    
    The Orchestrator:
    1. Receives user queries
    2. Decides which agents to invoke
    3. Manages agent-to-agent communication
    4. Ensures shared memory is properly used
    5. Returns final response to user
    
    Best Practice: The orchestrator doesn't process content itself -
    it delegates to specialized agents and coordinates their work.
    """
    
    def __init__(self, memory: SharedMemoryLayer):
        """
        Initialize orchestrator with all agents.
        """
        self.memory = memory
        
        # Initialize specialized agents with SHARED memory
        self.research_agent = ResearchAgent(memory)
        self.analyst_agent = AnalystAgent(memory)
        self.response_agent = ResponseAgent(memory)
        
        # Agent execution order (can be dynamic based on query)
        self.default_pipeline = [
            self.research_agent,
            self.analyst_agent,
            self.response_agent
        ]
    
    async def process_query(
        self, 
        thread_id: str, 
        user_query: str
    ) -> str:
        """
        Process a user query through the agent pipeline.
        
        Flow:
        1. Store user query in shared memory
        2. Run Research Agent → stores findings in memory
        3. Run Analyst Agent → reads research, adds analysis
        4. Run Response Agent → reads all, produces final response
        5. Return final response
        
        All agents share the same memory, so each can see what
        the others have done.
        """
        print(f"\n{'='*60}")
        print(f"🎯 Processing Query: {user_query[:50]}...")
        print(f"📍 Thread ID: {thread_id}")
        print(f"{'='*60}\n")
        
        # Step 1: Store user query (PROGRAMMATIC)
        self.memory.write_conversation(
            thread_id=thread_id,
            role="user",
            content=user_query,
            agent_source="user"
        )
        
        # Step 2: Run agent pipeline
        inter_agent_messages: List[AgentMessage] = []
        final_response = ""
        
        for agent in self.default_pipeline:
            print(f"\n🤖 Running {agent.name}...")
            
            # Each agent gets messages from previous agents
            response, new_messages = await agent.process(
                thread_id=thread_id,
                query=user_query,
                inter_agent_context=inter_agent_messages
            )
            
            # Collect messages for next agents
            inter_agent_messages.extend(new_messages)
            
            # Last agent's response is the final response
            final_response = response
            
            print(f"   ✅ {agent.name} complete")
            
            # Show inter-agent messages
            for msg in new_messages:
                print(f"   📨 Message to {msg.to_agent}: {msg.content[:50]}...")
        
        # Step 3: Check if context summarization needed
        context = self.memory.build_agent_context(thread_id, user_query, "Orchestrator")
        if context["context_usage"]["usage_percent"] > 80:
            print("\n⚠️ Context usage high - triggering summarization...")
            await self._summarize_old_context(thread_id)
        
        print(f"\n{'='*60}")
        print("✅ Query processing complete")
        print(f"{'='*60}\n")
        
        return final_response
    
    async def _summarize_old_context(self, thread_id: str):
        """
        Summarize old conversation to free up context space.
        
        Best Practice: Mark summarized messages, don't delete them.
        The original is preserved in the summary entry.
        """
        conversations = self.memory.read_conversation(thread_id, limit=50)
        
        if len(conversations) > 10:
            # Take older messages for summarization
            to_summarize = conversations[:-10]
            
            # Create summary (in production: use LLM)
            full_content = "\n".join([e.content for e in to_summarize])
            summary = f"Summary of {len(to_summarize)} earlier messages: Discussion covered various topics."
            
            # Store summary
            summary_id = self.memory.write_summary(
                thread_id=thread_id,
                summary=summary,
                full_content=full_content,
                description=f"Conversation summary ({len(to_summarize)} messages)"
            )
            
            # Mark original messages as summarized
            for entry in to_summarize:
                entry.metadata["summarized"] = True
                entry.metadata["summary_id"] = summary_id
            
            print(f"   📦 Created summary {summary_id} for {len(to_summarize)} messages")


# =============================================================================
# SECTION 6: USAGE EXAMPLE
# =============================================================================

async def main():
    """
    Demonstrate the multi-agent system with shared session management.
    """
    print("\n" + "="*70)
    print("🚀 MULTI-AGENT SYSTEM WITH SHARED SESSION MANAGEMENT")
    print("="*70 + "\n")
    
    # Create shared memory layer (single instance for all agents)
    shared_memory = SharedMemoryLayer(context_threshold=0.8)
    
    # Create orchestrator (initializes all agents with shared memory)
    orchestrator = Orchestrator(shared_memory)
    
    # Create a session
    session_id = str(uuid.uuid4())[:8]
    print(f"📝 New session started: {session_id}\n")
    
    # Simulate a multi-turn conversation
    queries = [
        "What are the best practices for building AI agents?",
        "Can you elaborate on memory management specifically?",
        "How do agents share context in your system?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─'*70}")
        print(f"👤 USER QUERY {i}: {query}")
        print(f"{'─'*70}")
        
        response = await orchestrator.process_query(session_id, query)
        
        print(f"\n💬 FINAL RESPONSE:")
        print(f"{'─'*40}")
        print(response)
        
        # Show memory state after each turn
        print(f"\n📊 MEMORY STATE AFTER TURN {i}:")
        context = shared_memory.build_agent_context(session_id, query, "debug")
        print(f"   - Conversation entries: {len(context['conversation_history'])}")
        print(f"   - Knowledge entries: {len(context['relevant_knowledge'])}")
        print(f"   - Known entities: {len(context['known_entities'])}")
        print(f"   - Context usage: {context['context_usage']['usage_percent']}%")
    
    print("\n" + "="*70)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*70)
    
    # Final summary
    print("\n📋 KEY TAKEAWAYS:")
    print("""
    1. SHARED MEMORY: All 3 agents read/write to the same SharedMemoryLayer
    2. PROGRAMMATIC OPS: Memory reads happen automatically at each agent turn
    3. CONTEXT BUILDING: Each agent receives full context including other agents' work
    4. INTER-AGENT COMM: Agents pass messages through the orchestrator
    5. SESSION ISOLATION: Thread ID keeps conversations separate
    6. AUTO SUMMARIZATION: Context is compressed when threshold exceeded
    """)


if __name__ == "__main__":
    asyncio.run(main())

