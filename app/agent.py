"""
LangChain agent for Indian Legal Rights Advisory.
Uses create_agent (LangGraph) when available; tools: structured law DB + ChromaDB.
"""

import os
from typing import Any

from langchain_core.messages import HumanMessage

# Prefer new LangChain create_agent (LangGraph-based)
try:
    from langchain.agents import create_agent as create_agent_graph
    _USE_NEW_AGENT = True
except ImportError:
    create_agent_graph = None
    _USE_NEW_AGENT = False

# Fallback: legacy AgentExecutor + tool-calling (older LangChain)
if not _USE_NEW_AGENT:
    try:
        from langchain.agents.agent import AgentExecutor
    except ImportError:
        try:
            from langchain.agents import AgentExecutor
        except ImportError:
            AgentExecutor = None  # type: ignore[misc, assignment]
    try:
        from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
    except ImportError:
        try:
            from langchain.agents import create_tool_calling_agent
        except ImportError:
            create_tool_calling_agent = None
    from langchain_community.chat_models import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.tools import get_agent_tools

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = """You are an expert legal rights advisor for Indian citizens. Your role is to:
1. Classify whether the user's query relates to CRIMINAL LAW (IPC, CrPC) or CONSUMER PROTECTION (Consumer Protection Act 2019).
2. Use the provided tools to retrieve relevant law:
   - structured_law_db: get key sections by category ('criminal_law' or 'consumer_protection').
   - vector_search: semantic search over legal text; pass the user's situation as query, and optionally law_category to filter.
3. After gathering information, respond with a single JSON object (no markdown code fence) in this exact format:
{
  "is_crime": "yes" or "no",
  "law_category": "criminal_law" or "consumer_protection",
  "relevant_sections": ["e.g. IPC 506", "CrPC 154"],
  "legal_explanation": "Clear explanation in plain language",
  "citizen_actions": ["Step 1", "Step 2", ...],
  "possible_punishment": "If criminal: punishment; else N/A or remedy",
  "escalation_authority": "e.g. Police station / District Consumer Commission",
  "disclaimer": "This is general information only, not legal advice. Consult a qualified lawyer."
}
Always call at least one tool (structured_law_db and/or vector_search) before giving your final JSON answer. Use law_category to decide which category to query. Output only the JSON object at the end."""


def _get_llm():
    """Local LLM via Ollama."""
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )


def _create_agent_new():
    """Build agent using create_agent (LangGraph) - current LangChain API."""
    model = _get_llm()
    tools = get_agent_tools()
    graph = create_agent_graph(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return graph


def _create_agent_legacy():
    """Build agent using AgentExecutor + create_tool_calling_agent (older API)."""
    if AgentExecutor is None or create_tool_calling_agent is None:
        raise ImportError(
            "Legacy agent requires AgentExecutor and create_tool_calling_agent. "
            "Install a LangChain version that provides them, or use the new create_agent API."
        )
    llm = _get_llm()
    tools = get_agent_tools()
    llm_with_tools = llm.bind_tools(tools)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ])
    agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=os.environ.get("AGENT_VERBOSE", "0").lower() in ("1", "true"),
        handle_parsing_errors=True,
        max_iterations=5,
    )


def create_agent():
    """Build the legal advisory agent (new or legacy API)."""
    if _USE_NEW_AGENT:
        return _create_agent_new()
    return _create_agent_legacy()


_agent_instance: Any = None


def get_agent():
    """Singleton agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = create_agent()
    return _agent_instance


def _extract_output_new(result: dict) -> str:
    """Get final assistant message content from LangGraph state."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        name = type(msg).__name__ if hasattr(msg, "__class__") else ""
        if name == "HumanMessage" or (isinstance(msg, dict) and msg.get("type") == "human"):
            continue
        content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
        if content is None or content == "":
            continue
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                (c.get("text", c) if isinstance(c, dict) else str(c)) for c in content
            ).strip() or str(content)
        return str(content)
    return ""


def run_query(query: str, chat_history: list[Any] | None = None) -> str:
    """Run the agent on a legal query and return the final answer text."""
    agent = get_agent()
    if _USE_NEW_AGENT:
        inputs = {"messages": [HumanMessage(content=query)]}
        result = agent.invoke(inputs)
        return _extract_output_new(result)
    # Legacy AgentExecutor
    result = agent.invoke({
        "input": query,
        "chat_history": chat_history or [],
    })
    return result.get("output", "")
