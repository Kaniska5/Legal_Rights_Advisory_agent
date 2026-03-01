"""
LangChain tool wrappers for the agent.
"""

from langchain_core.tools import tool

from .structured_law_tool import structured_law_search
from .vector_retrieval_tool import vector_retrieval_search


@tool
def structured_law_db(category: str) -> str:
    """
    Look up key law sections from the structured database.
    Use this to get exact section numbers, punishments, and bailability for IPC, CrPC, and Consumer Protection Act 2019.
    category must be either 'criminal_law' or 'consumer_protection'.
    """
    return structured_law_search(category)


@tool
def vector_search(query: str, law_category: str = "") -> str:
    """
    Semantic search over Indian legal text (IPC, CrPC, Consumer Protection Act).
    Use when the user describes a situation and you need relevant legal provisions.
    query: the user's situation or question in natural language.
    law_category: optional - 'criminal_law' or 'consumer_protection' to filter; leave empty to search all.
    """
    return vector_retrieval_search(query, law_category=law_category or None, k=5)


def get_agent_tools():
    """Return list of LangChain tools for the agent."""
    return [structured_law_db, vector_search]
