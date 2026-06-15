"""Agentic layer built on LangGraph.

A supervisor routes each query to the right specialist agent (retrieval, FAQ,
memory, database, analytics) and a response agent synthesises the final,
cited answer. The graph degrades to an equivalent sequential orchestrator if
LangGraph is not installed, so the package is always runnable."""
