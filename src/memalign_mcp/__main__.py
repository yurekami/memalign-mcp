"""Entry point for running MemAlign MCP server via python -m memalign_mcp."""

from memalign_mcp.server import mcp

if __name__ == "__main__":
    mcp.run()
