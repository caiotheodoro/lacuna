"""A tiny MCP server over streamable HTTP: one tool that succeeds, one that fails."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("spike", host="0.0.0.0", port=8000)


@mcp.tool()
def cancel_order(order_id: str) -> str:
    """Cancel an order."""
    return f"cancelled {order_id}"


@mcp.tool()
def always_fails(order_id: str) -> str:
    """Raise, so the result is a tool error, not a transport error."""
    raise ValueError(f"cannot cancel {order_id}")


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
