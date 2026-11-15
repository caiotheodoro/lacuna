"""Call both tools through the gateway with a fixed W3C traceparent."""

import asyncio
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
HEADERS = {"traceparent": f"00-{TRACE_ID}-b7ad6b7169203331-01"}


async def main(url: str) -> None:
    async with streamablehttp_client(url, headers=HEADERS) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            ok = await session.call_tool("cancel_order", {"order_id": "O-123"})
            bad = await session.call_tool("always_fails", {"order_id": "O-456"})
            print("ok.isError =", ok.isError, "| bad.isError =", bad.isError)
            print("sent trace id", TRACE_ID)


asyncio.run(main(sys.argv[1]))
