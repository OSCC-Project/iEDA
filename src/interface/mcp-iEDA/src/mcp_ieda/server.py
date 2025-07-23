#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File : server.py
@Time : 2025/04/27 11:38:51
@Author : simin tao
@Version : 1.0
@Contact : taosm@pcl.ac.cn
@Desc : The mcp server for iEDA.
'''
import logging
import os

from pathlib import Path

from mcp.server import Server
from mcp.server.session import ServerSession
from mcp.server.stdio import stdio_server
from mcp.types import (
    TextContent,
    Tool,
)

from enum import Enum
from pydantic import BaseModel

from pathlib import Path

class iEDARun(BaseModel):
    script_path: str
    

class iEDAMcpTools(str, Enum):
    """
    iEDA MCP tools
    """
    iEDA_RUN = "iEDA_RUN"
    
def run_ieda(iEDA: Path, script_path: str):
    os.system(f"{iEDA} -script {script_path}")
    os.system(f"echo {iEDA} -script {script_path} finished")
    
def serve(iEDA: Path, transport="stdio"):
    logger = logging.getLogger(__name__)
    
    server = Server("mcp-iEDA")
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(name=iEDAMcpTools.iEDA_RUN, description="Run iEDA with script", inputSchema=iEDARun.schema())
        ]

    @server.call_tool()
    async def call_tool(tool: str, arguments: dict) -> list[TextContent]:
        if tool == iEDAMcpTools.iEDA_RUN:
            script_path = arguments.get("script_path")
            if not script_path:
                raise ValueError("Missing 'script_path' in arguments")
            logger.info(f"Run iEDA with script: {script_path}")
            run_ieda(iEDA, script_path)
            return [TextContent(type="text", text=f"Run iEDA with script: {script_path} successfully")]
        else:
            raise ValueError(f"Unknown tool: {tool}")

    options = server.create_initialization_options()
        
    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await server.run(
                    streams[0], streams[1], options
                )

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn

        uvicorn.run(starlette_app, host="127.0.0.1", port=3002)
    else:
        import anyio
        async def arun():
            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream, options, raise_exceptions=True)
                
        anyio.run(arun)


