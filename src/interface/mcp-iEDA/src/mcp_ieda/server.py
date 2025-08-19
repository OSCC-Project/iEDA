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

current_dir = os.path.split(os.path.abspath(__file__))[0]

class iEDARun(BaseModel):
    script_path: str 
    
class iEDARunExample(BaseModel):
    example_name: str 

class iEDAMcpTools(str, Enum):
    """
    iEDA MCP tools
    """
    iEDA_RUN = "iEDA_RUN"
    iEDA_RUN_EXAMPLE = "iEDA_RUN_EXAMPLE"
    
def run_ieda(iEDA: Path, script_path: str):
    """Run iEDA with the given script path."""
    
    import subprocess  
    script = f"{iEDA} -script {script_path}"
    
    logging.info(f"Run iEDA with script: {script}")

    process = subprocess.run(script, shell=True, check=True)
    if process.returncode != 0:
        raise RuntimeError(f"Subprocess failed with return code {process.returncode}")
    
    
def get_server_url() -> str:
    """
    Get the server URL from environment variable or default to 'http://localhost'.
    """
    return os.getenv("MCP_SERVER_URL", "localhost")

def get_server_port() -> int:
    """
    Get the server port from environment variable or default to 3002.
    """
    return int(os.getenv("MCP_SERVER_PORT", 3002))
    
def serve(iEDA: Path, transport="stdio"):
    logger = logging.getLogger(__name__)
    
    server = Server("mcp-iEDA")
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(name=iEDAMcpTools.iEDA_RUN, description="Run iEDA with script", inputSchema=iEDARun.schema()),
            Tool(name=iEDAMcpTools.iEDA_RUN_EXAMPLE, description="Run iEDA example", inputSchema=iEDARunExample.schema())
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
        elif tool == iEDAMcpTools.iEDA_RUN_EXAMPLE:
            example_name = arguments.get("example_name")
            if not example_name:
                raise ValueError("Missing 'example_name' in arguments")
            logger.info(f"Run iEDA example: {example_name}")
            example_script_path = f"{current_dir}/./example/{example_name}/run_iEDA.tcl"
            if os.path.exists(example_script_path):
                run_ieda(iEDA, example_script_path)
                return [TextContent(type="text", text=f"Run iEDA example: {example_name} successfully")]
            else:
                return [TextContent(type="text", text=f"Example: {example_script_path} not found")]
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

        server_url = get_server_url()
        server_port = get_server_port()
        logger.info(f"Starting iEDA MCP server at {server_url}")
        uvicorn.run(starlette_app, host=server_url, port=server_port)
    else:
        import anyio
        async def arun():
            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream, options, raise_exceptions=True)
                
        anyio.run(arun)


