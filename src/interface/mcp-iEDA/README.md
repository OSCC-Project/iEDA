## Build

Docker build:

```bash
cd src/interface/mcp-iEDA
docker build -t mcp-ieda:1.0 .
```
you can modify the Dockerfile to suit your needs, such as change MCP_SERVER_TYPE to "stdio" or "sse". If you want to use "sse" mode, you need config mcp server use mcp-ieda-sse method below.

## Installation

### Using PIP
```
pip install mcp-iEDA
```

After installation, you can run it as a script using:

```
python -m mcp_ieda
```

## Configuration
### Usage with VS Code
you can add it to a file called `.vscode/mcp.json` in your workspace. The github Copilot extension will automatically detect and use this configuration.The iEDA Path need to be set in the environment variable.
<details>
<summary>Using pip installation</summary>

```json
"mcpServers": {
    "mcp-ieda": {
        "type": "stdio",
        "command": "python",
        "args": [
            "-m",
            "mcp_ieda"
        ],
        "env": {
            "iEDA": "${workspaceFolder}/scripts/design/sky130_gcd/iEDA",
            "WORKSPACE":"${workspaceFolder}/scripts/design/sky130_gcd",
            "MCP_SERVER_TYPE":"stdio"
        }
    },
    "mcp-ieda-sse": {
        "type": "sse",
        "url": "http://localhost:3002/sse"
    }
}
```
</details>

<details>
<summary>Using docker</summary>

```json
"mcpServers": {
  "mcp-ieda": {
    "command": "docker",
    "args": [
        "run", 
        "-p", 
        "3002:3002",
        "-v",
        "/lib/x86_64-linux-gnu/libgomp.so.1:/lib/x86_64-linux-gnu/libgomp.so.1",
        "-v",
        "/lib/x86_64-linux-gnu/libunwind.so.8:/lib/x86_64-linux-gnu/libunwind.so.8",
        "--rm",
        "-i",
        "mcp-ieda:1.0"
    ]
  }
}
```
</details>

## Example
If you config mcp server as above, you can run mcp server by prompt in VSCode Copilot on Agent mode : "run iEDA example gcd".

