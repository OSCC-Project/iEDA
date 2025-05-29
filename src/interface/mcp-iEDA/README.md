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

```json
{
    "servers": {
        "mcp-ieda": {
            "type": "stdio",
            "command": "python",
            "args": [
                "-m",
                "mcp_ieda"
            ],
            "env": {
                "iEDA": "${workspaceFolder}/scripts/design/sky130_gcd/iEDA",
            }
        }
    }
}
```