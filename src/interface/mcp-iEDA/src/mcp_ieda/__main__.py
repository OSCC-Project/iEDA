# __main__.py

from mcp_ieda import main, get_ieda_path

try:
    ieda_path = get_ieda_path()
    print(f"iEDA path: {ieda_path}")
except EnvironmentError as e:
    ieda_path = None
    print(e)
    
main(ieda_path, verbose=2)