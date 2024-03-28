import os


def set_environment_variables(env_vars: dict):
    for env_var, value in env_vars.items():
        os.environ[env_var] = value


def execute_shell_command(command: str):
    returncode = os.system(command)
    
    if returncode != 0:
        raise Exception(f"Shell command failed with return code {returncode}")
