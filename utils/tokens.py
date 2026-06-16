import os

def get_or_prompt_token(var_name: str, prompt: str) -> str:
    token = os.getenv(var_name)
    if not token:
        print(f"{var_name} not found in .env.")
        token = input(f"{prompt}: ").strip()
        save = input("Save this token to .env for future use? [y/N] ").strip().lower()
        if save == "y":
            with open(".env", "a") as f:
                f.write(f"{var_name}={token}\n")
        os.environ[var_name] = token
    return token