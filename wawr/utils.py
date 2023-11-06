import os
def check_env(required_keys: set) -> None:
    present = required_keys.intersection(os.environ.keys())
    if len(present) < len(required_keys):
        raise RuntimeError(f'Missing credentials in env: {required_keys.difference(present)}')