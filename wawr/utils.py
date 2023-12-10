import os
import json
from typing import Dict, Set, Any

def check_env(required_keys: Set) -> None:
    present = required_keys.intersection(os.environ.keys())
    if len(present) < len(required_keys):
        raise RuntimeError(f'Missing credentials in env: {required_keys.difference(present)}')

def read_json(s: str) -> Dict[Any, Any]:
    if '```json' in s:
        s = s.split('```json')[1]
    if 'json```' in s:
        s = s.split('json```')[1]
    if '```' in s:
        s = s.split('```')[0]
    return json.loads(s)

def read_html(s: str) -> Dict[Any, Any]:
    if '```html' in s:
        s = s.split('```html')[1]
    if 'html```' in s:
        s = s.split('html```')[1]
    if '```' in s:
        s = s.split('```')[0]
    return s
