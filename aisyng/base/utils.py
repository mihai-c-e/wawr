import json
from typing import List, Any, Callable, Dict
from datetime import datetime
import pandas as pd


def strptime_ymdhms(s: str) -> datetime:
    return datetime.strptime(s, "%Y %m %d %H:%M:%S")

def strftime_ymdhms(d: datetime) -> str:
    return d.strftime("%Y %m %d %H:%M:%S")

def strptime_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def strftime_ymd(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")

def strptime_admyhmsgmt(s: str) -> datetime:
    return datetime.strptime(s, "%a, %d %b %Y %H:%M:%S GMT")

def strftime_admyhmsgmt(d: datetime) -> str:
    return d.strftime("%a, %d %b %Y %H:%M:%S GMT")

def _validate_date(obj: Any, date_validators: List[Callable]) -> datetime:
    if isinstance(obj, datetime):
        return obj
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    for validator in date_validators:
        try:
            return validator(obj)
        except Exception:
            pass
    raise ValueError(f"Can't treat value of type: {type(obj)}, or all validators failed: {obj}")

def read_json(s: str) -> Dict[Any, Any]:
    if '```json' in s:
        s = s.split('```json')[1]
    if 'json```' in s:
        s = s.split('json```')[1]
    if '```' in s:
        s = s.split('```')[0]
    #s = s.replace("\\", r"\\\\")
    return json.loads(s)

def read_html(s: str) -> Dict[Any, Any]:
    if '```html' in s:
        s = s.split('```html')[1]
    if 'html```' in s:
        s = s.split('html```')[1]
    if '```' in s:
        s = s.split('```')[0]
    return s
