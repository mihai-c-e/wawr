from typing import List
from datetime import datetime
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