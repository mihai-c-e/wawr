from datetime import datetime
from typing import Any, List, Callable

import pandas as pd


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
