from functools import cache

SERVICE_CODE = "R1"


@cache
def label_prefix() -> str:
    return SERVICE_CODE
