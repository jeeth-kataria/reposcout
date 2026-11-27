"""The LLM adapter calls before_request immediately before every provider request."""

import fcntl
import os
import time
from pathlib import Path


def before_request() -> None:
    """Space request starts across localizer processes in the same evaluation run."""
    path = os.environ.get("REPOSCOUT_REQUEST_CLOCK")
    interval = float(os.environ.get("REPOSCOUT_REQUEST_INTERVAL", "0"))
    if not path or not interval:
        return
    with Path(path).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        last = float(handle.read().strip() or "0")
        delay = last + interval - time.time()
        if delay > 0:
            time.sleep(delay)
        handle.seek(0)
        handle.truncate()
        handle.write(str(time.time()))
        handle.flush()
