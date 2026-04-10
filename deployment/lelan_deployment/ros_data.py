"""Small helper for storing the latest ROS message payload with timeouts."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional


class ROSData:
    def __init__(self, timeout: float = 3.0, queue_size: int = 1, name: str = ""):
        self.timeout = timeout
        self.last_time_received = float("-inf")
        self.queue_size = queue_size
        self.data: Any = None
        self.name = name

    def get(self) -> Any:
        return self.data

    def set(self, data: Any) -> None:
        time_waited = time.monotonic() - self.last_time_received
        if self.queue_size == 1:
            self.data = data
        else:
            if self.data is None or time_waited > self.timeout:
                self.data = []
            if len(self.data) == self.queue_size:
                self.data.pop(0)
            self.data.append(data)
        self.last_time_received = time.monotonic()

    def is_valid(self, verbose: bool = False, log_fn: Optional[Callable[[str], None]] = None) -> bool:
        time_waited = time.monotonic() - self.last_time_received
        valid = time_waited < self.timeout
        if self.queue_size > 1:
            valid = valid and self.data is not None and len(self.data) == self.queue_size
        if verbose and not valid:
            message = (
                f"Not receiving {self.name} data for {time_waited:.2f} seconds "
                f"(timeout: {self.timeout:.2f} seconds)"
            )
            (log_fn or print)(message)
        return valid
