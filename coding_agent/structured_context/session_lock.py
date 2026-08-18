"""Cross-process advisory lock for one structured session."""
from __future__ import annotations

import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

try:  # POSIX
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False

try:  # Windows
    import msvcrt

    _HAVE_MSVCRT = True
except ImportError:  # pragma: no cover
    msvcrt = None  # type: ignore[assignment]
    _HAVE_MSVCRT = False


class SessionLock(AbstractContextManager[None]):
    def __init__(self, sessions_root: Path, session_id: str, *, timeout: float = 5.0) -> None:
        if Path(session_id).name != session_id:
            raise ValueError(f"Invalid session id: {session_id!r}")
        self.path = Path(sessions_root) / "locks" / f"{session_id}.lock"
        self.timeout = timeout
        self._handle: Any = None

    def __enter__(self) -> SessionLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        if _HAVE_FCNTL:
            import time

            deadline = time.monotonic() + max(0.0, self.timeout)
            while True:
                try:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        self._handle.close()
                        self._handle = None
                        raise TimeoutError(f"Timed out waiting for session lock {self.path}")
                    time.sleep(0.05)
        elif _HAVE_MSVCRT:  # pragma: no cover - Windows
            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(f"pid={os.getpid()}\n")
        self._handle.flush()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._handle is None:
            return False
        try:
            if _HAVE_FCNTL:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            elif _HAVE_MSVCRT:  # pragma: no cover - Windows
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._handle.close()
            self._handle = None
        return False
