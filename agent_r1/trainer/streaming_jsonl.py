"""Crash-resilient append-only JSONL support for validation outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _record_key(record: dict[str, Any]) -> str:
    """Use the dataset-stable sample key; retain old-row compatibility."""

    sample_key = str(record.get("sample_key") or "").strip()
    if sample_key:
        return f"sample_key:{sample_key}"
    if "sample_index" in record:
        return f"sample_index:{int(record['sample_index'])}"
    raise KeyError("Streaming result record requires sample_key")


class StreamingJsonlWriter:
    """Append one complete JSON object per durable filesystem write."""

    def __init__(self, path: str | os.PathLike[str], *, resume: bool, fsync: bool = True):
        self.path = Path(path)
        self.fsync = fsync
        self._record_keys: set[str] = set()
        self._fd: int | None = None

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not resume:
                raise FileExistsError(
                    f"Streaming validation file already exists: {self.path}. "
                    "Use a new RUN_ID or set HOTPOTQA_STREAMING_RESUME=1."
                )
            self._load_existing_keys()
        else:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)

        self._fd = os.open(self.path, os.O_APPEND | os.O_WRONLY)

    def _load_existing_keys(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Cannot resume {self.path}: invalid JSON on line {line_number}"
                    ) from exc
                try:
                    record_key = _record_key(record)
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Cannot resume {self.path}: line {line_number} has no valid sample identity"
                    ) from exc
                self._record_keys.add(record_key)

    @property
    def count(self) -> int:
        return len(self._record_keys)

    def append(self, record: dict[str, Any]) -> bool:
        """Append and sync one record; return False when it is a resume duplicate."""

        if self._fd is None:
            raise RuntimeError("StreamingJsonlWriter is closed")
        record_key = _record_key(record)
        if record_key in self._record_keys:
            return False

        payload = (
            json.dumps(record, ensure_ascii=False, default=_json_default, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        offset = 0
        while offset < len(payload):
            offset += os.write(self._fd, payload[offset:])
        if self.fsync:
            os.fsync(self._fd)
        self._record_keys.add(record_key)
        return True

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> StreamingJsonlWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
