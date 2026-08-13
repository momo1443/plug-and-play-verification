"""Append-only, single-file JSONL output for training rollouts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


def append_jsonl_records(path: str | os.PathLike[str], records: Iterable[dict[str, Any]], *, fsync: bool = True) -> int:
    """Append records to one JSONL file and return the number written."""

    record_list = list(records)
    if not record_list:
        return 0

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in record_list
    ).encode("utf-8")

    fd = os.open(output_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        if fsync:
            os.fsync(fd)
    finally:
        os.close(fd)
    return len(record_list)
