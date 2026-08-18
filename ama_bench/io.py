"""Robust JSONL loading shared by the runner and the judge.

AMA-Bench observation text contains characters that ``str.splitlines()``
treats as line boundaries (``\\x85`` NEL, ``\\u2028``) even though they are
legal inside JSON strings.  Plain ``splitlines()`` + ``json.loads`` therefore
corrupts such records.  ``iter_jsonl_records`` streams records with
``JSONDecoder.raw_decode``, which handles them correctly.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_WHITESPACE = " \t\r\n"


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per record, tolerating exotic whitespace."""
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    pos = 0
    length = len(text)
    while pos < length:
        while pos < length and text[pos] in _WHITESPACE:
            pos += 1
        if pos >= length:
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed JSONL record at offset {pos} in {path}: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise TypeError(f"JSONL record at offset {pos} in {path} is not an object")
        yield obj
        pos = end
