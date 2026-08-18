"""One-time registration of the NovaCode method inside the AMA-Bench repo.

Run from the AMA-Bench repository root so ``src`` is importable::

    python -m ama_bench.register_ama --novacode-root D:/novacode2

This binds :class:`ama_bench.method.NovaCodeMemoryMethod` to AMA-Bench's real
``BaseMethod`` (via multiple inheritance, so the abstract-method check passes)
and registers it under the method name ``novacode``.  After registration,
``--method novacode`` works in ``src/run.py`` for the current interpreter
session; for persistent registration, drop the same two lines into a small
``sitecustomize.py`` at the benchmark root as described in the integration
README.

The method re-exports the AMA-compatible entry point even when AMA-Bench is
not importable (the local ``BaseMethod`` fallback in :mod:`ama_bench.method`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def register(*, novacode_root: str | Path | None = None) -> str:
    """Register the method under the name ``novacode``; returns the method name."""
    if novacode_root is not None:
        root = Path(novacode_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        from src.method.base_method import BaseMethod as AmaBaseMethod
        from src.method_register import register_method
    except ImportError as exc:
        raise RuntimeError(
            "AMA-Bench is not importable. Run this from the AMA-Bench repository "
            "root (or point PYTHONPATH at it) so 'src' resolves."
        ) from exc
    from ama_bench.method import NovaCodeMemoryMethod

    class NovacodeAmaMethod(AmaBaseMethod, NovaCodeMemoryMethod):
        """NovaCode method bound to AMA-Bench's real BaseMethod."""

    register_method("novacode", NovacodeAmaMethod)
    return "novacode"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--novacode-root", default=None, help="Path to the NovaCode checkout (added to sys.path)")
    args = parser.parse_args()
    name = register(novacode_root=args.novacode_root)
    print(f"registered method: {name}")


if __name__ == "__main__":
    main()
