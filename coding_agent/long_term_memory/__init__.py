"""KV-Cache friendly structured long-term memory system for NovaCode."""
from __future__ import annotations

from .models import MemoryEntry, MemoryHeader, MemoryType, QuotaConfig
from .store import MemoryStore
from .suppression import SuppressionEngine, SuppressionVerdict
from .tools import DeleteMemoryTool, SaveMemoryTool, UpdateMemoryTool, create_memory_tools
from .retriever import MemoryRetriever, PrefetchGate, HeaderScanner, SideQueryEngine, LexicalScorer
from .injector import MemoryInjector
from .hook import MemoryLifecycleHook

__all__ = [
    "MemoryType",
    "MemoryHeader",
    "MemoryEntry",
    "QuotaConfig",
    "MemoryStore",
    "SuppressionEngine",
    "SuppressionVerdict",
    "SaveMemoryTool",
    "UpdateMemoryTool",
    "DeleteMemoryTool",
    "create_memory_tools",
    "MemoryRetriever",
    "PrefetchGate",
    "HeaderScanner",
    "SideQueryEngine",
    "LexicalScorer",
    "MemoryInjector",
    "MemoryLifecycleHook",
]
