"""Deterministic, local-only fixture materialization for full-run evaluation.

The recipes in :mod:`evals.structured_context.scenarios` describe pressure at
production scale.  A materializer deliberately caps physical output so unit
and smoke runs remain inexpensive, while preserving the original recipe and
pressure metadata in ``.eval-fixture.json``.  No fixture downloads packages or
contacts a network service.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scenarios import FixtureRecipe, Scenario


FIXTURE_METADATA_FILE = ".eval-fixture.json"
_HASH_IGNORED_TOP_LEVEL = {
    ".agent",
    ".eval-run",
    ".git",
    ".sessions",
    ".traces",
    "__pycache__",
}
_HASH_IGNORED_FILES = {FIXTURE_METADATA_FILE}


class FixtureMaterializationError(RuntimeError):
    """Raised for an invalid recipe or an unsafe/non-empty destination."""


@dataclass(frozen=True)
class MaterializedFixture:
    """The stable facts needed to copy and verify a generated workspace."""

    root: Path
    fixture_hash: str
    file_hashes: dict[str, str]
    scenario_id: str | None
    template: str
    seed: int
    requested_parameters: dict[str, Any]
    materialized_parameters: dict[str, Any]
    pressure: dict[str, Any]

    @property
    def metadata_path(self) -> Path:
        return self.root / FIXTURE_METADATA_FILE

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "fixture_hash": self.fixture_hash,
            "file_hashes": dict(self.file_hashes),
            "scenario_id": self.scenario_id,
            "template": self.template,
            "seed": self.seed,
            "requested_parameters": dict(self.requested_parameters),
            "materialized_parameters": dict(self.materialized_parameters),
            "pressure": dict(self.pressure),
        }


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _should_hash(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if not relative.parts:
        return False
    if relative.parts[0] in _HASH_IGNORED_TOP_LEVEL:
        return False
    return relative.name not in _HASH_IGNORED_FILES


def workspace_file_hashes(root: str | Path) -> dict[str, str]:
    """Return deterministic content hashes, excluding runtime-only state."""

    workspace = Path(root)
    result: dict[str, str] = {}
    for path in sorted(workspace.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink() or not _should_hash(workspace, path):
            continue
        relative = path.relative_to(workspace).as_posix()
        result[relative] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def workspace_hash(root: str | Path) -> str:
    """Hash relative path + bytes, independent of absolute temp directories."""

    digest = hashlib.sha256()
    for relative, file_hash in workspace_file_hashes(root).items():
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def copy_fixture(source: str | Path, destination: str | Path) -> Path:
    """Copy a complete fixture into a fresh destination without mutating source."""

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_dir():
        raise FixtureMaterializationError(f"fixture source does not exist: {source_path}")
    if destination_path.exists():
        if any(destination_path.iterdir()):
            raise FixtureMaterializationError(f"fixture destination is not empty: {destination_path}")
        destination_path.rmdir()
    shutil.copytree(source_path, destination_path, symlinks=False)
    return destination_path


class FixtureMaterializer:
    """Generate all eight published templates using only the standard library."""

    def __init__(
        self,
        *,
        scale: float = 0.20,
        max_files: int = 24,
        max_lines_per_file: int = 180,
        max_records: int = 600,
        max_log_chars: int = 32_000,
    ) -> None:
        if not 0 < scale <= 1:
            raise ValueError("scale must be in (0, 1]")
        self.scale = scale
        self.max_files = max(1, int(max_files))
        self.max_lines_per_file = max(8, int(max_lines_per_file))
        self.max_records = max(1, int(max_records))
        self.max_log_chars = max(256, int(max_log_chars))

    def materialize(
        self,
        source: Scenario | FixtureRecipe | Mapping[str, Any] | object,
        destination: str | Path,
    ) -> MaterializedFixture:
        """Create a fresh fixture.  Existing non-empty destinations are rejected."""

        recipe, scenario_id, pressure = _extract_recipe(source)
        root = Path(destination)
        if root.exists() and any(root.iterdir()):
            raise FixtureMaterializationError(f"fixture destination is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        rng = random.Random(recipe.seed)

        builders = {
            "python-micro-package": self._python_micro_package,
            "python-log-pipeline": self._python_log_pipeline,
            "python-multimodule-service": self._python_multimodule_service,
            "python-generated-tree": self._python_generated_tree,
            "javascript-browser-app": self._javascript_browser_app,
            "mixed-config-repo": self._mixed_config_repo,
            "git-recovery-repo": self._git_recovery_repo,
            "python-monorepo": self._python_monorepo,
        }
        try:
            builder = builders[recipe.template]
        except KeyError as exc:
            raise FixtureMaterializationError(f"unsupported fixture template: {recipe.template}") from exc

        materialized_parameters = builder(root, dict(recipe.parameters), rng)
        self._write_text(
            root / "README.md",
            "# Offline evaluation fixture\n\n"
            "Generated deterministically from a local recipe. Network access is not required.\n",
        )
        file_hashes = workspace_file_hashes(root)
        fixture_hash = workspace_hash(root)
        metadata = {
            "schema_version": "1.0",
            "network_allowed": False,
            "scenario_id": scenario_id,
            "fixture": recipe.to_dict(),
            "pressure": pressure,
            "materializer": {
                "scale": self.scale,
                "max_files": self.max_files,
                "max_lines_per_file": self.max_lines_per_file,
                "max_records": self.max_records,
                "max_log_chars": self.max_log_chars,
            },
            "materialized_parameters": materialized_parameters,
            "fixture_hash": fixture_hash,
            "file_hashes": file_hashes,
        }
        self._write_text(root / FIXTURE_METADATA_FILE, _canonical_json(metadata) + "\n")
        return MaterializedFixture(
            root=root,
            fixture_hash=fixture_hash,
            file_hashes=file_hashes,
            scenario_id=scenario_id,
            template=recipe.template,
            seed=recipe.seed,
            requested_parameters=dict(recipe.parameters),
            materialized_parameters=materialized_parameters,
            pressure=pressure,
        )

    def _scaled(self, value: object, *, maximum: int, minimum: int = 1) -> int:
        try:
            requested = int(value)
        except (TypeError, ValueError):
            requested = minimum
        requested = max(minimum, requested)
        scaled = max(minimum, round(requested * self.scale))
        return min(scaled, maximum)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")

    def _padding(self, count: object, *, label: str) -> str:
        lines = self._scaled(count, maximum=self.max_lines_per_file, minimum=0)
        return "".join(f"# {label} padding {index:04d}\n" for index in range(lines))

    def _python_micro_package(
        self, root: Path, params: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        defect = str(params.get("defect", "inclusive-range-off-by-one"))
        padding = self._padding(params.get("padding_lines", 0), label="micro")
        modules = self._scaled(params.get("module_count", 1), maximum=self.max_files)
        tests = self._scaled(params.get("test_count", 1), maximum=12)

        cases: dict[str, tuple[str, str]] = {
            "inclusive-range-off-by-one": (
                "def inclusive_total(start: int, end: int) -> int:\n"
                "    return sum(range(start, end))\n",
                "from totals import inclusive_total\n\n"
                "class FixtureTests(unittest.TestCase):\n"
                "    def test_inclusive_total(self):\n"
                "        self.assertEqual(inclusive_total(1, 3), 6)\n",
            ),
            "retry-config-validation": (
                "def parse_retry_count(value: str) -> int:\n"
                "    try:\n"
                "        return int(value)\n"
                "    except (TypeError, ValueError):\n"
                "        return 3\n",
                "from config import parse_retry_count\n\n"
                "class FixtureTests(unittest.TestCase):\n"
                "    def test_negative_uses_default(self):\n"
                "        self.assertEqual(parse_retry_count('-1'), 3)\n"
                "    def test_zero_is_valid(self):\n"
                "        self.assertEqual(parse_retry_count('0'), 0)\n",
            ),
            "unstable-json-order": (
                "import json\n\n"
                "def serialize(value: object) -> str:\n"
                "    return json.dumps(value)\n",
                "from serializer import serialize\n\n"
                "class FixtureTests(unittest.TestCase):\n"
                "    def test_stable_order(self):\n"
                "        self.assertEqual(serialize({'b': 1, 'a': 2}), '{\\\"a\\\": 2, \\\"b\\\": 1}\\n')\n",
            ),
            "field-alias-map": (
                "def normalize_fields(payload: dict[str, object]) -> dict[str, object]:\n"
                "    return dict(payload)\n",
                "from parser import normalize_fields\n\n"
                "class FixtureTests(unittest.TestCase):\n"
                "    def test_alias(self):\n"
                "        self.assertEqual(normalize_fields({'user_id': 2})['userId'], 2)\n",
            ),
            "decoder-empty-frame": (
                "def decode_frame(frame: bytes) -> str:\n"
                "    return frame.decode('utf-8')\n",
                "from decoder import decode_frame\n\n"
                "class FixtureTests(unittest.TestCase):\n"
                "    def test_empty_frame(self):\n"
                "        self.assertEqual(decode_frame(b''), '')\n",
            ),
            "non-atomic-write": (
                "from pathlib import Path\n\n"
                "def write_text(path: str, content: str) -> None:\n"
                "    Path(path).write_text(content, encoding='utf-8')\n",
                "from writer import write_text\n\n"
                "class FixtureTests(unittest.TestCase):\n"
                "    def test_write(self):\n"
                "        self.assertTrue(callable(write_text))\n",
            ),
            "cache-key-order": (
                "def cache_key(values: dict[str, object]) -> str:\n"
                "    return '&'.join(f'{key}={value}' for key, value in values.items())\n",
                "from cache_key import cache_key\n\n"
                "class FixtureTests(unittest.TestCase):\n"
                "    def test_stable_key(self):\n"
                "        self.assertEqual(cache_key({'b': 2, 'a': 1}), 'a=1&b=2')\n",
            ),
        }
        module_name = {
            "inclusive-range-off-by-one": "totals.py",
            "retry-config-validation": "config.py",
            "unstable-json-order": "serializer.py",
            "field-alias-map": "parser.py",
            "decoder-empty-frame": "decoder.py",
            "non-atomic-write": "writer.py",
            "cache-key-order": "cache_key.py",
        }.get(defect, "app.py")
        source, test_body = cases.get(
            defect,
            (
                "def run(value: int) -> int:\n    return value\n",
                "from app import run\n\nclass FixtureTests(unittest.TestCase):\n    def test_run(self):\n        self.assertEqual(run(1), 1)\n",
            ),
        )
        self._write_text(root / module_name, source + "\n" + padding)
        for index in range(1, modules + 1):
            self._write_text(
                root / f"module_{index}.py",
                f"SEED = {rng.randrange(1_000_000)}\n\ndef helper_{index}() -> int:\n    return {index}\n",
            )
        extra_tests = "".join(
            f"    def test_generated_{index}(self):\n        self.assertTrue({index} >= 0)\n"
            for index in range(1, tests)
        )
        self._write_text(
            root / "tests" / "test_fixture.py",
            "import unittest\n\n" + test_body + extra_tests + "\n\nif __name__ == '__main__':\n    unittest.main()\n",
        )
        return {"module_count": modules, "test_count": tests, "padding_lines": padding.count("\n")}

    def _python_log_pipeline(
        self, root: Path, params: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        defect = str(params.get("defect", "timezone-colon"))
        log_chars = self._scaled(params.get("log_chars", 4096), maximum=self.max_log_chars, minimum=256)
        records = self._scaled(params.get("record_count", 10), maximum=self.max_records)
        marker_position = str(params.get("marker_position", "middle"))
        marker = {
            "timezone-colon": "TZ_COLON_CASE",
            "group-boundary-reset": "GROUP_BOUNDARY_CASE",
            "registry-normalization": "REGISTRY_CASEFOLD_FAILURE",
        }.get(defect, "PIPELINE_CASE")
        log_text = _sized_marker_text(log_chars, marker, marker_position, seed=rng.randrange(1_000_000))
        self._write_text(root / "generated.log", log_text)
        self._write_text(root / "build.log", log_text)
        parser = (
            "from datetime import datetime\n\n"
            "def parse_timestamp(value: str) -> datetime:\n"
            "    if ':' in value[-6:]:\n"
            "        raise ValueError('timezone colon unsupported')\n"
            "    return datetime.fromisoformat(value)\n"
        )
        aggregator = (
            "def group_records(records):\n"
            "    grouped = {}\n"
            "    for record in records:\n"
            "        current = {}  # reset inside loop\n"
            "        current.setdefault(record['kind'], []).append(record)\n"
            "        grouped.update(current)\n"
            "    return grouped\n"
        )
        registry = "def normalize_registry_name(value: str) -> str:\n    return value\n"
        self._write_text(root / "parser.py", parser)
        self._write_text(root / "aggregator.py", aggregator)
        self._write_text(root / "registry.py", registry)
        test_by_defect = {
            "timezone-colon": "from parser import parse_timestamp\n\n    def test_timezone(self):\n        self.assertIsNotNone(parse_timestamp('2020-01-01T00:00:00+08:00'))\n",
            "group-boundary-reset": "from aggregator import group_records\n\n    def test_grouping(self):\n        self.assertEqual(len(group_records([{'kind': 'x'}, {'kind': 'x'}])['x']), 2)\n",
            "registry-normalization": "from registry import normalize_registry_name\n\n    def test_casefold(self):\n        self.assertEqual(normalize_registry_name('Invoice'), 'invoice')\n",
        }
        body = test_by_defect.get(defect, "    def test_records(self):\n        self.assertGreaterEqual(1, 1)\n")
        self._write_text(
            root / "tests" / "test_pipeline.py",
            "import unittest\n\nclass PipelineTests(unittest.TestCase):\n" + body + "\n\nif __name__ == '__main__':\n    unittest.main()\n",
        )
        return {"log_chars": len(log_text), "record_count": records, "marker": marker}

    def _python_generated_tree(
        self, root: Path, params: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        file_count = self._scaled(params.get("file_count", 1), maximum=self.max_files)
        lines_per_file = self._scaled(
            params.get("lines_per_file", 20), maximum=self.max_lines_per_file, minimum=8
        )
        target = str(params.get("target_symbol", "generated_symbol"))
        duplicates = self._scaled(params.get("duplicate_symbols", 0), maximum=8, minimum=0)
        stale = self._scaled(params.get("stale_fact_count", 0), maximum=32, minimum=0)
        self._write_text(root / "src" / "__init__.py", "")
        for index in range(file_count):
            filler = "".join(f"# generated line {line:04d}\n" for line in range(lines_per_file - 4))
            self._write_text(
                root / "src" / f"generated_{index:03}.py",
                f"SEED = {rng.randrange(1_000_000)}\n\ndef generated_symbol_{index}(value):\n    return value\n\n" + filler,
            )
        handler = (
            "def handle_invoice_v17(invoice: dict[str, str]) -> dict[str, str]:\n"
            "    return {'currency': invoice.get('currency')}\n"
        )
        self._write_text(root / "generated_handlers.py", handler)
        production_ids = (
            "def normalize_customer_id(value: str) -> str:\n"
            "    return value.strip()\n\n"
            "def fetch_user_record(user_id: str) -> dict[str, str]:\n"
            "    return {'id': user_id}\n\n"
            "def load_layered_config() -> dict[str, str]:\n"
            "    return {'order': 'base-first'}\n\n"
            "def stable_schedule(items):\n"
            "    return list(items)\n"
        )
        self._write_text(root / "src" / "production_ids.py", production_ids)
        for index in range(duplicates):
            self._write_text(
                root / "src" / f"duplicate_{index}.py",
                f"def normalize_customer_id(value):\n    return 'duplicate-{index}-' + value\n",
            )
        self._write_text(
            root / "stale_findings.md",
            "".join(f"- stale finding {index}\n" for index in range(stale)),
        )
        test = (
            "import unittest\n\n"
            "class GeneratedTreeTests(unittest.TestCase):\n"
            "    def test_target_is_declared(self):\n"
            f"        self.assertTrue({target!r})\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n"
        )
        self._write_text(root / "tests" / "test_generated_tree.py", test)
        self._write_text(
            root / "tests" / "test_production_ids.py",
            "import unittest\nfrom src.production_ids import normalize_customer_id\n\n"
            "class ProductionIdsTests(unittest.TestCase):\n"
            "    def test_normalizes(self):\n        self.assertEqual(normalize_customer_id(' a '), 'a')\n",
        )
        return {
            "file_count": file_count,
            "lines_per_file": lines_per_file,
            "duplicate_symbols": duplicates,
            "stale_fact_count": stale,
        }

    def _javascript_browser_app(
        self, root: Path, params: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        components = self._scaled(params.get("component_count", 1), maximum=self.max_files)
        events = self._scaled(params.get("event_count", 1), maximum=32)
        padding = self._padding(params.get("padding_lines", 0), label="browser")
        app = (
            "let attempts = 1;\n"
            "let status = 'in progress';\n\n"
            "function restart() {\n"
            "  status = 'restarted';\n"
            "}\n\n"
            "module.exports = { restart, get attempts() { return attempts; }, get status() { return status; } };\n"
            + padding
        )
        self._write_text(root / "app.js", app)
        for index in range(components):
            self._write_text(root / f"component_{index}.js", f"module.exports = {index};\n")
        self._write_text(
            root / "tests" / "check.js",
            "const assert = require('assert');\nconst app = require('../app');\n"
            "app.restart();\nassert.strictEqual(app.attempts, 0);\nassert.strictEqual(app.status, 'ready');\n"
            f"console.log('checked {events} events');\n",
        )
        return {"component_count": components, "event_count": events, "padding_lines": padding.count("\n")}

    def _python_multimodule_service(
        self, root: Path, params: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        modules = self._scaled(params.get("module_count", 1), maximum=self.max_files)
        tests = self._scaled(params.get("test_count", 1), maximum=12)
        depth = self._scaled(params.get("dependency_depth", 1), maximum=8)
        self._write_text(root / "domain" / "__init__.py", "")
        self._write_text(
            root / "service.py",
            "PUBLIC_API_VERSION = 'v1'\n\ndef validate_order(order: dict) -> bool:\n    return bool(order.get('id'))\n\ndef create_order(order: dict) -> dict:\n    if not validate_order(order):\n        raise ValueError('invalid order')\n    return dict(order)\n",
        )
        for index in range(modules):
            self._write_text(
                root / "app" / f"layer_{index:03}.py",
                f"def layer_{index}(value):\n    return value\n",
            )
        self._write_text(root / "catalog" / "__init__.py", "")
        self._write_text(
            root / "catalog" / "export.py",
            "def export_catalog(items):\n    return list(items)\n\n# EMPTY_PAGE_MUST_ADVANCE\n",
        )
        self._write_text(root / "decimal_precision.py", "DECIMAL_PRECISION = '0.10'\n")
        self._write_text(root / "docs" / "adr-null-policy.md", "# Null handling\n\nDraft policy.\n")
        self._write_text(root / "tests" / "golden" / "responses.json", "{\"status\": \"ok\"}\n")
        self._write_text(
            root / "tests" / "test_service.py",
            "import unittest\nfrom service import create_order\n\nclass ServiceTests(unittest.TestCase):\n"
            "    def test_create(self):\n        self.assertEqual(create_order({'id': 'x'})['id'], 'x')\n"
            + "".join(
                f"    def test_generated_{index}(self):\n        self.assertTrue({index} >= 0)\n"
                for index in range(1, tests)
            ),
        )
        return {"module_count": modules, "test_count": tests, "dependency_depth": depth}

    def _mixed_config_repo(
        self, root: Path, params: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        records = self._scaled(params.get("json_records", 1), maximum=self.max_records)
        unicode_paths = bool(params.get("unicode_paths", False))
        defect = str(params.get("defect", "missing-kind-record"))
        rows = [
            {"id": index, "kind": "item", "name": f"record-{index:04d}"}
            for index in range(records)
        ]
        if rows:
            rows[len(rows) // 2].pop("kind")
        self._write_text(root / "records.json", _canonical_json(rows) + "\n")
        self._write_text(
            root / "loader.py",
            "import json\n\ndef load_records(path: str):\n"
            "    records = json.loads(open(path, encoding='utf-8').read())\n"
            "    return [record for record in records if record.get('kind')]\n",
        )
        self._write_text(root / "docs" / "migration.md", "# Config migration\n\nKeep unknown fields.\n")
        self._write_text(root / "tests" / "golden" / "v2.json", "{\"schema_version\": 1, \"x-extension\": true}\n")
        if unicode_paths:
            self._write_text(root / "配置" / "生产环境.json", "{\"重试次数\": 3, \"名称\": \"生产\"}\n")
            self._write_text(root / "unicode_names.py", "DISPLAY_NAME = 'café'\n")
        self._write_text(
            root / "tests" / "test_loader.py",
            "import unittest\nfrom loader import load_records\n\nclass LoaderTests(unittest.TestCase):\n"
            "    def test_loader_runs(self):\n        self.assertIsInstance(load_records('records.json'), list)\n",
        )
        return {"json_records": records, "unicode_paths": unicode_paths, "defect": defect}

    def _git_recovery_repo(
        self, root: Path, params: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        self._write_text(root / "src" / "orders.py", "def order_total(items):\n    return sum(items)\n")
        self._write_text(root / "unrelated.txt", "initial unrelated content\n")
        self._write_text(
            root / "tests" / "test_orders.py",
            "import unittest\nfrom src.orders import order_total\n\nclass OrderTests(unittest.TestCase):\n"
            "    def test_total(self):\n        self.assertEqual(order_total([1, 2]), 3)\n",
        )
        git_initialized = self._initialize_git(root)
        return {
            "drift": str(params.get("drift", "LOW")),
            "branch_change": bool(params.get("branch_change", False)),
            "git_initialized": git_initialized,
        }

    def _python_monorepo(
        self, root: Path, params: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        packages = self._scaled(params.get("package_count", 1), maximum=12)
        files_per_package = self._scaled(params.get("files_per_package", 1), maximum=8)
        for package_index in range(packages):
            package = root / "packages" / f"pkg_{package_index:02d}"
            self._write_text(package / "__init__.py", "")
            for file_index in range(files_per_package):
                self._write_text(
                    package / f"module_{file_index}.py",
                    f"def value_{file_index}():\n    return {package_index + file_index}\n",
                )
            self._write_text(
                package / "test_package.py",
                "import unittest\n\nclass PackageTests(unittest.TestCase):\n"
                f"    def test_package_{package_index}(self):\n        self.assertGreaterEqual({package_index}, 0)\n",
            )
        self._write_text(root / "events.py", "EVENT_LOSS_COUNT = 1\n")
        if bool(params.get("large_outputs", False)):
            self._write_text(root / "logs" / "build.log", _sized_marker_text(4096, "MONOREPO_BUILD_MARKER", "tail", seed=rng.randrange(1_000_000)))
        return {"package_count": packages, "files_per_package": files_per_package}

    @staticmethod
    def _initialize_git(root: Path) -> bool:
        """Initialize a local deterministic repository; this never uses a remote."""

        environment = dict(os.environ)
        environment.update(
            {
                "GIT_AUTHOR_NAME": "NovaCode Fixture",
                "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                "GIT_COMMITTER_NAME": "NovaCode Fixture",
                "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
                "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
                "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
            }
        )
        try:
            subprocess.run(["git", "init", "-q"], cwd=root, env=environment, check=True, capture_output=True)
            subprocess.run(["git", "add", "."], cwd=root, env=environment, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "fixture baseline"],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return False
        return True


def _extract_recipe(
    source: Scenario | FixtureRecipe | Mapping[str, Any] | object,
) -> tuple[FixtureRecipe, str | None, dict[str, Any]]:
    """Accept local recipes and Phase-1 schema objects through a small protocol."""

    if isinstance(source, FixtureRecipe):
        return source, None, {}
    if isinstance(source, Scenario):
        return source.fixture, source.id, dict(source.pressure)
    if isinstance(source, Mapping):
        if "fixture" in source:
            fixture_raw = source["fixture"]
            if not isinstance(fixture_raw, Mapping):
                raise FixtureMaterializationError("scenario.fixture must be an object")
            return (
                FixtureRecipe.from_mapping(fixture_raw),
                str(source.get("id")) if source.get("id") is not None else None,
                dict(source.get("pressure") or {}),
            )
        return FixtureRecipe.from_mapping(source), None, {}
    fixture = getattr(source, "fixture", None)
    if fixture is not None:
        if isinstance(fixture, FixtureRecipe):
            recipe = fixture
        elif isinstance(fixture, Mapping):
            recipe = FixtureRecipe.from_mapping(fixture)
        else:
            recipe = FixtureRecipe(
                template=str(getattr(fixture, "template")),
                seed=int(getattr(fixture, "seed")),
                parameters=dict(getattr(fixture, "parameters", {})),
            )
        return recipe, getattr(source, "id", None), dict(getattr(source, "pressure", {}) or {})
    raise FixtureMaterializationError(f"cannot extract fixture recipe from {type(source).__name__}")


def _sized_marker_text(size: int, marker: str, position: str, *, seed: int) -> str:
    """Create exact-ish deterministic logs while guaranteeing marker retention."""

    target = max(size, len(marker) + 3)
    ratios = {"head": 0.08, "middle": 0.50, "tail": 0.88, "last-2-percent": 0.97, "last-5-percent": 0.94}
    ratio = ratios.get(position, 0.50)
    marker_line = f"\n{marker}\n"
    before_size = max(0, min(target - len(marker_line), int(target * ratio) - len(marker_line) // 2))
    pattern = f"{seed:06d} INFO deterministic fixture output\n"
    before = (pattern * ((before_size // len(pattern)) + 1))[:before_size]
    after_size = target - len(before) - len(marker_line)
    after = (pattern * ((after_size // len(pattern)) + 1))[:after_size]
    return before + marker_line + after


def materialize_fixture(
    source: Scenario | FixtureRecipe | Mapping[str, Any] | object,
    destination: str | Path,
    *,
    materializer: FixtureMaterializer | None = None,
) -> MaterializedFixture:
    """Convenience materializer used by tests, the runner, and CLI wiring."""

    return (materializer or FixtureMaterializer()).materialize(source, destination)


__all__ = [
    "FIXTURE_METADATA_FILE",
    "FixtureMaterializationError",
    "FixtureMaterializer",
    "MaterializedFixture",
    "copy_fixture",
    "materialize_fixture",
    "workspace_file_hashes",
    "workspace_hash",
]
