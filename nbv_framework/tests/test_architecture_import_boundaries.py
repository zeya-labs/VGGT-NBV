from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, Set


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Keep these rules small and explicit. They are meant to prevent major
# architectural regressions in dependency direction.
_FORBIDDEN_IMPORT_ROOTS: Dict[str, Set[str]] = {
    "application": {"interfaces", "infrastructure"},
    "infrastructure": {"interfaces"},
    "domain": {"interfaces", "infrastructure", "application"},
    "shared": {"interfaces", "infrastructure", "application", "domain"},
}


def _iter_py_files(package_root: Path) -> Iterable[Path]:
    for path in package_root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        if "/tests/" in path.as_posix():
            continue
        yield path


def _extract_import_roots(path: Path) -> Set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if not name.startswith("nbv_framework."):
                    continue
                parts = name.split(".")
                if len(parts) >= 2:
                    roots.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module is None or node.level != 0:
                continue
            if not module.startswith("nbv_framework."):
                continue
            parts = module.split(".")
            if len(parts) >= 2:
                roots.add(parts[1])
    return roots


def test_architecture_import_boundaries() -> None:
    violations = []

    for path in _iter_py_files(_PACKAGE_ROOT):
        top_level_pkg = path.relative_to(_PACKAGE_ROOT).parts[0]
        forbidden = _FORBIDDEN_IMPORT_ROOTS.get(top_level_pkg)
        if not forbidden:
            continue

        imported_roots = _extract_import_roots(path)
        bad = sorted(imported_roots & forbidden)
        if bad:
            violations.append((path.relative_to(_PACKAGE_ROOT).as_posix(), bad))

    assert not violations, "Import boundary violations:\n" + "\n".join(
        f"- {rel_path}: {', '.join(bad_roots)}" for rel_path, bad_roots in violations
    )
