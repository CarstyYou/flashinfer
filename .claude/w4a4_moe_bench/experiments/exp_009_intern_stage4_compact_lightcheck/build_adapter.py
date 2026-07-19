#!/usr/bin/env python3
"""Build the immutable exp_009 compatibility overlay.

The intern source intentionally remains untouched.  The generated overlay adds
only the three constructor keywords that the current production dispatcher
always passes to ``MoEDynamicKernel``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


EXP_ROOT = Path(__file__).resolve().parent
BENCH_ROOT = EXP_ROOT.parents[1]
DEFAULT_SOURCE = BENCH_ROOT / "moe_dyanmice_kernel_ab_stage4_compact.py"
DEFAULT_OUTPUT_DIR = EXP_ROOT / "results/overlays/intern_stage4_compact"
ADAPTER_NAME = "moe_dynamic_kernel.py"
DIFF_NAME = "adapter.diff"
IDENTITY_NAME = "identity.json"

EXPECTED_ORIGINAL_SHA256 = (
    "91034c7cd3b3b9fe8cbde6dbf1bb2c8c13e4261ff9e9e7d642f3ce9d83788768"
)
INSERTED_KEYWORDS = (
    "swiglu_alpha",
    "swiglu_beta",
    "swiglu_limit",
)
INSERTED_LINES = (
    "        swiglu_alpha: float = 1.702,\n"
    "        swiglu_beta: float = 1.0,\n"
    "        swiglu_limit: float | None = None,\n"
)
SOURCE_ANCHOR = (
    '        activation: str = "silu",\n'
    "        share_input_across_experts: bool = False,\n"
)
ADAPTER_ANCHOR = (
    '        activation: str = "silu",\n'
    + INSERTED_LINES
    + "        share_input_across_experts: bool = False,\n"
)


class AdapterError(RuntimeError):
    """The source or generated adapter violated the fail-closed contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _constructor(tree: ast.AST) -> ast.FunctionDef:
    classes = [
        node
        for node in getattr(tree, "body", ())
        if isinstance(node, ast.ClassDef) and node.name == "MoEDynamicKernel"
    ]
    if len(classes) != 1:
        raise AdapterError(f"expected one MoEDynamicKernel class, got {len(classes)}")
    constructors = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    if len(constructors) != 1:
        raise AdapterError(
            f"expected one MoEDynamicKernel.__init__, got {len(constructors)}"
        )
    return constructors[0]


def _kwonly_names(tree: ast.AST) -> tuple[str, ...]:
    return tuple(arg.arg for arg in _constructor(tree).args.kwonlyargs)


def _without_adapter_keywords(tree: ast.AST) -> ast.AST:
    normalized = copy.deepcopy(tree)
    constructor = _constructor(normalized)
    names = [arg.arg for arg in constructor.args.kwonlyargs]
    for keyword in reversed(INSERTED_KEYWORDS):
        if keyword not in names:
            raise AdapterError(f"adapter constructor is missing {keyword}")
        index = names.index(keyword)
        del names[index]
        del constructor.args.kwonlyargs[index]
        del constructor.args.kw_defaults[index]
    return ast.fix_missing_locations(normalized)


def validate_adapter(source_text: str, adapter_text: str) -> dict[str, Any]:
    """Prove that the adapter differs only by the three keyword parameters."""

    try:
        source_tree = ast.parse(source_text)
        adapter_tree = ast.parse(adapter_text)
    except SyntaxError as error:
        raise AdapterError(f"source/adapter syntax error: {error}") from error

    source_names = _kwonly_names(source_tree)
    adapter_names = _kwonly_names(adapter_tree)
    if any(keyword in source_names for keyword in INSERTED_KEYWORDS):
        raise AdapterError("original source already contains an adapter keyword")
    if (
        "activation" not in source_names
        or "share_input_across_experts" not in source_names
    ):
        raise AdapterError("constructor anchor keywords are missing")
    activation_index = source_names.index("activation")
    if source_names[activation_index + 1] != "share_input_across_experts":
        raise AdapterError("constructor keyword order drifted around activation")
    expected_names = (
        source_names[: activation_index + 1]
        + INSERTED_KEYWORDS
        + source_names[activation_index + 1 :]
    )
    if adapter_names != expected_names:
        raise AdapterError(
            f"adapter keyword order drift: {adapter_names!r} != {expected_names!r}"
        )

    normalized_adapter = _without_adapter_keywords(adapter_tree)
    source_dump = ast.dump(source_tree, include_attributes=False)
    adapter_dump = ast.dump(normalized_adapter, include_attributes=False)
    if adapter_dump != source_dump:
        raise AdapterError("adapter changes semantics outside the constructor keywords")

    diff_lines = list(
        difflib.unified_diff(
            source_text.splitlines(keepends=True),
            adapter_text.splitlines(keepends=True),
            fromfile=DEFAULT_SOURCE.name,
            tofile=ADAPTER_NAME,
            n=3,
        )
    )
    additions = [
        line
        for line in diff_lines
        if line.startswith("+") and not line.startswith("+++")
    ]
    deletions = [
        line
        for line in diff_lines
        if line.startswith("-") and not line.startswith("---")
    ]
    if additions != [f"+{line}" for line in INSERTED_LINES.splitlines(keepends=True)]:
        raise AdapterError(f"unexpected adapter additions: {additions!r}")
    if deletions:
        raise AdapterError(f"adapter unexpectedly deletes source text: {deletions!r}")

    return {
        "ast_equal_after_removing_keywords": True,
        "inserted_keywords": list(INSERTED_KEYWORDS),
        "unified_diff_additions": len(additions),
        "unified_diff_deletions": len(deletions),
        "unified_diff": "".join(diff_lines),
    }


def make_adapter(source_text: str) -> str:
    if source_text.count(SOURCE_ANCHOR) != 1:
        raise AdapterError("expected exactly one constructor text anchor")
    adapter_text = source_text.replace(SOURCE_ANCHOR, ADAPTER_ANCHOR, 1)
    validate_adapter(source_text, adapter_text)
    return adapter_text


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise AdapterError(f"immutable artifact drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_adapter(
    *,
    source: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    expected_original_sha256: str = EXPECTED_ORIGINAL_SHA256,
) -> dict[str, Any]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not source.is_file():
        raise AdapterError(f"missing intern source: {source}")
    original_bytes = source.read_bytes()
    original_sha256 = sha256_bytes(original_bytes)
    if original_sha256 != expected_original_sha256:
        raise AdapterError(
            f"intern source SHA-256 drift: {original_sha256} != {expected_original_sha256}"
        )
    try:
        source_text = original_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AdapterError("intern source is not UTF-8") from error

    adapter_text = make_adapter(source_text)
    validation = validate_adapter(source_text, adapter_text)
    diff_text = str(validation.pop("unified_diff"))
    adapter_bytes = adapter_text.encode("utf-8")
    diff_bytes = diff_text.encode("utf-8")
    adapter_path = output_dir / ADAPTER_NAME
    diff_path = output_dir / DIFF_NAME
    identity_path = output_dir / IDENTITY_NAME

    identity = {
        "schema": "exp009.intern-stage4-adapter-identity.v1",
        "transformation": {
            "kind": "constructor-keyword-compatibility-only",
            **validation,
        },
        "original": {
            "path": source.name,
            "sha256": original_sha256,
            "size_bytes": len(original_bytes),
        },
        "adapter": {
            "path": ADAPTER_NAME,
            "sha256": sha256_bytes(adapter_bytes),
            "size_bytes": len(adapter_bytes),
        },
        "diff": {
            "path": DIFF_NAME,
            "sha256": sha256_bytes(diff_bytes),
            "size_bytes": len(diff_bytes),
        },
        "activation_scope": {
            "validated_case": "silu",
            "new_keywords_are_unused_by_intern_kernel_body": True,
            "broader_activation_compatibility_claimed": False,
        },
    }
    identity_bytes = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    _write_immutable(adapter_path, adapter_bytes)
    _write_immutable(diff_path, diff_bytes)
    _write_immutable(identity_path, identity_bytes)
    return identity


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    identity = build_adapter(source=args.source, output_dir=args.output_dir)
    print(json.dumps(identity, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
