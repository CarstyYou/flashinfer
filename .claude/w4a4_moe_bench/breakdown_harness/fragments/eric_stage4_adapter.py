"""Build the reviewed Eric stage-4 constructor compatibility fragment.

The caller owns source identity, output location, and experiment schema.  This
module owns only the validated three-keyword source transformation.
"""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any


ADAPTER_NAME = "moe_dynamic_kernel.py"
DIFF_NAME = "adapter.diff"
IDENTITY_NAME = "identity.json"

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


def validate_adapter(
    source_text: str,
    adapter_text: str,
    *,
    source_name: str = "source.py",
) -> dict[str, Any]:
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
            fromfile=source_name,
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
    source: Path,
    output_dir: Path,
    expected_original_sha256: str,
    identity_schema: str,
) -> dict[str, Any]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not source.is_file():
        raise AdapterError(f"missing Eric source: {source}")
    original_bytes = source.read_bytes()
    original_sha256 = sha256_bytes(original_bytes)
    if original_sha256 != expected_original_sha256:
        raise AdapterError(
            f"Eric source SHA-256 drift: {original_sha256} != {expected_original_sha256}"
        )
    try:
        source_text = original_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AdapterError("Eric source is not UTF-8") from error

    adapter_text = make_adapter(source_text)
    validation = validate_adapter(source_text, adapter_text, source_name=source.name)
    diff_text = str(validation.pop("unified_diff"))
    adapter_bytes = adapter_text.encode("utf-8")
    diff_bytes = diff_text.encode("utf-8")
    adapter_path = output_dir / ADAPTER_NAME
    diff_path = output_dir / DIFF_NAME
    identity_path = output_dir / IDENTITY_NAME

    identity = {
        "schema": identity_schema,
        "transformer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
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
