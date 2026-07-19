import json

import pytest

from build_adapter import (
    ADAPTER_NAME,
    DEFAULT_SOURCE,
    DIFF_NAME,
    EXPECTED_ORIGINAL_SHA256,
    IDENTITY_NAME,
    AdapterError,
    build_adapter,
    sha256_file,
    validate_adapter,
)


def test_adapter_is_only_three_constructor_keywords(tmp_path):
    original_hash = sha256_file(DEFAULT_SOURCE)
    identity = build_adapter(source=DEFAULT_SOURCE, output_dir=tmp_path)

    adapter = tmp_path / ADAPTER_NAME
    diff = tmp_path / DIFF_NAME
    identity_path = tmp_path / IDENTITY_NAME
    assert original_hash == EXPECTED_ORIGINAL_SHA256
    assert sha256_file(DEFAULT_SOURCE) == original_hash
    assert adapter.is_file() and diff.is_file() and identity_path.is_file()
    assert sha256_file(adapter) == identity["adapter"]["sha256"]
    assert sha256_file(diff) == identity["diff"]["sha256"]
    assert json.loads(identity_path.read_text()) == identity

    validation = validate_adapter(DEFAULT_SOURCE.read_text(), adapter.read_text())
    assert validation["inserted_keywords"] == [
        "swiglu_alpha",
        "swiglu_beta",
        "swiglu_limit",
    ]
    assert validation["ast_equal_after_removing_keywords"] is True
    assert validation["unified_diff_additions"] == 3
    assert validation["unified_diff_deletions"] == 0

    # Rebuilding an identical immutable overlay is allowed and deterministic.
    assert build_adapter(source=DEFAULT_SOURCE, output_dir=tmp_path) == identity


def test_adapter_refuses_immutable_output_drift(tmp_path):
    build_adapter(source=DEFAULT_SOURCE, output_dir=tmp_path)
    (tmp_path / ADAPTER_NAME).write_text("drift\n")
    with pytest.raises(AdapterError, match="immutable artifact drift"):
        build_adapter(source=DEFAULT_SOURCE, output_dir=tmp_path)


def test_adapter_refuses_source_hash_drift(tmp_path):
    source = tmp_path / DEFAULT_SOURCE.name
    source.write_bytes(DEFAULT_SOURCE.read_bytes() + b"\n")
    with pytest.raises(AdapterError, match="source SHA-256 drift"):
        build_adapter(source=source, output_dir=tmp_path / "out")
