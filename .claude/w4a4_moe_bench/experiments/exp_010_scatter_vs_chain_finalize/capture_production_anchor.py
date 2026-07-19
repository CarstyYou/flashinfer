#!/usr/bin/env python3
"""Run exp_006's audited production capture while retaining exp_010 replay inputs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
EXP006 = ROOT.parent / "exp_006_fc2_completion_anchored_breakdown"
EXP004 = ROOT.parent / "exp_004_fused_phase_timing_breakdown"
for path in (EXP006, EXP004):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import capture_completion_timing as capture  # noqa: E402


_original_loader = capture._load_gpu_modules
_original_snapshot = capture._snapshot


def _patched_loader():
    torch, worker = _original_loader()
    if getattr(worker, "_exp010_snapshot_patch", False):
        return torch, worker
    original_workspace_snapshot = worker.workspace_snapshot

    def workspace_snapshot(wrapper: Any, fixture: Any):
        tensors, payload = original_workspace_snapshot(wrapper, fixture)
        workspace = wrapper._dynamic_workspace
        tensors = dict(tensors)
        tensors.update(
            {
                "token_map": workspace.token_map.detach().cpu().clone(),
                "token_weights": workspace.token_weights.detach().cpu().clone(),
                "topk_ids": fixture.topk_ids.detach().cpu().clone(),
                "topk_weights": fixture.topk_weights.detach().cpu().clone(),
            }
        )
        return tensors, payload

    worker.workspace_snapshot = workspace_snapshot
    worker._exp010_snapshot_patch = True
    return torch, worker


def _patched_snapshot(arm: Any, workspace_tensors: Mapping[str, Any]):
    timing, descriptors, digest = _original_snapshot(arm, workspace_tensors)
    timing.update(
        {
            name: workspace_tensors[name]
            for name in ("token_map", "token_weights", "topk_ids", "topk_weights")
        }
    )
    return timing, descriptors, digest


capture._load_gpu_modules = _patched_loader
capture._snapshot = _patched_snapshot


if __name__ == "__main__":
    raise SystemExit(capture.main())
