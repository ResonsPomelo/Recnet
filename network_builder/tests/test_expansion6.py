"""C3 扩网（expansion6）的契约测试（R309，2026-09-26）。

契约：
1. **默认关闭**：不传 flag 时物种表仍是 frozen 29（既有字节一致测试另测）；
2. **档位机械可复算**：四档物种数 = 45 / 85 / 136 / 241，且与审计 JSON 的判据一一对应；
3. **实建通道数**：closed-shell 档 390（+91）、all 档 1371（+1072）——回归护栏；
4. 每档物种都能 materialise（公式/锚点断言不炸）；
5. 未知档名**显式报错**，不静默回退；
6. `make_expansion_c3.py --check` 与已提交的 `expansion6.py` 一致（可复算性护栏；审计件缺失则 skip）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from network_builder.build import build_species_table
from network_builder.channels import enumerate_channels

REPO = Path(__file__).resolve().parents[2]          # Recnet/
BASE = dict(expansion=True, expansion2=True, expansion3=True, expansion4=True, expansion5=True)
TIER_SPECIES = {"closed-shell": 45, "o1": 85, "ready2": 136, "all": 241}


def _table(tier: str):
    return build_species_table(expansion6=True, expansion6_tier=tier, **BASE)


def test_default_off_keeps_frozen_table() -> None:
    species = build_species_table()
    assert len(species) == 29
    assert len(build_species_table(**BASE)) == 97


@pytest.mark.parametrize("tier,expected", sorted(TIER_SPECIES.items()))
def test_tier_species_counts_and_materialisation(tier: str, expected: int) -> None:
    species = _table(tier)
    assert len(species) == 97 + expected
    # 基线表本身含 2 个"同分子多条目"（gas + 吸附条目，97 条目 / 95 身份）；新增部分必须身份互异
    added = species[97:]
    assert len({s.identity for s in added}) == len(added)
    baseline_identities = {s.identity for s in build_species_table(**BASE)}
    assert not ({s.identity for s in added} & baseline_identities)
    keys = [s.key for s in species]
    assert len(keys) == len(set(keys))
    for s in added:
        assert s.source == "closure-c3-gap"
        assert s.natoms >= 3


def test_tiers_are_distinct_selection_criteria_not_nested() -> None:
    """closed-shell 与 o1 是**不同判据**（交集 12），因此不是包含关系——写下来防止误用。"""
    closed = {s.identity for s in _table("closed-shell")[97:]}
    o1 = {s.identity for s in _table("o1")[97:]}
    every = {s.identity for s in _table("all")[97:]}
    assert len(closed) == 45 and len(o1) == 85 and len(every) == 241
    assert len(closed & o1) == 12
    assert not closed <= o1
    assert closed <= every          # all 是全量，必含各档


def test_enumerated_channel_counts_regression() -> None:
    baseline = len(enumerate_channels(build_species_table(**BASE))[0])
    assert baseline == 299
    assert len(enumerate_channels(_table("closed-shell"))[0]) == 390
    assert len(enumerate_channels(_table("all"))[0]) == 1371


def test_unknown_tier_raises() -> None:
    with pytest.raises(ValueError):
        build_species_table(expansion6=True, expansion6_tier="nope", **BASE)


def test_generator_check_matches_committed_module() -> None:
    audit = REPO / "network_inputs" / "closure_audit-c3" / "closure_audit.json"
    tool = REPO / "recnet_adapter" / "tools" / "make_expansion_c3.py"
    if not (audit.exists() and tool.exists()):
        pytest.skip("审计件/生成器缺失（仅历史快照环境）")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    proc = subprocess.run(
        [sys.executable, str(tool), "--audit", str(audit), "--out",
         str(REPO / "network_builder" / "expansion6.py"), "--check"],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
