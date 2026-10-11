"""Tests for the scission-rehearsal verdict filter (R249).

Guards two properties: the unfiltered rehearsal still reproduces the recorded wave-6 ceiling
(142 species / 538 channels / +244 vs baseline — R225), and the opt-in ``--verdicts`` filter
shrinks the run while recording what it skipped (review material for the ①b decision).
"""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from recnet_adapter.tools.rehearse_scission_closure import rehearse


def test_unfiltered_rehearsal_reproduces_the_recorded_ceiling():
    report = rehearse(max_rounds=6)
    assert report["verdict_filter"] is None
    assert report["skipped_by_verdict"] == {}
    assert report["baseline"]["channels"] == 294
    assert report["final"] == {
        "species_entries": 142, "channels": 538, "dropped_cuts": 0, "channels_vs_baseline": 244,
    }


def test_zwitterion_only_filter_shrinks_the_rehearsal_and_records_skips():
    report = rehearse(max_rounds=6, verdicts={"zwitterionic"})
    assert report["verdict_filter"] == ["zwitterionic"]
    assert report["final"]["channels"] < 538
    assert report["final"]["dropped_cuts"] > 0
    assert report["skipped_by_verdict"].get("radical_or_open_shell", 0) > 0
