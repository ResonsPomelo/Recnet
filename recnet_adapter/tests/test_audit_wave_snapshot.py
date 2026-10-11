"""`audit_wave_snapshot.py` 的契约测试（R353）：用合成交付表验证四类断言。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "audit_wave_snapshot.py"


def _write(tmp: Path, name: str, rows: list) -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "network_channels_final.json").write_text(json.dumps(rows))
    return d


def _row(ch: str, barrier: float) -> dict:
    return {"channel": ch, "barrier_best_eV": barrier, "qc_status": "qc_pass"}


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)


def test_pass_when_baseline_preserved_and_delta_expected(tmp_path: Path) -> None:
    base = _write(tmp_path, "base", [_row(f"ch{i}", 1.0 + i) for i in range(5)])
    new = _write(tmp_path, "new", [_row(f"ch{i}", 1.0 + i) for i in range(5)] + [_row("c3w1_a", 0.5)])
    p = _run("--snapshot", str(new), "--baseline", str(base / "network_channels_final.json"),
             "--expected-new", "1", "--expected-total", "6")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "全部通过" in p.stdout


def test_fail_when_baseline_channel_missing(tmp_path: Path) -> None:
    base = _write(tmp_path, "base", [_row("ch0", 1.0), _row("ch1", 2.0)])
    new = _write(tmp_path, "new", [_row("ch0", 1.0)])
    p = _run("--snapshot", str(new), "--baseline", str(base / "network_channels_final.json"))
    assert p.returncode == 1
    assert "baseline_channels_present" in p.stdout and "FAIL" in p.stdout


def test_fail_when_baseline_barrier_drifted(tmp_path: Path) -> None:
    base = _write(tmp_path, "base", [_row("ch0", 1.0)])
    new = _write(tmp_path, "new", [_row("ch0", 1.3)])
    p = _run("--snapshot", str(new), "--baseline", str(base / "network_channels_final.json"))
    assert p.returncode == 1
    assert "baseline_barriers_unchanged" in p.stdout


def test_fail_when_delta_mismatch(tmp_path: Path) -> None:
    base = _write(tmp_path, "base", [_row("ch0", 1.0)])
    new = _write(tmp_path, "new", [_row("ch0", 1.0), _row("x", 0.1), _row("y", 0.2)])
    p = _run("--snapshot", str(new), "--baseline", str(base / "network_channels_final.json"),
             "--expected-new", "1")
    assert p.returncode == 1
    assert "expected_new_channels" in p.stdout


def test_input_missing_is_rc2(tmp_path: Path) -> None:
    p = _run("--snapshot", str(tmp_path / "nope"), "--baseline", str(tmp_path / "nope.json"))
    assert p.returncode == 2
