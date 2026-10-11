"""Tests for the module-table single-arm annotation (R252).

`barrier_best_eV` is empty when one arm has no accepted site; the new
`n_arms_with_barrier` column (2/1/0) plus its caliber line make that shape explicit so a
blank cell is not misread as "no barrier known".
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from recnet_adapter.tools.export_module_table import main as export_main


def _table(tmp_path: Path) -> Path:
    payload = {"channels": [
        {"wave": "v1", "channel": "A* -> B* C*", "qc_status": "qc_pass",
         "barrier_S_best_eV": 0.80, "barrier_M_best_eV": 0.50, "barrier_best_eV": 0.50,
         "ddE_best_M_minus_S_eV": -0.30, "arm_sensitive": False},
        {"wave": "v1", "channel": "D* -> E*", "qc_status": "qc_pass",
         "barrier_S_best_eV": None, "barrier_M_best_eV": 1.20, "barrier_best_eV": None,
         "ddE_best_M_minus_S_eV": None, "arm_sensitive": False},
        {"wave": "v1", "channel": "F* -> G*", "qc_status": "missing",
         "barrier_S_best_eV": None, "barrier_M_best_eV": None, "barrier_best_eV": None},
    ]}
    path = tmp_path / "table.json"
    path.write_text(json.dumps(payload))
    return path


def test_n_arms_with_barrier_marks_single_arm_and_empty_rows(tmp_path: Path):
    rc = export_main(["--table", str(_table(tmp_path)), "--out", str(tmp_path / "out")])
    assert rc == 0
    payload = json.loads((tmp_path / "out" / "module_table.json").read_text())
    by_channel = {row["channel"]: row for row in payload["rows"]}
    assert by_channel["A* -> B* C*"]["n_arms_with_barrier"] == 2
    assert by_channel["D* -> E*"]["n_arms_with_barrier"] == 1
    assert by_channel["D* -> E*"]["arm_of_best"] == "M"
    assert by_channel["D* -> E*"]["barrier_best_eV"] is None
    assert by_channel["F* -> G*"]["n_arms_with_barrier"] == 0
    assert any("n_arms_with_barrier = 2/1/0" in line for line in payload["caliber"])
    rows = list(csv.DictReader(open(tmp_path / "out" / "module_table.csv")))
    assert rows[0]["n_arms_with_barrier"] in {"2", "1", "0"}
