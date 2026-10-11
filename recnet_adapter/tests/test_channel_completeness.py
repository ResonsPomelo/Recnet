"""Tests for the channel-level completeness audit (R222).

The audit is the machine-checked completeness certificate: the delivered table must be a
subset of the generator's full enumeration, and the only allowed gap is the in-flight wave's
expected set.
"""
from __future__ import annotations

import json

import yaml

from recnet_adapter.tools.audit_channel_completeness import main as audit_main


def _dataset(path, labels):
    rxns = []
    for reactant, product in labels:
        rxns.append({"reactant": reactant, "product": product, "broken_bond": [0, 1],
                     "reactant_species": ["sp_000"], "product_species": ["sp_001", "sp_002"]})
    path.write_text(yaml.safe_dump({"rxns": rxns}), encoding="utf-8")


def _table(path, labels):
    path.write_text(json.dumps({"channels": [{"channel": c} for c in labels]}), encoding="utf-8")


D = [("CH3*", "CH2* H*"), ("CH3*", "CH* H2*"), ("CH2*", "CH* H*")]
E = [("CH3O*", "CH3* O*")]


def test_strict_pass_when_table_equals_full(tmp_path):
    full, table = tmp_path / "full.yaml", tmp_path / "table.json"
    _dataset(full, D)
    _table(table, [f"{r} -> {p}" for r, p in D])
    assert audit_main(["--table", str(table), "--full", str(full)]) == 0


def test_pending_matches_expected_diff(tmp_path):
    full, table, expect = tmp_path / "full.yaml", tmp_path / "table.json", tmp_path / "expect.yaml"
    _dataset(full, D + E)
    _table(table, [f"{r} -> {p}" for r, p in D])
    _dataset(expect, E)
    assert audit_main(["--table", str(table), "--full", str(full),
                       "--expect-diff-from", str(expect)]) == 0


def test_pending_without_expectation_fails(tmp_path):
    full, table = tmp_path / "full.yaml", tmp_path / "table.json"
    _dataset(full, D + E)
    _table(table, [f"{r} -> {p}" for r, p in D])
    assert audit_main(["--table", str(table), "--full", str(full)]) == 2


def test_phantom_channel_fails(tmp_path):
    full, table = tmp_path / "full.yaml", tmp_path / "table.json"
    _dataset(full, D)
    _table(table, [f"{r} -> {p}" for r, p in D] + ["PHANTOM* -> X* Y*"])
    assert audit_main(["--table", str(table), "--full", str(full)]) == 2


def test_json_report_records_duplicates(tmp_path):
    full, table, out = tmp_path / "full.yaml", tmp_path / "table.json", tmp_path / "audit.json"
    _dataset(full, D + [D[0]])  # duplicate row (e.g. gas reference sharing a label)
    _table(table, [f"{r} -> {p}" for r, p in D])
    assert audit_main(["--table", str(table), "--full", str(full), "--json-out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["full"]["rows"] == len(D) + 1
    assert report["full"]["unique"] == len(D)
    assert report["full"]["duplicate_labels"] == {"CH3* -> CH2* H*": 2}
    assert report["pass"] is True
