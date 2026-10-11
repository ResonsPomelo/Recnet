"""Tests for the network-level analysis tool (R179/R180).

The tool is embedded in both snapshot chains, so its contract is fixed here: composition counts,
bond-class attribution through the prepared dataset, gas exclusion from the kinetic-trap list and
graceful behaviour on a table without barriers.
"""
from __future__ import annotations

import json

import yaml

from recnet_adapter.tools.analyze_network_table import main as analyze_main


def _template(path, symbols):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(symbols)), ""]
    for i, s in enumerate(symbols):
        lines.append(f"{s} 0.0 {i * 1.0:.2f} 0.0")
    path.write_text("\n".join(lines) + "\n")


def _dataset(tmp_path):
    base = tmp_path / "ds"
    tpl = base / "ads_templates"
    _template(tpl / "sp_000.xyz", ["C", "H", "H", "H"])    # CH3
    _template(tpl / "sp_001.xyz", ["C", "H", "H"])          # CH2
    _template(tpl / "sp_002.xyz", ["H"])                    # H
    _template(tpl / "sp_003.xyz", ["H", "H"])               # H2 (H-H bond)
    payload = {
        "species": {
            "sp_000": {"name": "CH3", "adjlist": "", "template_xyz": "ads_templates/sp_000.xyz",
                       "ad_idx": [0]},
            "sp_001": {"name": "CH2", "adjlist": "", "template_xyz": "ads_templates/sp_001.xyz",
                       "ad_idx": [0]},
            "sp_002": {"name": "H", "adjlist": "", "template_xyz": "ads_templates/sp_002.xyz",
                       "ad_idx": [0]},
            "sp_003": {"name": "H2(g)", "adjlist": "", "template_xyz": "ads_templates/sp_003.xyz",
                       "ad_idx": []},
        },
        "rxns": [
            {"reactant": "CH3*", "product": "CH2* H*", "broken_bond": [0, 1],
             "reactant_species": ["sp_000"], "product_species": ["sp_001", "sp_002"]},
            {"reactant": "H2(g)", "product": "H* H*", "broken_bond": [0, 1],
             "reactant_species": ["sp_003"], "product_species": ["sp_002", "sp_002"]},
        ],
    }
    yml = base / "prepared_rmg_data.yaml"
    yml.write_text(yaml.safe_dump(payload, allow_unicode=True))
    return yml


def _run(tmp_path, rows):
    ds = _dataset(tmp_path)
    table = tmp_path / "network_channels_final.json"
    table.write_text(json.dumps({"channels": rows}))
    anchors = tmp_path / "anchors.json"
    anchors.write_text(json.dumps({"rows": []}))
    out = tmp_path / "out"
    rc = analyze_main(["--table", str(table), "--out-dir", str(out),
                       "--dataset", str(ds), "--anchors", str(anchors)])
    assert rc == 0
    return json.loads((out / "network_analysis.json").read_text()), (out / "network_analysis.md")


def test_composition_classes_and_gas_exclusion(tmp_path):
    rows = [
        {"channel": "CH3* -> CH2* H*", "wave": "c2r5", "qc_status": "qc_pass",
         "barrier_S_best_eV": 0.80, "barrier_M_best_eV": 1.60, "barrier_best_eV": 0.80,
         "ddE_best_M_minus_S_eV": 0.80, "arm_sensitive": True},
        {"channel": "H2(g) -> H* H*", "wave": "v1", "qc_status": "missing"},
    ]
    rep, md = _run(tmp_path, rows)
    assert rep["channels"] == 2
    assert rep["waves"] == {"c2r5": 1, "v1": 1}
    assert rep["qc"] == {"qc_pass": 1, "missing": 1}
    # bond classes come from the templates, not from element counting
    assert rep["classes"]["C-H"]["n"] == 1
    assert rep["classes"]["H-H"]["n"] == 1
    assert rep["unclassified_channels"] == 0
    # gases never appear as kinetic traps
    trap_names = [s for s, _ in rep["kinetic_traps"]]
    assert "H2(g)" not in trap_names
    assert trap_names == ["CH3*"]
    # the arm-sensitive channel without DFT evidence is counted
    assert rep["arm_sensitive"] == 1 and rep["arm_sensitive_without_dft"] == 1
    # divergence binning uses |Δ(S−M)|
    assert rep["arm_divergence_bins"]["0.6-1.0"] == 1
    assert md.exists() and "动力学瓶颈" in md.read_text()


def test_table_without_barriers_is_handled(tmp_path):
    rows = [{"channel": "CH3* -> CH2* H*", "wave": "c2r5", "qc_status": "missing"}]
    rep, _ = _run(tmp_path, rows)
    assert rep["barriers"]["best"] == {}
    assert rep["kinetic_traps"] == []
    assert rep["downhill_le_0.05eV"] == 0


def test_bep_section_needs_thermo_and_uses_same_site_pairs(tmp_path):
    """§8 joins the thermo side table on channel and fits Ea = a + b·ΔE per class (R187)."""
    import json as _json

    rows = [
        {"channel": "CH3* -> CH2* H*", "wave": "c2r5", "qc_status": "qc_pass",
         "barrier_S_best_eV": 0.8, "barrier_M_best_eV": 1.6, "barrier_best_eV": 0.8,
         "ddE_best_M_minus_S_eV": 0.8, "arm_sensitive": True},
        {"channel": "H2(g) -> H* H*", "wave": "v1", "qc_status": "qc_pass",
         "barrier_S_best_eV": 0.2, "barrier_M_best_eV": 0.2, "barrier_best_eV": 0.2},
    ]
    ds = _dataset(tmp_path)
    table = tmp_path / "network_channels_final.json"
    table.write_text(_json.dumps({"channels": rows}))
    anchors = tmp_path / "anchors.json"
    anchors.write_text(_json.dumps({"rows": []}))
    thermo = tmp_path / "thermo.json"
    # four (ΔE, Ea) pairs of the SAME class (C-H) so the fit is defined
    thermo.write_text(_json.dumps({"rows": [
        {"channel": "CH3* -> CH2* H*", "delta_e_eV": de, "barrier_eV": ea}
        for de, ea in ((-1.0, 0.5), (-0.5, 0.8), (0.0, 1.1), (0.5, 1.4))]}))
    out = tmp_path / "out"
    from recnet_adapter.tools.analyze_network_table import main as analyze_main
    assert analyze_main(["--table", str(table), "--out-dir", str(out), "--dataset", str(ds),
                         "--anchors", str(anchors), "--thermo", str(thermo)]) == 0
    bep = _json.loads((out / "network_analysis.json").read_text())["bep"]
    assert bep["C-H"]["n"] == 4
    # Ea = a + b·ΔE with b = 0.6 through (mean ΔE = −0.25, mean Ea = 0.95) ⇒ a = 1.1
    assert bep["C-H"]["slope"] == 0.6 and bep["C-H"]["intercept"] == 1.1
    assert bep["C-H"]["pearson_r"] == 1.0
    assert "## 8." in (out / "network_analysis.md").read_text()
