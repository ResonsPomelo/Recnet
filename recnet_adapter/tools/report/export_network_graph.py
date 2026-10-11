#!/usr/bin/env python3
"""Export the explored Fe5C2(510) C2-network as a machine-readable graph.

Inputs
  --prepared   the 42-channel prepared dataset (topology + species names)
  --summary    aggregate_network.py output (per-channel best barriers per arm)
  --table      merged three-wave channel table (network_channels_final.json); replaces
               --prepared/--summary and additionally carries QC status, arm sensitivity
               and the three classes of DFT evidence (added 2026-09-23)
  --anchors    paper_network_table.json (optional; adds DFT/SCF anchor values)
  --out        output prefix; writes <prefix>.json and <prefix>.dot

Graph model: nodes = adsorbate/gas species (name, kind), edges = elementary
channels (reactant -> products) annotated with the best barrier found by each arm
and a QC flag.  Channels that never produced a QC-passing TS are kept but marked
``boundary: true`` with the evidence class (see the report's boundary matrix).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

# channels whose free search never produced a QC-passing saddle (v3/v4 gate)
BOUNDARY = {
    ("CO", "C + O"), ("CCO", "C + CO"), ("CCO", "CC + O"), ("CHCO", "CCH + O"),
    ("CH", "C + H"), ("H2", "H + H"),
}
ANCHOR_HINT = {
    ("CO", "C + O"): ("TS_CO-C+O", "context"),
    ("CCH2", "CCH + H"): ("CCH2+H TS z-", "reverse"),
    ("CH3", "CH2 + H"): ("TS right z+", "context"),
    ("CH4(g)", "CH3 + H"): ("TS", "context"),
}


def _write_graph(out: Path, payload: dict, nodes: dict, edge_lines: list) -> None:
    """Write the machine-readable graph (.json) plus a graphviz file (.dot)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    lines = ["digraph c2_network {", "  rankdir=LR;", '  node [shape=box, fontsize=10];']
    for n in sorted(nodes):
        # attribute lists must live inside brackets, otherwise graphviz rejects the file
        style = " [style=filled, fillcolor=lightgrey]" if nodes[n]["kind"] == "gas" else ""
        lines.append(f'  "{n}"{style};')
    lines += edge_lines
    lines.append("}")
    out.with_suffix(".dot").write_text("\n".join(lines) + "\n")
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.dot')}")
    print("totals:", payload["totals"])


def build_from_table(table_path: Path, out: Path) -> int:
    """Merged three-wave graph: every channel of ``network_channels_final.json``.

    Edge annotations: wave, QC status, per-arm best barriers, arm sensitivity and the
    three classes of DFT evidence (cross-validation pairs / arm arbitration / boundary
    profile).  QC colouring: black = qc_pass, orange = review_only, red = missing;
    edges carrying DFT evidence get penwidth=2.5, arm-sensitive ones are dashed.
    """
    payload_in = json.loads(table_path.read_text())
    rows = payload_in["channels"] if isinstance(payload_in, dict) else payload_in
    nodes, edges = {}, []
    for row in rows:
        label = row["channel"]
        if "->" not in label:
            continue
        reactant, rest = (s.strip() for s in label.split("->", 1))
        products = rest.split()
        for name in [reactant] + products:
            kind = "gas" if name.endswith("(g)") else "adsorbate"
            cur = nodes.get(name)
            if cur is None or (cur["kind"] == "gas" and kind == "adsorbate"):
                nodes[name] = {"name": name, "kind": kind}
        edge = {
            "id": f"{reactant}->{' '.join(products)}",
            "reactant": reactant,
            "products": products,
            "wave": row.get("wave"),
            "qc_status": row.get("qc_status"),
            "boundary": row.get("qc_status") != "qc_pass",
            "barrier_S_eV": row.get("barrier_S_best_eV"),
            "barrier_M_eV": row.get("barrier_M_best_eV"),
            "barrier_best_eV": row.get("barrier_best_eV"),
            "ddE_M_minus_S_eV": row.get("ddE_best_M_minus_S_eV"),
            "arm_sensitive": bool(row.get("arm_sensitive")),
        }
        evidence = {}
        if row.get("dft_pairs"):
            evidence["xval_pairs"] = row.get("dft_pairs")
            evidence["xval_median_delta_eV"] = row.get("dft_median_delta_eV")
        if row.get("arbitrated"):
            evidence["arbitration_closer_arm"] = row.get("dft_closer_arm")
            evidence["arbitration_err_S_eV"] = row.get("arb_err_S_eV")
            evidence["arbitration_err_M_eV"] = row.get("arb_err_M_eV")
        if row.get("dft_profile_arms"):
            evidence["profile_arms"] = row.get("dft_profile_arms")
        if evidence:
            edge["dft_evidence"] = evidence
        edges.append(edge)
    payload = {
        "schema": "ft2dp_c2_network/2",
        "source_table": str(table_path),
        "species": [nodes[n] for n in sorted(nodes)],
        "channels": edges,
        "totals": {
            "species": len(nodes),
            "channels": len(edges),
            "by_wave": {str(w): sum(1 for e in edges if e["wave"] == w)
                        for w in sorted({e["wave"] for e in edges}, key=str)},
            "qc_pass": sum(1 for e in edges if e["qc_status"] == "qc_pass"),
            "qc_review_only": sum(1 for e in edges if e["qc_status"] == "review_only"),
            "qc_missing": sum(1 for e in edges if e["qc_status"] == "missing"),
            "arm_sensitive": sum(1 for e in edges if e["arm_sensitive"]),
            "channels_with_xval": sum(1 for e in edges if "xval_pairs" in e.get("dft_evidence", {})),
            "channels_with_arbitration": sum(
                1 for e in edges if "arbitration_closer_arm" in e.get("dft_evidence", {})),
            "channels_with_profile": sum(1 for e in edges if "profile_arms" in e.get("dft_evidence", {})),
        },
    }
    color = {"qc_pass": "black", "review_only": "orange", "missing": "red"}
    edge_lines = []
    for e in edges:
        if e["barrier_S_eV"] is not None or e["barrier_M_eV"] is not None:
            bs = "?" if e["barrier_S_eV"] is None else f'{e["barrier_S_eV"]:.3f}'
            bm = "?" if e["barrier_M_eV"] is None else f'{e["barrier_M_eV"]:.3f}'
            lbl = f"S {bs} / M {bm}"
        else:
            lbl = e["qc_status"] or "no-barrier"
        attrs = [f'label="{lbl}"', f'color={color.get(e["qc_status"], "black")}']
        if e.get("dft_evidence"):
            attrs.append("penwidth=2.5")
        if e["arm_sensitive"]:
            attrs.append("style=dashed")
        attr = ", ".join(attrs)
        prod = ", ".join(f'"{p}"' for p in e["products"])
        if len(e["products"]) == 1:
            edge_lines.append(f'  "{e["reactant"]}" -> {prod} [{attr}];')
        else:
            for k, p in enumerate(e["products"]):
                a = attr if k == 0 else ", ".join(x for x in attrs if not x.startswith("label="))
                edge_lines.append(f'  "{e["reactant"]}" -> "{p}" [{a}];')
    _write_graph(out, payload, nodes, edge_lines)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", help="prepared dataset (single wave); use with --summary")
    ap.add_argument("--summary", help="aggregate_network.py output (single wave)")
    ap.add_argument("--table", help="merged three-wave channel table "
                                    "(network_channels_final.json); alternative to --prepared/--summary")
    ap.add_argument("--anchors", default=None)
    ap.add_argument("--gate", default=None, help="c2_gate.json; its missing-channel list is authoritative for qc_pass/boundary")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.table:
        return build_from_table(Path(args.table), Path(args.out))
    if not (args.prepared and args.summary):
        ap.error("either --table, or both --prepared and --summary, are required")

    gate_missing = set()
    if args.gate and Path(args.gate).exists():
        g = json.loads(Path(args.gate).read_text())
        for x in g.get("missing_qc_pass_channels", []):
            gate_missing.add((x.get("reactant", "").rstrip("*"), " + ".join(x.get("product", "").replace("*", "").split())))
    prep = yaml.safe_load(Path(args.prepared).read_text())
    species = prep["species"]
    summ = json.loads(Path(args.summary).read_text())
    anchor_rows = {}
    if args.anchors and Path(args.anchors).exists():
        for r in json.loads(Path(args.anchors).read_text())["rows"]:
            anchor_rows.setdefault(r["step"], r)

    # per-channel best barrier per arm, from the aggregated summary
    per_channel = {}
    for ch_label, per_case in summ["channels"].items():
        for case, st in per_case.items():
            arm = "S" if case.startswith("S") else "M"
            b = st.get("barrier_best")
            if b is None:
                continue
            cur = per_channel.setdefault(ch_label, {})
            if arm not in cur or b < cur[arm]:
                cur[arm] = b
    # map the aggregate's channel keys (reactant strings from the prepared dataset,
    # e.g. "CH3*", "CH4(g)") onto bare species names (e.g. "CH3", "CH4")
    key_by_species = {}
    for k in per_channel:
        bare = k[:-1] if k.endswith("*") else k
        if bare.endswith("(g)"):
            bare = bare[:-3]
        key_by_species.setdefault(bare, k)

    def best(rname, arm):
        k = key_by_species.get(rname)
        if k is None:
            return None
        return per_channel.get(k, {}).get(arm)

    nodes, edges = {}, []
    for i, rxn in enumerate(prep["rxns"]):
        rname = species[rxn["reactant_species"][0]]["name"]
        pnames = [species[p]["name"] for p in rxn["product_species"]]
        pstr = " + ".join(pnames)
        is_gas = not species[rxn["reactant_species"][0]]["ad_idx"]
        for n in [rname] + pnames:
            nodes.setdefault(n, {"name": n, "kind": "gas" if is_gas and n == rname else "adsorbate"})
        a_s, a_m = best(rname, "S"), best(rname, "M")
        gate_key = (rname, pstr)
        in_gate_missing = gate_key in gate_missing or (gate_missing and any(
            g[0] == rname and g[1] == pstr for g in gate_missing))
        edge = {
            "id": f"{rname}->{pstr}",
            "reactant": rname,
            "products": pnames,
            "broken_bond": rxn["broken_bond"],
            "barrier_S_eV": None if a_s is None else round(a_s, 3),
            "barrier_M_eV": None if a_m is None else round(a_m, 3),
            "has_barrier_row": (a_s is not None) or (a_m is not None),
            "qc_pass": (not in_gate_missing) and ((a_s is not None) or (a_m is not None)),
            "boundary": in_gate_missing or (rname, pstr) in BOUNDARY,
        }
        hint = ANCHOR_HINT.get((rname, pstr))
        if hint and hint[0] in anchor_rows:
            row = anchor_rows[hint[0]]
            for k in ("dE_DFT_eV", "dE_SCF_eV", "dE_DPA2_eV"):
                if k in row:
                    edge["paper_anchor_eV"] = row[k]
                    edge["paper_anchor_kind"] = k.split("_")[1]
                    edge["paper_relation"] = hint[1]
                    break
        edges.append(edge)

    payload = {
        "schema": "ft2dp_c2_network/1",
        "species": list(nodes.values()),
        "channels": edges,
        "totals": {
            "species": len(nodes),
            "channels": len(edges),
            "channels_with_barrier": sum(1 for e in edges if e["qc_pass"]),
            "boundary_channels": sum(1 for e in edges if e["boundary"]),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    lines = ["digraph c2_network {", '  rankdir=LR;', '  node [shape=box, fontsize=10];']
    for n in sorted(nodes):
        style = " [style=filled, fillcolor=lightgrey]" if nodes[n]["kind"] == "gas" else ""
        lines.append(f'  "{n}"{style};')
    for e in edges:
        lbl = f'{e["barrier_S_eV"]}/{e["barrier_M_eV"]}' if e["qc_pass"] else "boundary"
        color = "black" if e["qc_pass"] else "red"
        prod = ", ".join(f'"{p}"' for p in e["products"])
        if len(e["products"]) == 1:
            lines.append(f'  "{e["reactant"]}" -> {prod} [label="{lbl}", color={color}];')
        else:
            # one edge per product so graphviz stays simple (barrier shown once)
            for k, p in enumerate(e["products"]):
                lab = lbl if k == 0 else ""
                lines.append(f'  "{e["reactant"]}" -> "{p}" [label="{lab}", color={color}];')
    lines.append("}")
    out.with_suffix(".dot").write_text("\n".join(lines) + "\n")
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.dot')}")
    print("totals:", payload["totals"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
