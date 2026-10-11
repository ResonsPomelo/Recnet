#!/usr/bin/env python3
"""Model-vs-DFT barrier table for the closure-expanded network (2026-09-22).

Consumes:

* ``abacus_xval_summary.json`` from ``build_abacus_cases_from_campaign.py``
  (selected IS/TS pairs with the model barrier ``barrier_ts_minus_is``)
* ``single_points/<label>_{IS,TS}/meta.json`` from the ABACUS batch

and writes ``campaign_xval_table.{json,md}`` with, per pair, the model barrier,
the DFT barrier ``E(TS) - E(IS)``, their difference and both SCF states.
This is the extended-network analogue of the frozen-network ABACUS check.

Usage:
  summarize_campaign_xval.py --workdir . --summary abacus_xval_summary.json \\
      --out-dir c2r2-xval-report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_meta(workdir: Path, label: str):
    path = workdir / "single_points" / label / "meta.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Extended-network model vs DFT barriers")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    selected = json.loads(Path(args.summary).read_text()).get("selected", [])

    rows = []
    for entry in selected:
        label = entry["label"]
        is_meta = read_meta(workdir, f"{label}_IS")
        ts_meta = read_meta(workdir, f"{label}_TS")

        def status(meta):
            if meta is None:
                return "missing"
            if meta.get("error"):
                return "error"
            return str(meta.get("scf_converged"))

        row = {
            "label": label,
            "case": entry["case"],
            "arm": entry["arm"],
            "species": entry["species"],
            "rxn_idx": entry["rxn_idx"],
            "site": entry["site"],
            "model_barrier_eV": entry["model_barrier_eV"],
            "e_is_eV": None if not is_meta else is_meta.get("energy_eV"),
            "e_ts_eV": None if not ts_meta else ts_meta.get("energy_eV"),
            "status_is": status(is_meta),
            "status_ts": status(ts_meta),
        }
        if row["e_is_eV"] is not None and row["e_ts_eV"] is not None:
            row["dfT_barrier_eV"] = round(row["e_ts_eV"] - row["e_is_eV"], 4)
            row["delta_eV"] = round(row["dfT_barrier_eV"] - row["model_barrier_eV"], 4)
        rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "campaign_xval_table.json").write_text(
        json.dumps({"schema": "ft2dp_campaign_xval_table/1", "rows": rows},
                   ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    lines = ["# 扩展网络：模型势垒 vs ABACUS 单点势垒（IS → TS）", "",
             "| 物种 | 臂 | case | rxn | site | 模型势垒 (eV) | DFT 势垒 (eV) | Δ (DFT−模型) | SCF(IS/TS) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        model = f"{row['model_barrier_eV']:.3f}"
        dfT = "-" if row.get("dfT_barrier_eV") is None else f"{row['dfT_barrier_eV']:.3f}"
        delta = "-" if row.get("delta_eV") is None else f"{row['delta_eV']:+.3f}"
        lines.append(f"| {row['species']} | {row['arm']} | {row['case']} | {row['rxn_idx']} | "
                     f"{row['site']} | {model} | {dfT} | {delta} | "
                     f"{row['status_is']}/{row['status_ts']} |")
    done = [r for r in rows if r.get("dfT_barrier_eV") is not None]
    if done:
        deltas = [r["delta_eV"] for r in done]
        lines += ["", f"完成 {len(done)}/{len(rows)} 对；Δ 统计：min {min(deltas):+.3f} / "
                      f"max {max(deltas):+.3f} eV"]
    else:
        lines += ["", f"完成 0/{len(rows)} 对（等待 ABACUS 作业产出 meta.json）。"]
    (out_dir / "campaign_xval_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_dir}/campaign_xval_table.{{json,md}} ({len(done)}/{len(rows)} pairs)")
    for row in done:
        print(f"  {row['species']:8s} {row['arm']} rxn{row['rxn_idx']:<3} "
              f"model {row['model_barrier_eV']:.3f} -> DFT {row['dfT_barrier_eV']:.3f} "
              f"(Δ{row['delta_eV']:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
