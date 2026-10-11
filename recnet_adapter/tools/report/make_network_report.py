#!/usr/bin/env python3
"""Generate the Fe5C2(510) C2-network report from aggregator JSON + paper anchors.

Inputs
  --summary   network_summary.json produced by aggregate_network.py
  --anchors   paper_network_table.json produced by validation-pipeline/reaction/paper_network_table.py
  --out       markdown path

The channel -> paper-step mapping below is the chemistry judgement used for the
comparison table (``direct`` = same bond change; ``context`` = same bond change but
the paper evaluated it in a different co-adsorbate environment; ``none`` = new
channel not present in the paper's table).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# our reactant label -> (paper step label, kind)
MAPPING = {
    "CO*":   ("TS_CO-C+O", "context"),      # paper: CH+CO+H -> CH+CCv+O+H
    "CH*":   ("TS_C-CH", "context"),        # paper: C+CH -> CCH (reverse, C-C not C-H)
    "CH2*":  ("TS_near1 / TS_near2", "context"),
    "CH3*":  ("TS right z+", "context"),    # paper: CH3 -> CH2 + H at its site
    # CCH* channels here are C-C / C-H scissions; the paper's "CCH+H TS" row is the
    # *association* CCH+H->CCH2 (i.e. our CCH2* -> CCH* + H* in reverse).  Keep the
    # scissions unmapped so they are not compared against a different reaction.
    "CCH*":  (None, "none"),
    "CCH2*": ("CCH2+H TS z-", "reverse"),
    "CCH3*": ("TS", "context"),             # paper: CCH3 splitting rows
    "CHCH*": ("TS left", "context"),
    "CHCH2*": ("TS", "context"),
    "CH2CH2*": ("TS", "context"),
    "CH2CH3*": ("TS", "context"),
    "CC*":   ("TS_C-CH", "context"),
    "HCO*":  ("TS_CO-C+O", "context"),
    "H2CO*": ("TS", "context"),
    "H3CO*": ("TS", "context"),
    "COH*":  ("TS_CO-C+O", "context"),
    "OH*":   (None, "none"),
    "H2(g)": (None, "none"),
    "H2O(g)": (None, "none"),
    "CH4(g)": ("TS", "context"),
    "C2H6(g)": (None, "none"),
}


def load(path):
    return json.loads(Path(path).read_text())


def anchor_index(anchors):
    idx = defaultdict(list)
    for row in anchors["rows"]:
        idx[row["step"]].append(row)
    return idx


def best_anchor(idx, label):
    """Return the most representative anchor row for a paper step label."""
    rows = idx.get(label) or []
    with_dft = [r for r in rows if "dE_DFT_eV" in r]
    pick = (with_dft or rows)
    return pick[0] if pick else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Fe5C2(510) C2 及以下反应网络探索报告")
    args = ap.parse_args()

    summary = load(args.summary)
    anchors = load(args.anchors)
    idx = anchor_index(anchors)

    channels = summary["channels"]
    comparison = {c["channel"]: c for c in summary.get("arm_comparison", [])}

    lines = [f"# {args.title}", "",
             f"- cases: {summary['totals']['n_cases']}；通道: {summary['totals']['n_channels']}；"
             f"接受 TS: {summary['totals']['n_ts_records']}；可匹配能量行: {summary['totals']['n_rows_accepted']}",
             "", "## 1. case 状态", "",
             "| case | model | head | device | exit | failure_class | TS records | rows accepted |",
             "|---|---|---|---|---|---|---|---|"]
    for name in sorted(summary["cases"]):
        v = summary["cases"][name]
        lines.append(f"| {name} | {(v.get('model') or '').split('/')[-1]} | {v.get('head') or ''} | "
                     f"{v.get('device') or ''} | {v.get('exit_code') or ''} | {v.get('failure_class') or ''} | "
                     f"{v.get('n_ts_records')} | {v.get('n_rows_accepted')} |")

    lines += ["", "## 2. 通道势垒（电子能口径，eV）", "",
              "| 通道 | case | rows | 接受行 | best | mean | 论文锚点 | 锚值(eV) | 关系 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for ch in sorted(channels):
        paper_label, kind = MAPPING.get(ch, (None, "none"))
        row = best_anchor(idx, paper_label) if paper_label else None
        anchor_val = None
        if row:
            for key in ("dE_DFT_eV", "dE_SCF_eV", "dE_DPA2_eV"):
                if key in row:
                    anchor_val = f"{row[key]} ({key.split('_')[1]})"
                    break
        for case in sorted(channels[ch]):
            v = channels[ch][case]
            bb = v.get("barrier_best")
            bm = v.get("barrier_mean")
            lines.append(f"| {ch} | {case} | {v.get('n_sites')} | {v.get('n_rows_accepted')} | "
                         f"{'' if bb is None else round(bb, 3)} | {'' if bm is None else round(bm, 3)} | "
                         f"{paper_label or '-'} | {anchor_val or '-'} | {kind} |")

    if comparison:
        lines += ["", "## 3. 两臂对照（同一通道 best barrier）", "",
                  "| 通道 | S (eV) | M (eV) | ΔE(M−S) |", "|---|---|---|---|"]
        for ch in sorted(comparison):
            c = comparison[ch]
            lines.append(f"| {ch} | {c['barrier_best_S']} | {c['barrier_best_M']} | {c['ddE_M_minus_S']} |")

    lines += ["", "## 4. 说明", "",
              "- 势垒口径：`barrier_ts_minus_is`（电子能）；`barrier_zpe`/`barrier_g` 为含 ZPE/自由能修正的变体，"
              "JSON 中逐行保留。",
              "- 关系列：`direct` = 与论文表同键变化；`reverse` = 论文步的逆反应（键变化相同、方向相反）；"
              "`context` = 同键变化但论文在别的共吸附环境/位点评估；`none` = 论文表未覆盖或不适用（本轮新增通道）。"
              "**定量比较仅对 direct/reverse 且环境可比者成立**；本轮可校准的对照见验证线的同世代 D2S 家族。",
              "- 论文锚点取自 `paper_network_table.json`（源 `Fe5C2-network-0709.xlsx`，含 DPA2/SCF/DFT 三列）。"]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
