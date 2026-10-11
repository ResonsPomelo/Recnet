#!/usr/bin/env python3
"""Network-level analysis of a delivered channel table (2026-09-25, R179).

The delivery table answers "which channels exist and what are their barriers"; this tool answers
"what does the network look like" — composition, reaction-class breakdown, arm divergence,
species connectivity, literature-anchor coverage and the hard-step list.  Everything is computed
from the table itself (plus the prepared dataset for bond identity and the paper anchors), so it
is a zero-GPU analysis that can be re-run on every snapshot.

Usage::

    analyze_network_table.py --table <network_channels_final.json> --out-dir <dir> \
        [--dataset <prepared_rmg_data.yaml>] [--anchors paper_network_table.json]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "Recnet"))

import yaml  # noqa: E402

from recnet_adapter.tools.campaign.build_campaign import channel_bond_type  # noqa: E402


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    vs = sorted(values)
    return {
        "n": len(vs),
        "min": round(vs[0], 3),
        "q1": round(vs[len(vs) // 4], 3),
        "median": round(statistics.median(vs), 3),
        "q3": round(vs[(3 * len(vs)) // 4], 3),
        "max": round(vs[-1], 3),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Network-level analysis of a delivered channel table")
    ap.add_argument("--table", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", default=str(ROOT / "Recnet/network_inputs/c2r5/"
                                             "fe5c2_510_c2r5.prepared_rmg_data.yaml"))
    ap.add_argument("--anchors", default=str(ROOT / "validation-pipeline/reaction/"
                                             "paper_network_table.json"))
    ap.add_argument("--thermo", action="append", default=None,
                    help="thermo_table.json (export_thermo_table.py; repeatable — rows merged, "
                         "first occurrence wins) — adds the BEP section")
    ap.add_argument("--summary", default=None,
                    help="optional network_summary.json (gate) — adds the site-dispersion section")
    args = ap.parse_args(argv)

    table_path = Path(args.table)
    rows = json.loads(table_path.read_text())["channels"]
    payload = yaml.safe_load(Path(args.dataset).read_text())
    species = payload["species"]
    by_label = {}
    for r in payload["rxns"]:
        by_label.setdefault(f"{r['reactant']} -> {r['product']}", r)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. composition ------------------------------------------------------------
    waves = collections.Counter(c["wave"] for c in rows)
    qc = collections.Counter(c["qc_status"] for c in rows)
    evidence = {
        "arbitrated": sum(1 for c in rows if c.get("arbitrated")),
        "dft_pairs(xval)": sum(1 for c in rows if c.get("dft_pairs")),
        "dft_profile_arms": sum(1 for c in rows if c.get("dft_profile_arms")),
    }
    arm_sens = [c for c in rows if c.get("arm_sensitive")]
    no_evidence = [c for c in arm_sens
                   if not (c.get("arbitrated") or c.get("dft_pairs") or c.get("dft_profile_arms"))]

    # ---- 2. reaction class (element pair of the broken bond) ------------------------
    classes: dict[str, list[dict]] = collections.defaultdict(list)
    unclassified = []
    for c in rows:
        r = by_label.get(c["channel"])
        if not r:
            unclassified.append(c["channel"])
            continue
        sp = r["reactant_species"][0]
        bt = channel_bond_type(sp, r["broken_bond"], species,
                               Path(args.dataset).parent)
        classes[bt].append(c)

    best = [c["barrier_best_eV"] for c in rows if c.get("barrier_best_eV") is not None]
    barriers_S = [c["barrier_S_best_eV"] for c in rows if c.get("barrier_S_best_eV") is not None]
    barriers_M = [c["barrier_M_best_eV"] for c in rows if c.get("barrier_M_best_eV") is not None]
    dd = [abs(c["barrier_M_best_eV"] - c["barrier_S_best_eV"]) for c in rows
          if c.get("barrier_M_best_eV") is not None and c.get("barrier_S_best_eV") is not None]
    bins = {"<0.2": 0, "0.2-0.6": 0, "0.6-1.0": 0, ">=1.0": 0}
    for v in dd:
        bins["<0.2" if v < 0.2 else "0.2-0.6" if v < 0.6 else "0.6-1.0" if v < 1.0 else ">=1.0"] += 1
    downhill = [c for c in rows if (c.get("barrier_best_eV") or 99) <= 0.05]

    # ---- 3. species connectivity ----------------------------------------------------
    as_reactant = collections.Counter()
    as_product = collections.Counter()
    for c in rows:
        lhs, _, rhs = c["channel"].partition("->")
        as_reactant[lhs.strip()] += 1
        for p in rhs.split():
            as_product[p.strip()] += 1
    species_all = set(as_reactant) | set(as_product)
    terminal = sorted(s for s in species_all if as_reactant.get(s, 0) == 0)   # only produced
    hub = sorted(species_all, key=lambda s: -(as_reactant.get(s, 0) + as_product.get(s, 0)))[:8]

    # ---- 4. literature anchors ------------------------------------------------------
    anchor_rows = json.loads(Path(args.anchors).read_text())["rows"]
    try:
        from recnet_adapter.tools.report.make_network_report import MAPPING
    except Exception:                                            # pragma: no cover
        MAPPING = {}
    by_reactant_kind = {r: (step, kind) for r, (step, kind) in MAPPING.items()}
    kinds = collections.Counter()
    unmapped_channels = 0
    for c in rows:
        lhs = c["channel"].split(" -> ")[0]
        entry = by_reactant_kind.get(lhs)
        if entry is None:
            unmapped_channels += 1
        else:
            kinds[entry[1]] += 1
    covered_steps = sorted({step for r, (step, kind) in MAPPING.items()
                            if step and kind in ("context", "reverse", "direct")
                            and any(c["channel"].startswith(r) for c in rows)})

    # ---- 6b. kinetic traps: per-species escape / formation barriers -------------------
    out_bar: dict[str, float] = {}
    in_bar: dict[str, float] = {}
    for c in rows:
        b = c.get("barrier_best_eV")
        if b is None:
            continue
        lhs, _, rhs = c["channel"].partition("->")
        lhs = lhs.strip()
        out_bar[lhs] = min(out_bar.get(lhs, 1e9), b)
        for p in rhs.split():
            p = p.strip()
            in_bar[p] = min(in_bar.get(p, 1e9), b)
    # gases appear as reactants in "gas entry" channels; their escape barrier is a desorption /
    # adsorption number, not a surface kinetic trap — keep the trap lists to surface species.
    surface_out = {s: b for s, b in out_bar.items() if not s.endswith("(g)")}
    traps = sorted(surface_out.items(), key=lambda kv: -kv[1])[:10]
    easiest = sorted(surface_out.items(), key=lambda kv: kv[1])[:5]

    # ---- 5. hard steps --------------------------------------------------------------
    hardest = sorted([c for c in rows if c.get("barrier_best_eV") is not None],
                     key=lambda c: -c["barrier_best_eV"])[:10]
    most_divergent = sorted([c for c in rows if c.get("ddE_best_M_minus_S_eV") is not None],
                            key=lambda c: -abs(c["ddE_best_M_minus_S_eV"]))[:10]

    # ---- 8. Ea–ΔE relation (BEP-style) per reaction class ----------------------------
    bep: dict[str, dict] = {}
    thermo_rows, seen_t = [], set()
    for tp in (args.thermo or []):
        p = Path(tp)
        if not p.exists():
            continue
        for r in json.loads(p.read_text()).get("rows", []):
            key = (r.get("channel"), r.get("arm"))
            # R216: only de-duplicate rows that carry an arm; a file without the field must not
            # collapse its rows into one (the unit test's minimal fixture has no `arm`).
            if r.get("arm") is not None and key in seen_t:
                continue
            seen_t.add(key)
            thermo_rows.append(r)
    if thermo_rows:
        chan_class = {ch: k for k, v in classes.items() for ch in {c["channel"] for c in v}}
        by_class: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
        for r in thermo_rows:
            if r.get("delta_e_eV") is None or r.get("barrier_eV") is None:
                continue
            by_class[chan_class.get(r["channel"], "unknown")].append(
                (r["delta_e_eV"], r["barrier_eV"]))
        by_class["(all)"] = [p for v in by_class.values() for p in v]
        for k, pairs in by_class.items():
            if len(pairs) < 3:
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in pairs)
            syy = sum((y - my) ** 2 for y in ys)
            slope = sxy / sxx if sxx else 0.0
            r = sxy / (sxx * syy) ** 0.5 if sxx and syy else 0.0
            bep[k] = {"n": len(pairs), "slope": round(slope, 3),
                      "intercept": round(my - slope * mx, 3), "pearson_r": round(r, 3),
                      "delta_e_median": round(statistics.median(xs), 3),
                      "barrier_median": round(statistics.median(ys), 3)}

    # ---- 9. site dispersion: headline "best" vs the spread over accepted sites -------
    # (2026-09-25, R197) The delivered table quotes the *best* accepted site per channel; on the
    # O-rich wave that best can sit ~1.4 eV below the median site (max observed 3.7 eV), so the
    # dispersion is a reading caveat that must travel with the headline number.
    dispersion: dict = {}
    if args.summary and Path(args.summary).exists():
        summary_payload = json.loads(Path(args.summary).read_text())
        per_channel_sites: dict[str, list[float]] = {}
        for chan, cases in summary_payload.get("channels", {}).items():
            vals = [e["barrier"] for rec in cases.values() for e in rec.get("sites", [])
                    if e.get("accepted") and e.get("barrier") is not None]
            if vals:
                per_channel_sites[chan] = vals
        spreads = {c: max(v) - min(v) for c, v in per_channel_sites.items()}
        if spreads:
            vals = sorted(spreads.values())
            worst = sorted(spreads.items(), key=lambda kv: -kv[1])[:5]
            dispersion = {
                "channels": len(spreads),
                "entries": sum(len(v) for v in per_channel_sites.values()),
                "spread_median": round(statistics.median(vals), 3),
                "spread_max": round(vals[-1], 3),
                "largest": [(c, round(s, 3), round(min(per_channel_sites[c]), 3),
                             round(max(per_channel_sites[c]), 3)) for c, s in worst],
                "best_median": round(statistics.median([min(v) for v in per_channel_sites.values()]), 3),
            }

    sha = hashlib.sha256(table_path.read_bytes()).hexdigest()
    report = {
        "schema": "network-analysis-v1",
        "generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "table": str(table_path),
        "table_sha256": sha,
        "dataset": str(args.dataset),
        "channels": len(rows),
        "waves": dict(waves),
        "qc": dict(qc),
        "evidence": evidence,
        "arm_sensitive": len(arm_sens),
        "arm_sensitive_without_dft": len(no_evidence),
        "barriers": {"best": _stats(best), "S": _stats(barriers_S), "M": _stats(barriers_M)},
        "arm_divergence_bins": bins,
        "classes": {k: {"n": len(v), "best": _stats([c["barrier_best_eV"] for c in v
                                                     if c.get("barrier_best_eV") is not None])}
                    for k, v in sorted(classes.items(), key=lambda kv: -len(kv[1]))},
        "unclassified_channels": len(unclassified),
        "species": {"n": len(species_all), "terminal_only": len(terminal),
                    "hubs": [(s, as_reactant.get(s, 0), as_product.get(s, 0)) for s in hub]},
        "anchors": {"paper_steps": len(anchor_rows), "mapped_reactants": len(MAPPING),
                    "channel_kinds": dict(kinds),
                    "unmapped_channels": unmapped_channels,
                    "covered_paper_steps": covered_steps},
        "kinetic_traps": [(s, round(b, 3)) for s, b in traps],
        "easiest_escape": [(s, round(b, 3)) for s, b in easiest],
        "downhill_le_0.05eV": len(downhill),
        "hardest": [(c["channel"], c["barrier_best_eV"], c["wave"]) for c in hardest],
        "most_divergent": [(c["channel"], round(c["ddE_best_M_minus_S_eV"], 3),
                            bool(c.get("arbitrated"))) for c in most_divergent],
        "bep": bep,
        "site_dispersion": dispersion,
    }
    (out_dir / "network_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    md = [f"# Fe5C2(510) C≤2 网络分析（{report['generated']}）", ""]
    md.append(f"数据源：`{table_path.name}` sha256 `{sha[:16]}…`（{len(rows)} 通道）；"
              f"数据集 `{Path(args.dataset).name}`。")
    md += ["", "## 1. 构成", "",
           f"- 波次：{' / '.join(f'{k} {v}' for k, v in sorted(waves.items()))}",
           f"- QC：`qc_pass` {qc.get('qc_pass', 0)} / `review_only` {qc.get('review_only', 0)} / "
           f"`missing` {qc.get('missing', 0)}",
           f"- 证据：仲裁 {evidence['arbitrated']} / 交叉验证 {evidence['dft_pairs(xval)']} / "
           f"边界剖面 {evidence['dft_profile_arms']}",
           f"- 臂敏感（表内 `arm_sensitive` 标记，同址最大差口径）{len(arm_sens)} 条，"
           f"其中**尚无 DFT 证据 {len(no_evidence)} 条**；"
           f"以最优臂差口径 |Δ(S−M)|≥0.6 eV 计 **{bins['0.6-1.0'] + bins['>=1.0']} 条**"
           f"（= 仲裁队列口径，见 §6）",
           f"- 势垒（最优臂）：{report['barriers']['best']}",
           f"- 下坡步（≤0.05 eV）：{len(downhill)} 条", "",
           "## 2. 反应类（按断裂键的元素对）", "",
           "| 类型 | 通道数 | 势垒 min/中位/max (eV) |", "|---|---|---|"]
    for k, v in report["classes"].items():
        st = v["best"]
        md.append(f"| {k} | {v['n']} | {st.get('min', 'NA')} / {st.get('median', 'NA')} / "
                  f"{st.get('max', 'NA')} |")
    md += ["", f"（无法归类：{len(unclassified)} 条——数据集里没有对应反应行）", "",
           "## 3. 双臂分歧分布", "",
           "| \\|Δ(S−M)\\| | 通道数 |", "|---|---|"]
    for k, v in bins.items():
        md.append(f"| {k} eV | {v} |")
    md += ["", "## 4. 物种连通性", "",
           f"- 表内身份物种 **{len(species_all)}**；仅作为产物出现（无 outgoing）**{len(terminal)}** 个",
           "- 度最高的物种（作反应物/作产物）："]
    for s, r_, p_ in report["species"]["hubs"]:
        md.append(f"  - `{s}`：反应物 {r_} / 产物 {p_}")
    md += ["", "## 5. 文献锚定覆盖", "",
           f"- 论文锚表 {len(anchor_rows)} 行（其中 TS 行 {sum(1 for r in anchor_rows if str(r.get('step', '')).startswith('TS'))}）",
           f"- 本表通道按论文映射口径："
           + " / ".join(f"{k} {v}" for k, v in sorted(kinds.items()))
           + f"；**无对应论文步** {unmapped_channels} 条",
           f"- 被映射到的论文步：{', '.join(covered_steps) if covered_steps else '（无）'}", "",
           "## 6. 难点与分歧清单", "", "**势垒最高 10 条**", ""]
    for ch, b, w in report["hardest"]:
        md.append(f"- {b:.2f} eV — `{ch}`（{w}）")
    md += ["", "**臂分歧最大 10 条**（`arbitrated` = 是否已有双臂 DFT 仲裁）", ""]
    for ch, d, a in report["most_divergent"]:
        md.append(f"- Δ={d:+.2f} eV — `{ch}`{'（已仲裁）' if a else '（**待仲裁**）'}")
    md += ["", "## 7. 动力学瓶颈（物种的逃逸势垒 = 其 outgoing 通道的最低势垒）", "",
           "**最难离开的物种（动力学陷阱）**", ""]
    for s, b in report["kinetic_traps"]:
        md.append(f"- {b:.2f} eV — `{s}`")
    md += ["", "**最容易离开的物种**", ""]
    for s, b in report["easiest_escape"]:
        md.append(f"- {b:.2f} eV — `{s}`")
    if bep:
        md += ["", "## 8. Ea–ΔE 关系（同一最优势垒位点，BEP 口径）", "",
               "| 反应类 | n | 斜率 | 截距 (eV) | Pearson r | ΔE 中位 | Ea 中位 |",
               "|---|---|---|---|---|---|---|"]
        for k, v in sorted(bep.items(), key=lambda kv: -kv[1]["n"]):
            md.append(f"| {k} | {v['n']} | {v['slope']} | {v['intercept']} | {v['pearson_r']} | "
                      f"{v['delta_e_median']} | {v['barrier_median']} |")
        md += ["", "（本表约定 `Ea ≈ 截距 + 斜率 × ΔE`，ΔE = E_FS − E_IS。**斜率 > 0 ⇒ 更放热的步势垒更低**"
                   "（BEP 正相关，越接近 +1 说明过渡态越「类产物」）；斜率 ≈ 0 ⇒ 势垒与驱动力解耦。"
                   "样本 < 3 的类不列出。）"]
    if dispersion:
        md += ["", "## 9. 位点离散度（读势垒前必读）", "",
               f"- 统计口径：接受条目的**逐位点**势垒（{dispersion['entries']} 条 / {dispersion['channels']} 通道）；"
               f"表内 headline 是**最优位点**（中位 {dispersion['best_median']} eV）",
               f"- **同通道 max−min 离散度：中位 {dispersion['spread_median']} eV、最大 {dispersion['spread_max']} eV**",
               "", "| 通道 | 离散度 (eV) | 最优 | 最差 |", "|---|---|---|---|"]
        for ch, s, b, w in dispersion["largest"]:
            md.append(f"| `{ch}` | {s} | {b} | {w} |")
        md += ["", "（含义：势垒的**位点选择**贡献可与反应本身的差异同量级；引用单一位点数值时必须说明。"
                   "O-rich 新波尤其明显。）"]
    (out_dir / "network_analysis.md").write_text("\n".join(md) + "\n")
    print(f"[analysis] wrote {out_dir}/network_analysis.{{md,json}} "
          f"({len(rows)} channels, {len(species_all)} species, {len(no_evidence)} arm-sensitive without DFT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
