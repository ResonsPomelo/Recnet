#!/usr/bin/env python3
"""Aggregate RecNet reaction-network results into a barrier table + arm comparison.

Reads, for every discovered case directory:

* ``prepared_data/*prepared_rmg_data.yaml``  -> reaction labels / species names
* ``rxn/final_state_energy*_vg*.yaml``       -> IS/TS/FS energies and barriers
* ``rxn/TS_archive/ts_records*_vg*.yaml``    -> which TS were actually accepted
* ``farm/runs/*/<case>/RUN_MANIFEST.txt``    -> case provenance (model, device, exit)

and writes ``network_summary.json`` + ``network_summary.md``:

* per-channel barrier table (best/mean per reaction, per arm, split by vacancy group)
* S-vs-M arm comparison on shared channels (ΔΔE of the best barrier)
* coverage / failure accounting (channels without an accepted TS, case failure classes)

Usage::

    aggregate_network.py --root <dir> [--root <dir> ...] --out <outdir> [--label-names]

Notes: the electronic barrier ``barrier_ts_minus_is`` is the primary number; ZPE and
free-energy variants are carried through when present.  A channel counts as *accepted*
only if a TS record exists for it (the pipeline prints misses but still exits 0).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml
import numpy as np


def read_xyz(path: Path):
    """Minimal extxyz reader (symbols + positions) so the aggregator does not
    depend on ase being importable in every environment."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    syms, pos = [], []
    for ln in lines[2:2 + n]:
        parts = ln.split()
        syms.append(parts[0])
        pos.append([float(x) for x in parts[1:4]])
    return syms, np.asarray(pos)


def kabsch_rmsd(p1: np.ndarray, p2: np.ndarray) -> float:
    p1 = p1 - p1.mean(0)
    p2 = p2 - p2.mean(0)
    u, _, vt = np.linalg.svd(p1.T @ p2)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ d @ u.T
    return float(np.sqrt(((p1 @ rot.T - p2) ** 2).sum(1).mean()))

ENERGY_GLOBS = ("final_state_energy*.yaml", "final_state_energy_*.yaml")
TS_GLOBS = ("ts_records*.yaml",)


def find_cases(roots):
    """Discover case dirs: any dir holding a prepared dataset (or holding such dirs)."""
    cases = []
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            raise SystemExit(f"root not found: {root}")
        hits = sorted({p.parent.parent for p in root.rglob("prepared_data/*prepared_rmg_data.yaml")})
        if not hits:
            hits = sorted({p.parent for p in root.rglob("prepared_rmg_data.yaml")})
        cases.extend(hits)
    # de-dup, keep order
    seen, out = set(), []
    for c in cases:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_prepared(case: Path):
    cands = sorted((case / "prepared_data").glob("*prepared_rmg_data.yaml"))
    if not cands:
        return None
    payload = yaml.safe_load(cands[0].read_text())
    return payload


def read_yaml_list(path: Path, key):
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # corrupt/partial file -> report, do not crash the campaign
        return [], f"{type(exc).__name__}: {exc}"
    rows = payload.get(key) or []
    return rows, None


def case_files(case: Path, pattern: str):
    """Files under <case>/rxn matching one or more glob patterns.

    Deduplicated across patterns: ``**`` also matches the top level, and
    ``final_state_energy*.yaml`` overlaps ``final_state_energy_*.yaml``; without
    a shared set every row would be counted twice.
    """
    if isinstance(pattern, str):
        pattern = (pattern,)
    seen, out = set(), []
    for pat in pattern:
        for f in list(case.glob(f"rxn/**/{pat}")) + list(case.glob(f"rxn/{pat}")):
            if f not in seen:
                seen.add(f)
                out.append(f)
    return sorted(out)


def case_rows(case: Path):
    prepared = load_prepared(case)
    labels = {}
    if prepared:
        for i, r in enumerate(prepared.get("rxns", [])):
            labels[i] = {"reactant": r.get("reactant"), "product": r.get("product")}
    rows, problems = [], []
    for f in case_files(case, ENERGY_GLOBS):
            data, err = read_yaml_list(f, "final_state_energy")
            if err:
                problems.append({"file": str(f), "error": err})
                continue
            for r in data:
                rxn_idx = r.get("rxn_idx")
                row = {
                    "case": case.name,
                    "case_dir": str(case),
                    "file": str(f),
                    "vg": f.stem.split("_vg")[-1] if "_vg" in f.stem else None,
                    "rxn_idx": rxn_idx,
                    "site": r.get("site"),
                    "label": labels.get(rxn_idx, {}),
                    "e_is": r.get("e_is"),
                    "e_ts": r.get("e_ts"),
                    "e_fs": r.get("e_fs"),
                    "barrier": r.get("barrier_ts_minus_is"),
                    "barrier_zpe": r.get("barrier_ts_minus_is_zpe"),
                    "barrier_g": r.get("barrier_g_ts_minus_is"),
                    "delta_e_fs_minus_is": r.get("delta_e_fs_minus_is"),
                    # 2026-09-25 (R186): thermodynamic companions, additive only — the delivered
                    # channel table stays barrier-only (export_network_table.py picks its own
                    # columns), these fields exist for export_thermo_table.py / the analysis.
                    "delta_e_zpe_fs_minus_is": r.get("delta_e_fs_minus_is_zpe"),
                    "delta_g_fs_minus_is": r.get("delta_g_fs_minus_is"),
                    "tag": r.get("tag"),
                }
                rows.append(row)
    accepted = set()
    accepted_records = {}
    for f in case_files(case, TS_GLOBS):
        data, err = read_yaml_list(f, "ts_records")
        if err:
            problems.append({"file": str(f), "error": err})
            continue
        for rec in data:
            key = (rec.get("rxn_idx"), rec.get("site"),
                   f.stem.split("_vg")[-1] if "_vg" in f.stem else None)
            accepted.add(key)
            accepted_records[key] = rec
    manifest = None
    for m in sorted(case.glob("farm/runs/*/*/RUN_MANIFEST.txt")):
        manifest = m
    return rows, accepted, accepted_records, labels, problems, manifest


def parse_manifest(path: Path):
    if path is None:
        return {}
    fields = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fields.setdefault(k.strip(), v.strip())
    return fields


def unique_case_name(case: Path, used: set[str]) -> str:
    """Return a stable display name; disambiguate same-named cases across roots."""
    name = case.name
    if name not in used:
        return name
    suffix = ""
    stem = name
    for candidate in ("-S", "-M"):
        if name.endswith(candidate):
            suffix = candidate
            stem = name[:-2]
            break
    token = f"{abs(hash(str(case))) % 10000:04d}"
    return f"{stem}__{token}{suffix}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True, help="campaign/case root (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cases = find_cases(args.root)
    if not cases:
        raise SystemExit("no case directories found (looked for prepared_data/*prepared_rmg_data.yaml)")

    per_case, problems_all = {}, []
    used_names = set()
    for case in cases:
        display_name = unique_case_name(case, used_names)
        used_names.add(display_name)
        rows, accepted, accepted_records, labels, problems, manifest = case_rows(case)
        mf = parse_manifest(manifest)
        for r in rows:
            key = (r["rxn_idx"], r["site"], r["vg"])
            r["accepted"] = key in accepted
            r["case"] = display_name
        per_case[display_name] = {
            "case_dir": str(case),
            "model": mf.get("model"),
            "model_sha256": mf.get("model_sha256"),
            "head": mf.get("head"),
            "device": mf.get("device"),
            "exit_code": mf.get("exit_code"),
            "failure_class": mf.get("failure_class"),
            "n_rows": len(rows),
            # two distinct notions, kept separate on purpose:
            #   n_ts_records        -> unique accepted TS written to TS_archive
            #   n_rows_accepted     -> energy rows that can be matched to an accepted TS
            "n_ts_records": len(accepted),
            "n_rows_accepted": sum(1 for r in rows if r["accepted"]),
            "ts_records": {f"{k[0]}|{k[1]}|{k[2]}": v.get("ts_xyz") for k, v in accepted_records.items()},
            "labels": {str(k): v for k, v in labels.items()},
            "rows": rows,
        }
        problems_all.extend(problems)

    # channel-level aggregation: label -> per-case stats
    channels = {}
    for case, info in per_case.items():
        by_label = {}
        for r in info["rows"]:
            # A channel is (reactant -> products): keying on the reactant alone merged
            # every channel of a poly-functional species (e.g. all four C2H4O_1*
            # channels) into one bucket, which silently corrupted the barrier table
            # and the arm comparison (2026-09-22 fix). Fall back to the reaction index.
            reactant = r["label"].get("reactant")
            product = r["label"].get("product")
            if reactant and product:
                lab = f"{reactant} -> {product}"
            elif reactant:
                lab = f"{reactant} (rxn{r['rxn_idx']})"
            else:
                lab = f"rxn{r['rxn_idx']}"
            by_label.setdefault(lab, []).append(r)
        for lab, rs in by_label.items():
            bars = [r["barrier"] for r in rs if r.get("barrier") is not None]
            channels.setdefault(lab, {})[case] = {
                "n_sites": len(rs),
                "n_rows_accepted": sum(1 for r in rs if r["accepted"]),
                "barrier_best": min(bars) if bars else None,
                "barrier_mean": statistics.fmean(bars) if bars else None,
                "sites": [{"site": r["site"], "vg": r["vg"], "barrier": r["barrier"],
                           "barrier_zpe": r["barrier_zpe"], "accepted": r["accepted"],
                           "delta_e": r.get("delta_e_fs_minus_is"),
                           "delta_e_zpe": r.get("delta_e_zpe_fs_minus_is"),
                           "delta_g": r.get("delta_g_fs_minus_is")} for r in rs],
            }

    # arm comparison for shared channels. Two naming conventions are in use:
    #   * suffix — ``<base>-S`` / ``<base>-M`` (early pilots)
    #   * prefix — ``S-<base>`` / ``M-<base>`` (the c2r2 / c2r2w2 campaigns)
    # The prefix form used to be invisible here, which silently produced an empty
    # ``arm_comparison`` in the delivered summaries (2026-09-22 fix).
    def split_arm(case_name: str):
        for arm in ("S", "M"):
            if case_name.startswith(f"{arm}-"):
                return arm, case_name[2:]
            if case_name.endswith(f"-{arm}"):
                return arm, case_name[: -2]
        return None, case_name

    comparison = []
    by_base: dict = {}
    for lab, per in channels.items():
        for case_name, value in per.items():
            arm, base = split_arm(case_name)
            if arm is None:
                continue
            by_base.setdefault((lab, base), {})[arm] = value
    for (lab, base), arms in sorted(by_base.items()):
        if "S" in arms and "M" in arms:
            bs, bm = arms["S"]["barrier_best"], arms["M"]["barrier_best"]
            comparison.append({
                "channel": lab, "base": base,
                "barrier_best_S": bs, "barrier_best_M": bm,
                "ddE_M_minus_S": (None if (bs is None or bm is None) else round(bm - bs, 4)),
            })

    if False:  # 旧实现（后缀约定）保留为注释，避免再踩同一坑
        for lab, per in channels.items():
            s = {c: v for c, v in per.items() if c.endswith("-S")}
            m = {c: v for c, v in per.items() if c.endswith("-M")}
            for cs, vs in s.items():
                cm = cs[:-2] + "-M"
                if cm in per:
                    pass

    # like-for-like comparison: same channel AND same (site, vacancy group) in both arms
    same_site = []
    for (lab, base), arms in sorted(by_base.items()):
        if "S" not in arms or "M" not in arms:
            continue
        for r in arms["S"].get("sites", []):
            rm = next((x for x in arms["M"].get("sites", [])
                       if (x.get("site"), x.get("vg")) == (r.get("site"), r.get("vg"))), None)
            if rm is None or r.get("barrier") is None or rm.get("barrier") is None:
                continue
            same_site.append({
                "channel": lab, "base": base, "site": r.get("site"), "vg": r.get("vg"),
                "barrier_S": round(r["barrier"], 4), "barrier_M": round(rm["barrier"], 4),
                "ddE_M_minus_S": round(rm["barrier"] - r["barrier"], 4),
                "accepted_S": r.get("accepted"), "accepted_M": rm.get("accepted"),
            })

    # geometry fingerprint for saddles found by BOTH arms at the same (rxn, site, vg):
    # are they the same stationary point?  (Kabsch RMSD of ts.xyz, same atom order)
    fingerprints = []
    arms = sorted({c for c in per_case if c.endswith("-S") or c.endswith("-M")})
    for case_s in [c for c in arms if c.endswith("-S")]:
        case_m = case_s[:-2] + "-M"
        if case_m not in per_case:
            continue
        recs_s = per_case[case_s]["ts_records"]
        recs_m = per_case[case_m]["ts_records"]
        for k in sorted(set(recs_s) & set(recs_m)):
            p_s, p_m = recs_s[k], recs_m[k]
            if not (p_s and p_m and Path(p_s).exists() and Path(p_m).exists()):
                continue
            try:
                sy_s, pos_s = read_xyz(Path(p_s))
                sy_m, pos_m = read_xyz(Path(p_m))
                if sy_s != sy_m or pos_s.shape != pos_m.shape:
                    fingerprints.append({"TS_key": k, "S_case": case_s, "M_case": case_m,
                                         "rmsd": None, "note": "atom order/shape mismatch"})
                    continue
                fingerprints.append({"TS_key": k, "S_case": case_s, "M_case": case_m,
                                     "rmsd": round(kabsch_rmsd(pos_s, pos_m), 4), "note": None})
            except Exception as exc:
                fingerprints.append({"TS_key": k, "S_case": case_s, "M_case": case_m,
                                     "rmsd": None, "note": f"{type(exc).__name__}: {exc}"})

    summary = {
        "cases": per_case,
        "channels": channels,
        "arm_comparison": comparison,
        "arm_comparison_same_site": sorted(same_site, key=lambda x: (x["channel"], str(x["site"]), str(x["vg"]))),
        "saddle_fingerprints": fingerprints,
        "problems": problems_all,
        "totals": {
            "n_cases": len(per_case),
            "n_channels": len(channels),
            "n_ts_records": sum(v["n_ts_records"] for v in per_case.values()),
            "n_rows_accepted": sum(v["n_rows_accepted"] for v in per_case.values()),
            "n_rows": sum(v["n_rows"] for v in per_case.values()),
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "network_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))

    lines = ["# RecNet 反应网络汇总", "",
             f"- cases: {summary['totals']['n_cases']}；channels: {summary['totals']['n_channels']}；"
             f"TS records: {summary['totals']['n_ts_records']}；"
             f"accepted rows: {summary['totals']['n_rows_accepted']}；energy rows: {summary['totals']['n_rows']}",
             "", "## case 状态", "",
             "| case | model | head | device | exit | failure_class | rows | TS records | rows accepted |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(per_case):
        v = per_case[name]
        lines.append(f"| {name} | {(v['model'] or '').split('/')[-1]} | {v['head'] or ''} | {v['device'] or ''} | "
                     f"{v['exit_code'] or ''} | {v['failure_class'] or ''} | {v['n_rows']} | "
                     f"{v['n_ts_records']} | {v['n_rows_accepted']} |")
    lines += [
             "", "## 通道势垒（best，eV，电子能口径）", "",
             "| channel | case | rows | rows accepted | barrier_best | barrier_mean |", "|---|---|---|---|---|---|"]
    for lab in sorted(channels):
        for case in sorted(channels[lab]):
            v = channels[lab][case]
            lines.append(f"| {lab} | {case} | {v['n_sites']} | {v['n_rows_accepted']} | "
                         f"{'' if v['barrier_best'] is None else round(v['barrier_best'], 3)} | "
                         f"{'' if v['barrier_mean'] is None else round(v['barrier_mean'], 3)} |")
    if comparison:
        lines += ["", "## 两臂对照（同一通道 best barrier）", "",
                  "| channel | S | M | ΔE(M−S) |", "|---|---|---|---|"]
        for c in sorted(comparison, key=lambda x: x["channel"]):
            lines.append(f"| {c['channel']} | {c['barrier_best_S']} | {c['barrier_best_M']} | {c['ddE_M_minus_S']} |")
    if summary.get("saddle_fingerprints"):
        lines += ["", "## 鞍点几何指纹（两臂在同一 (rxn, site, vg) 都找到 TS 时）", "",
                  "| TS key | S case | M case | RMSD (Å) | note |", "|---|---|---|---|---|"]
        for f in summary["saddle_fingerprints"]:
            lines.append(f"| {f['TS_key']} | {f['S_case']} | {f['M_case']} | "
                         f"{'' if f['rmsd'] is None else f['rmsd']} | {f['note'] or ''} |")
    if summary.get("arm_comparison_same_site"):
        lines += ["", "## 两臂对照（同通道 + 同 site/vg，逐点）", "",
                  "| channel | site | vg | S | M | ΔE(M−S) |", "|---|---|---|---|---|---|"]
        for c in summary["arm_comparison_same_site"]:
            lines.append(f"| {c['channel']} | {c['site']} | {c['vg']} | {c['barrier_S']} | "
                         f"{c['barrier_M']} | {c['ddE_M_minus_S']} |")
    if problems_all:
        lines += ["", "## 解析问题", ""] + [f"- {p['file']}: {p['error']}" for p in problems_all]
    (out / "network_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'network_summary.json'} and {out/'network_summary.md'}")
    print(f"cases={summary['totals']['n_cases']} channels={summary['totals']['n_channels']} "
          f"ts_records={summary['totals']['n_ts_records']} rows_accepted={summary['totals']['n_rows_accepted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
