#!/usr/bin/env python3
"""Consolidated channel table for the Fe5C2(510) C2 network (2026-09-22).

The three exploration waves (v1 frozen 42 channels, c2r2 +50, c2r2w2 +47) each have
their own gate under ``$R/recnet-runs/reports/``. Downstream users should not have to
read three summaries, so this tool merges them into one authoritative table:

    validation-pipeline/summary/c2r2-network-20260922/network_channels_final.{json,csv,md}

Columns per channel: wave, reactant -> products, QC status (pass / review-only /
missing), best model barrier per arm, and an ``arm_sensitive`` flag when the two arms
differ by more than ``--arm-threshold`` eV at the same site (or in their best barriers).

Inputs are produced by ``finalize_network.sh`` runs made **after** the 2026-09-22
``aggregate_network.py`` fixes (arm-prefix naming + channel = reactant -> products).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml


def channel_label_of(case_dir: Path, rxn_idx) -> str | None:
    """Resolve a reaction index inside a campaign case to its channel label."""
    if not case_dir.is_dir() or rxn_idx is None:
        return None
    cands = sorted((case_dir / "prepared_data").glob("*prepared_rmg_data.yaml"))
    if not cands:
        return None
    payload = yaml.safe_load(cands[0].read_text()) or {}
    rxns = payload.get("rxns", [])
    if not (0 <= int(rxn_idx) < len(rxns)):
        return None
    rxn = rxns[int(rxn_idx)]
    return channel_key(rxn)


def load_xval(summary_path: Path, workdir: Path, cache: dict, case_roots):
    """channel -> list of (model_barrier, dft_barrier) from an ABACUS xval batch."""
    payload = json.loads(summary_path.read_text())
    out: dict = {}
    for entry in payload.get("selected", []):
        # NOTE: ``Path("")`` is ``Path(".")`` and ``.`` *is* a directory, so an absent
        # ``case_dir`` must be detected on the string, not via ``is_dir()`` (that bug
        # silently resolved zero channels on the first run of this tool).
        raw_dir = entry.get("case_dir")
        case_dir = Path(raw_dir) if raw_dir else None
        if case_dir is None or not case_dir.is_dir():
            # summaries store the case *name*; locate it under one of --case-root
            name = str(entry.get("case"))
            case_dir = next((root / name for root in case_roots if (root / name).is_dir()),
                            summary_path.parent / name)
        key = (str(case_dir), entry.get("rxn_idx"))
        if key not in cache:
            cache[key] = channel_label_of(case_dir, entry.get("rxn_idx"))
        channel = cache[key]
        if channel is None:
            continue
        label = entry["label"]
        is_meta = workdir / "single_points" / f"{label}_IS" / "meta.json"
        ts_meta = workdir / "single_points" / f"{label}_TS" / "meta.json"
        if not (is_meta.exists() and ts_meta.exists()):
            continue
        e_is = json.loads(is_meta.read_text()).get("energy_eV")
        e_ts = json.loads(ts_meta.read_text()).get("energy_eV")
        if e_is is None or e_ts is None:
            continue
        out.setdefault(channel, []).append((entry["model_barrier_eV"], round(e_ts - e_is, 4)))
    return out


def load_dft_sources(workdir: Path) -> list:
    """Source paths of every SCF-converged ABACUS single point (profile gate).

    The ``dft_profile_arms`` column must not over-claim: a scan manifest alone only proves
    that the *model* profile was produced.  The gate below requires that at least one of the
    profile's frames actually has a converged ABACUS single point.
    """
    import glob
    sources = []
    for path in glob.glob(str(workdir / "single_points" / "*" / "meta.json")):
        try:
            meta = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("energy_eV") is None:
            continue
        source = meta.get("source")
        if source:
            sources.append(str(source))
    return sources


def load_boundary_profile(manifest_paths, case_roots, cache: dict, dft_sources=None):
    """channel -> set of arms that have a protocol-matched constrained profile.

    With ``dft_sources`` given, an arm is only marked when at least one frame of that
    profile has a converged ABACUS single point (label convention
    ``<profile_dir>/<rxn_id>_d<distance>.xyz`` as written by ``make_seeds_scan --dump-frames``).
    """
    out: dict = {}
    gated: list = []
    for spec in manifest_paths or []:
        payload = json.loads(Path(spec).read_text())
        # NOTE: the case path lives at the manifest top level, not inside each scan record
        # (same trap as in build_boundary_abacus_cases.py).
        manifest_case = payload.get("case")
        for scan in payload.get("scans", []):
            arm = "M" if "multi200k" in str(scan.get("model")) else "S"
            case_value = str(scan.get("case") or manifest_case or "")
            case_dir = None
            # 2026-09-25 (R203, independent review F1): resolve the case the *manifest records*
            # first.  The old suffix search overwrote `case_dir` with the LAST matching root
            # (`break` left only the inner loop), so identical case names under several roots
            # (M-c1 exists in campaign-c2, c2r2-campaign, c2r2w2-campaign, c2r3-d, c2r4-c …)
            # silently re-pointed the profile of a v1 boundary channel at a wave case — flagging
            # the wrong channel and dropping the real ones.  Order: recorded absolute path →
            # recorded path's basename under each root (first match, earliest root wins) → skip.
            recorded = Path(case_value)
            if recorded.is_dir() and (recorded / "rxn").is_dir():
                case_dir = recorded
            else:
                for root in case_roots:
                    for candidate in sorted(root.glob("*")):
                        if candidate.is_dir() and (candidate / "rxn").is_dir() and \
                                candidate.name == recorded.name:
                            case_dir = candidate
                            break
                    if case_dir is not None:
                        break
            if case_dir is None:
                continue
            key = (str(case_dir), scan.get("rxn_index"))
            if key not in cache:
                cache[key] = channel_label_of(case_dir, scan.get("rxn_index"))
            channel = cache[key]
            if channel:
                if dft_sources is not None:
                    profile_dir = str(Path(str(scan.get("seed_path") or "")).parent)
                    scan_id = str(scan.get("id") or "")
                    if not any(src.startswith(profile_dir + "/") and f"{scan_id}_d" in src
                               for src in dft_sources):
                        gated.append((channel, arm))
                        continue
                out.setdefault(channel, set()).add(arm)
    if gated:
        print(f"[export_network_table] profile gate: skipped {len(gated)} arm(s) "
              f"without ABACUS points: {sorted(set(gated))}")
    return out


def channel_key(record) -> str:
    reactant = (record.get("reactant") or "").strip()
    product = (record.get("product") or "").strip()
    return f"{reactant} -> {product}" if product else reactant


def load_wave(report: Path, label: str, arm_threshold: float):
    summary = json.loads((report / "network_summary.json").read_text())
    gate = json.loads((report / "c2_gate.json").read_text())

    # Channel-level QC state comes from ts_qc.json (the gate itself only carries
    # counts): a channel is qc_pass when at least one accepted row passed the full
    # TS QC, review_only when it only has accepted-but-QC-failed rows.
    qc_pass, review = set(), set()
    qc_path = report / "ts_qc.json"
    if qc_path.exists():
        qc = json.loads(qc_path.read_text())
        for case in qc.get("cases", []):
            for record in case.get("ts_records", []):
                # NOTE: do **not** name this ``label`` — it would shadow the function's
                # ``label`` parameter (the wave name), which silently put a dict into
                # the ``wave`` column until it blew up in the summary loop.
                rec_label = record.get("label") or {}
                key = channel_key(rec_label)
                if record.get("qc", {}).get("qc_pass"):
                    qc_pass.add(key)
                else:
                    review.add(key)
    missing = {channel_key(c): c for c in gate.get("missing_qc_pass_channels") or []}

    # per-channel best barrier per arm, from the (fixed) per-case aggregation
    best = {}
    for channel, per_case in summary.get("channels", {}).items():
        for case, stats in per_case.items():
            arm = "S" if case.startswith("S-") else ("M" if case.startswith("M-") else None)
            if arm is None:
                continue
            barrier = stats.get("barrier_best")
            if barrier is None:
                continue
            entry = best.setdefault(channel, {"S": None, "M": None})
            if entry[arm] is None or barrier < entry[arm]:
                entry[arm] = round(barrier, 4)

    # same-site arm deltas, used for the sensitivity flag
    deltas = {}
    for row in summary.get("arm_comparison_same_site", []):
        deltas.setdefault(row["channel"], []).append(abs(row["ddE_M_minus_S"]))

    rows = []
    for channel in sorted(set(best) | set(missing)):
        arms = best.get(channel, {"S": None, "M": None})
        arm_delta = None
        if arms["S"] is not None and arms["M"] is not None:
            arm_delta = round(arms["M"] - arms["S"], 4)
        worst_site_delta = max(deltas.get(channel, [0.0]))
        if channel in qc_pass:
            status = "qc_pass"
        elif channel in review:
            status = "review_only"
        elif channel in missing:
            status = "missing"
        else:
            status = "unknown"
        rows.append({
            "wave": label,
            "channel": channel,
            "qc_status": status,
            "barrier_S_best_eV": arms["S"],
            "barrier_M_best_eV": arms["M"],
            "barrier_best_eV": None if None in (arms["S"], arms["M"]) else min(arms["S"], arms["M"]),
            "ddE_best_M_minus_S_eV": arm_delta,
            "max_same_site_abs_delta_eV": round(worst_site_delta, 4) if deltas.get(channel) else None,
            "arm_sensitive": bool(worst_site_delta > arm_threshold
                                  or (arm_delta is not None and abs(arm_delta) > arm_threshold)),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge the three wave gates into one channel table")
    ap.add_argument("--wave", action="append", required=True, metavar="LABEL:REPORT_DIR")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arm-threshold", type=float, default=0.6,
                    help="|Delta(S-M)| above which a channel is flagged arm-sensitive")
    ap.add_argument("--xval-summary", action="append", default=None,
                    metavar="SUMMARY_JSON",
                    help="ABACUS cross-validation summary (repeatable); adds DFT columns")
    ap.add_argument("--workdir", default=None,
                    help="dir holding single_points/ for the xval metas")
    ap.add_argument("--case-root", action="append", default=None,
                    help="campaign root(s) used to resolve 'case' names in xval summaries")
    ap.add_argument("--arbitration-summary", action="append", default=None,
                    metavar="SUMMARY_JSON",
                    help="arbitration summary (repeatable): adds, for the channels that were "
                         "arbitrated against ABACUS, which arm is closer and the errors")
    ap.add_argument("--boundary-profile-scan", action="append", default=None,
                    metavar="SCAN_MANIFEST",
                    help="scan manifest of a boundary-channel constrained scan (repeatable): "
                         "marks channels that have a protocol-matched DFT profile")
    ap.add_argument("--no-profile-dft-gate", action="store_true",
                    help="mark a boundary profile from the scan manifest alone, without requiring "
                         "that one of its frames has a converged ABACUS point (default: gate ON)")
    args = ap.parse_args()

    xval: dict = {}
    if args.xval_summary:
        if not args.workdir:
            raise SystemExit("--xval-summary needs --workdir (dir with single_points/)")
        cache: dict = {}
        case_roots = [Path(p) for p in (args.case_root or [])]
        for spec in args.xval_summary:
            for channel, pairs in load_xval(Path(spec), Path(args.workdir), cache,
                                            case_roots).items():
                xval.setdefault(channel, []).extend(pairs)

    arbitration: dict = {}
    for spec in args.arbitration_summary or []:
        payload = json.loads(Path(spec).read_text())
        for row in payload.get("rows", []):
            arbitration[row["channel"]] = {
                "closer_arm": row.get("closer_arm"),
                "err_S_eV": row.get("err_S_eV"),
                "err_M_eV": row.get("err_M_eV"),
                "dft_S_eV": row.get("dft_S_eV"),
                "dft_M_eV": row.get("dft_M_eV"),
            }

    profiles: dict = {}
    if args.boundary_profile_scan:
        roots = [Path(p) for p in (args.case_root or [])]
        sources = None
        if not args.no_profile_dft_gate:
            if not args.workdir:
                raise SystemExit("--workdir is required for the profile DFT gate "
                                 "(or pass --no-profile-dft-gate)")
            sources = load_dft_sources(Path(args.workdir))
            print(f"[export_network_table] profile gate: {len(sources)} converged ABACUS points indexed")
        profiles = load_boundary_profile(args.boundary_profile_scan, roots, {}, sources)

    rows = []
    for spec in args.wave:
        label, report = spec.split(":", 1)
        rows += load_wave(Path(report), label, args.arm_threshold)

    for row in rows:
        pairs = xval.get(row["channel"], [])
        row["dft_pairs"] = len(pairs)
        row["dft_median_delta_eV"] = (round(statistics.median(d - m for m, d in pairs), 4)
                                      if pairs else None)
        arb = arbitration.get(row["channel"])
        row["arbitrated"] = bool(arb)
        row["dft_closer_arm"] = (arb or {}).get("closer_arm")
        row["arb_err_S_eV"] = (arb or {}).get("err_S_eV")
        row["arb_err_M_eV"] = (arb or {}).get("err_M_eV")
        row["dft_profile_arms"] = ",".join(sorted(profiles.get(row["channel"], set()))) or None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "network_channels_final.json").write_text(
        json.dumps({"schema": "ft2dp_network_channels/1",
                    "arm_threshold_eV": args.arm_threshold,
                    "waves": [s.split(":", 1)[0] for s in args.wave],
                    "channels": rows}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    columns = ["wave", "channel", "qc_status", "barrier_S_best_eV", "barrier_M_best_eV",
               "barrier_best_eV", "ddE_best_M_minus_S_eV", "max_same_site_abs_delta_eV",
               "arm_sensitive", "dft_pairs", "dft_median_delta_eV",
               "arbitrated", "dft_closer_arm", "arb_err_S_eV", "arb_err_M_eV"]
    columns.append("dft_profile_arms")
    with open(out_dir / "network_channels_final.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Fe5C2(510) C2 网络 · 权威通道表（三波合并）", "",
             f"- 臂敏感阈值：同一 (通道, 位点) 上 |Δ(S−M)| > {args.arm_threshold} eV",
             f"- 通道总数：**{len(rows)}**", "",
             "| 波次 | 通道 | QC | 势垒 S (eV) | 势垒 M (eV) | 取优 (eV) | 臂敏 | DFT 对 | DFT 中位 Δ | 仲裁更近 | 误差 S/M (eV) | 剖面 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        def fmt(value):
            return "-" if value is None else f"{value:+.3f}"
        dft_text = ("-" if row.get("dft_median_delta_eV") is None
                    else f"{row['dft_median_delta_eV']:+.3f}")
        arb_s = ("-" if row.get("arb_err_S_eV") is None else f"{row['arb_err_S_eV']:+.3f}")
        arb_m = ("-" if row.get("arb_err_M_eV") is None else f"{row['arb_err_M_eV']:+.3f}")
        lines.append(f"| {row['wave']} | {row['channel']} | {row['qc_status']} | "
                     f"{fmt(row['barrier_S_best_eV'])} | {fmt(row['barrier_M_best_eV'])} | "
                     f"{fmt(row['barrier_best_eV'])} | {'⚠' if row['arm_sensitive'] else ''} | "
                     f"{row.get('dft_pairs', 0) or '-'} | "
                     f"{dft_text} | "
                     f"{row.get('dft_closer_arm') or '-'} | "
                     f"{arb_s}/{arb_m} | "
                     f"{row.get('dft_profile_arms') or '-'} |")
    (out_dir / "network_channels_final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_wave = {}
    for row in rows:
        by_wave.setdefault(row["wave"], []).append(row)
    print(f"channels={len(rows)}")
    for wave, items in by_wave.items():
        print(f"  {wave}: {len(items)} channels | qc_pass "
              f"{sum(1 for r in items if r['qc_status']=='qc_pass')} | "
              f"review_only {sum(1 for r in items if r['qc_status']=='review_only')} | "
              f"missing {sum(1 for r in items if r['qc_status']=='missing')} | "
              f"arm_sensitive {sum(1 for r in items if r['arm_sensitive'])}")
    print(f"wrote {out_dir}/network_channels_final.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
