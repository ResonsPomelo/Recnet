#!/usr/bin/env python3
"""Collect a live progress snapshot for a chunked Recnet campaign."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def case_roots(roots: list[str]) -> list[Path]:
    cases: list[Path] = []
    for root in roots:
        root_path = Path(root).resolve()
        if not root_path.exists():
            raise SystemExit(f"root not found: {root}")
        hits = sorted({p.parent.parent for p in root_path.rglob("prepared_data/*prepared_rmg_data.yaml")})
        cases.extend(hits)
    seen: set[Path] = set()
    out: list[Path] = []
    for case in cases:
        if case not in seen:
            seen.add(case)
            out.append(case)
    return out


def latest_real_manifest(case: Path) -> tuple[Path | None, dict[str, str] | None]:
    """Return the newest non-dry RUN_MANIFEST record, if present."""
    records: list[tuple[Path, dict[str, str]]] = []
    for manifest in case.glob("farm/runs/*/*/RUN_MANIFEST.txt"):
        current: dict[str, str] | None = None
        for raw in manifest.read_text(errors="replace").splitlines():
            line = raw.strip()
            if line.startswith("===== RUN_MANIFEST record"):
                current = {}
            elif line == "===== END record =====":
                if current is not None and current.get("dry_run", "").lower() == "false":
                    records.append((manifest, current))
                current = None
            elif current is not None and ":" in line:
                key, _, value = line.partition(":")
                current.setdefault(key.strip(), value.strip())
    if not records:
        return None, None
    records.sort(key=lambda item: (item[1].get("ended_at", ""), item[1].get("started_at", "")))
    return records[-1]


def latest_log(case: Path) -> Path | None:
    logs = list(case.glob("farm/runs/*/*/case.log"))
    return max(logs, key=lambda p: p.stat().st_mtime) if logs else None


def has_any_manifest(case: Path) -> bool:
    return any(case.glob("farm/runs/*/*/RUN_MANIFEST.txt"))


def infer_stage(text: str) -> str:
    tail = "\n".join(text.splitlines()[-500:])
    if "Final-state energy evaluation finished" in tail:
        return "energy_finished"
    if "All orientations rejected" in tail or "Processing rxn_" in tail:
        return "ts_search"
    if "Valid sites for sp_" in tail:
        return "adsorption_enumeration"
    if "Checkpoint loaded" in tail or "BFGSLineSearch" in tail or "MDMin" in tail:
        return "optimization"
    return "unknown"


def collect_case(case: Path, now: float) -> dict[str, Any]:
    prepared_files = sorted(case.glob("prepared_data/*prepared_rmg_data.yaml"))
    rxns: list[dict[str, Any]] = []
    species_n = 0
    if prepared_files:
        payload = yaml.safe_load(prepared_files[0].read_text()) or {}
        rxns = payload.get("rxns") or []
        species_n = len(payload.get("species") or {})

    ts_files = sorted(case.glob("rxn/**/ts_records*.yaml"))
    energy_files = sorted(case.glob("rxn/**/final_state_energy*.yaml"))
    adsorbates = list(case.glob("rxn/Adsorbates/*/*_opt.xyz"))

    manifest_path, manifest = latest_real_manifest(case)
    log_path = latest_log(case)
    log_mtime = log_path.stat().st_mtime if log_path else None
    log_age = now - log_mtime if log_mtime else None
    log_fresh = log_age is not None and log_age <= 300
    queued = manifest is None and has_any_manifest(case) and not log_fresh
    text = log_path.read_text(errors="replace") if log_path else ""
    valid_sp = [int(x) for x in re.findall(r"Valid sites for sp_(\d+):", text)]
    processing_rxn = re.findall(r"Processing (rxn_\S+) at site (\d+)", text)
    seed_overrides = len(re.findall(r"seed override:", text))

    return {
        "case": case.name,
        "case_dir": str(case),
        "prepared_file": str(prepared_files[0]) if prepared_files else None,
        "prepared_rxns": len(rxns),
        "prepared_species": species_n,
        "active_run": {
            "manifest": str(manifest_path) if manifest_path else None,
            "status": manifest.get("status") if manifest else ("QUEUED_DRYRUN" if queued else "RUNNING/NO_MANIFEST"),
            "failure_class": manifest.get("failure_class") if manifest else None,
            "ended_at": manifest.get("ended_at") if manifest else None,
            "wall_s": float(manifest["wall_s"]) if manifest and manifest.get("wall_s") else None,
            "device": manifest.get("device") if manifest else None,
            "model": manifest.get("model") if manifest else None,
            "head": manifest.get("head") if manifest else None,
        },
        "log": {
            "path": str(log_path) if log_path else None,
            "age_s": round(log_age, 1) if log_age is not None else None,
            "size_bytes": log_path.stat().st_size if log_path else None,
            "stage": (
                "completed"
                if manifest and manifest.get("status") == "SUCCESS"
                else infer_stage(text) if (text and log_fresh)
                else ("queued" if queued else ("unknown" if text else "no_log"))
            ),
            "last_line": text.splitlines()[-1] if text else None,
            "valid_site_species_seen": max(valid_sp) + 1 if valid_sp else 0,
            "processing_events": len(processing_rxn),
            "last_processing": processing_rxn[-1] if processing_rxn else None,
            "seed_override_events": seed_overrides,
        },
        "products": {
            "ts_record_files": len(ts_files),
            "energy_files": len(energy_files),
            "adsorbate_opt_files": len(adsorbates),
            "latest_product_mtime": max(
                [p.stat().st_mtime for p in ts_files + energy_files + adsorbates], default=None
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    now = time.time()
    cases = [collect_case(case, now) for case in case_roots(args.root)]
    stage_counts = Counter(x["log"]["stage"] for x in cases)
    result = {
        "schema": "recnet_campaign_progress/1",
        "generated_unix": now,
        "totals": {
            "cases": len(cases),
            "stages": dict(sorted(stage_counts.items())),
            "ts_record_files": sum(x["products"]["ts_record_files"] for x in cases),
            "energy_files": sum(x["products"]["energy_files"] for x in cases),
            "adsorbate_opt_files": sum(x["products"]["adsorbate_opt_files"] for x in cases),
        },
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}: cases={len(cases)} stages={dict(sorted(stage_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
