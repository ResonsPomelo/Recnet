#!/usr/bin/env python3
"""Audit whether channels missing from the C2 gate were attempted."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def channel_key(rxn: dict[str, Any]):
    bb = rxn.get("broken_bond") or []
    return (str(rxn.get("reactant")), str(rxn.get("product")), tuple(int(x) for x in bb))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-root", action="append", required=True)
    ap.add_argument("--gate-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gate = json.loads(Path(args.gate_json).read_text())
    missing = gate.get("missing_qc_pass_channels") or []
    wanted = {
        (x["reactant"], x["product"], tuple(int(z) for z in x["broken_bond"]))
        for x in missing
    }

    items = []
    for root in args.campaign_root:
        root_path = Path(root).resolve()
        for case in sorted({p.parent.parent for p in root_path.rglob("prepared_data/*prepared_rmg_data.yaml")}):
            prepared = next((case / "prepared_data").glob("*prepared_rmg_data.yaml"), None)
            if prepared is None:
                continue
            payload = yaml.safe_load(prepared.read_text()) or {}
            logs = list(case.glob("farm/runs/*/*/case.log"))
            log = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
            text = log.read_text(errors="replace") if log else ""
            for idx, rxn in enumerate(payload.get("rxns") or []):
                key = channel_key(rxn)
                if key not in wanted:
                    continue
                rid = f"rxn_{idx}"
                processing = len(__import__("re").findall(rf"Processing {rid} at site ", text))
                rejected = len(__import__("re").findall(rf"All orientations rejected for {rid}_site_", text))
                no_accept = len(__import__("re").findall(rf"No TS accepted on preferred top-x sites for {rid}", text))
                if processing and rejected:
                    status = "attempted_rejected"
                elif processing:
                    status = "attempted"
                else:
                    status = "pending"
                items.append({
                    "case": case.name,
                    "rxn_idx": idx,
                    "reactant": key[0],
                    "product": key[1],
                    "broken_bond": list(key[2]),
                    "processing_events": processing,
                    "rejected_events": rejected,
                    "no_accept_statements": no_accept,
                    "log": str(log) if log else None,
                    "status": status,
                })

    counts: dict[str, int] = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    result = {
        "schema": "c2_missing_attempt_audit/1",
        "missing_channels": len(wanted),
        "items": items,
        "status_counts": dict(sorted(counts.items())),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}: missing={len(wanted)} items={len(items)} statuses={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
