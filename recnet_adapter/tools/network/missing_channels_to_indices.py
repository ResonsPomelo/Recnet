#!/usr/bin/env python3
"""Map a c2_gate missing-channel list back to reaction indices (2026-09-22).

Round-2 recipe for an under-covered network::

    missing_channels_to_indices.py --gate-json <report>/c2_gate.json \\
        --prepared <prepared yaml> --out missing_rxn_indices.txt
    build_campaign.py --prepared <prepared yaml> --rxn-index-file missing_rxn_indices.txt \\
        --top-x 4 --out <round2 dir> --template-case <case>

The gate reports missing channels as ``{reactant, product, broken_bond}`` (the
channel identity used throughout the pipeline); this script matches them against
a prepared dataset's reaction list, so the follow-up campaign covers exactly the
channels the gate found uncovered — no hand-copied index lists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def channel_key(record) -> tuple:
    return (
        str(record.get("reactant", "")).replace(" ", ""),
        str(record.get("product", "")).replace(" ", ""),
        tuple(int(i) for i in (record.get("broken_bond") or [])),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="gate missing channels -> reaction indices")
    ap.add_argument("--gate-json", required=True)
    ap.add_argument("--prepared", required=True, help="prepared yaml the campaign used")
    ap.add_argument("--out", required=True, help="index list file for build_campaign --rxn-index-file")
    ap.add_argument("--report", default=None, help="optional JSON of the match outcome")
    args = ap.parse_args(argv)

    gate = json.loads(Path(args.gate_json).read_text())
    missing = gate.get("missing_qc_pass_channels") or []
    payload = yaml.safe_load(Path(args.prepared).read_text())
    rxns = payload.get("rxns", [])
    index = {channel_key(r): i for i, r in enumerate(rxns)}

    indices, unmatched = [], []
    for record in missing:
        key = channel_key(record)
        if key in index:
            indices.append(index[key])
        else:
            unmatched.append(record)

    Path(args.out).write_text(",".join(str(i) for i in sorted(set(indices))) + "\n")
    print(f"missing={len(missing)} matched={len(indices)} unmatched={len(unmatched)}")
    for i in sorted(set(indices)):
        r = rxns[i]
        print(f"  idx {i}: {r['reactant']} -> {r['product']}")
    for record in unmatched:
        print(f"  [unmatched] {record}")
    if args.report:
        Path(args.report).write_text(json.dumps({
            "gate_json": args.gate_json,
            "prepared": args.prepared,
            "missing": len(missing),
            "matched_indices": sorted(set(indices)),
            "unmatched": unmatched,
        }, ensure_ascii=False, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
