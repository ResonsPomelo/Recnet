#!/usr/bin/env python3
"""Subset / chunk a RecNet prepared dataset (2026-09-21).

The reaction-network campaigns run one worker per *case*, and a case consumes a
whole ``prepared_rmg_data.yaml``.  To parallelise a large network over several
GPUs (and to run small pilots) we need faithful subsets of that file that keep
the exact schema the pipeline expects.

Usage::

    subset_prepared.py --src <prepared_dir|prepared_rmg_data.yaml> --out <dir> \
        (--rxn-index 0,3,7 | --match CO* | --chunk 2/4 | --all) [--dry-run]

Writes ``<out>/prepared_rmg_data.yaml`` plus the ``ads_templates/*.xyz`` that are
actually referenced.  ``--chunk k/K`` splits the reaction list into K contiguous
chunks (k = 1..K) with stable ordering so parallel workers cover disjoint parts.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def load_prepared(src: Path):
    if src.is_file():
        yml = src
    else:
        yml = src / "prepared_rmg_data.yaml"
        if not yml.exists():
            # campaign inputs may use a descriptive name (e.g.
            # fe5c2_510_c2.prepared_rmg_data.yaml); accept a unique match.
            cands = sorted(src.glob("*prepared_rmg_data.yaml"))
            if len(cands) == 1:
                yml = cands[0]
            elif len(cands) > 1:
                raise SystemExit(f"ambiguous prepared yaml in {src}: {[c.name for c in cands]}")
    if not yml.exists():
        raise SystemExit(f"prepared yaml not found: {yml}")
    payload = yaml.safe_load(yml.read_text())
    return yml, payload


def select(rxns, args):
    n = len(rxns)
    if args.all:
        return list(range(n))
    if args.rxn_index:
        idx = []
        for part in args.rxn_index.replace(",", " ").split():
            i = int(part)
            if not 0 <= i < n:
                raise SystemExit(f"rxn-index {i} out of range (n={n})")
            idx.append(i)
        return sorted(set(idx))
    if args.match:
        keys = [k.lower() for k in args.match]
        out = []
        for i, rxn in enumerate(rxns):
            blob = " ".join(str(rxn.get(f, "")) for f in ("reactant", "product")).lower()
            if any(k in blob for k in keys):
                out.append(i)
        if not out:
            raise SystemExit(f"no reaction matched {args.match!r} (n={n})")
        return out
    if args.chunk:
        k, K = args.chunk
        if not 1 <= k <= K:
            raise SystemExit("--chunk expects k/K with 1<=k<=K")
        bounds = [round(i * n / K) for i in range(K + 1)]
        return list(range(bounds[k - 1], bounds[k]))
    raise SystemExit("one of --rxn-index / --match / --chunk / --all is required")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rxn-index")
    ap.add_argument("--match", action="append")
    ap.add_argument("--chunk", type=lambda s: tuple(int(x) for x in s.split("/")), metavar="k/K")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    src_root = Path(args.src).resolve()
    yml, payload = load_prepared(src_root)
    rxns, species = payload["rxns"], payload["species"]
    keep = select(rxns, args)

    used_species = []
    for i in keep:
        for field in ("reactant_species", "product_species"):
            for sp in rxns[i].get(field, []):
                if sp not in used_species:
                    used_species.append(sp)

    if not keep:
        # An empty prepared dataset would fail deep inside the pipeline; make the
        # empty chunk an explicit, early error unless it is a dry run/inspection.
        raise SystemExit(f"empty selection (n_rxns={len(rxns)}); adjust --chunk/--rxn-index/--match")

    print(f"src={yml}")
    print(f"reactions: {len(rxns)} -> {len(keep)} (indices {keep[:8]}{'...' if len(keep) > 8 else ''})")
    print(f"species  : {len(species)} -> {len(used_species)} ({used_species})")

    out = Path(args.out).resolve()
    new_payload = {"rxns": [rxns[i] for i in keep],
                   "species": {sp: species[sp] for sp in used_species}}
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0

    (out / "ads_templates").mkdir(parents=True, exist_ok=True)
    base = yml.parent
    copied = 0
    for sp in used_species:
        rec = species[sp]
        tpl = base / rec["template_xyz"]
        if not tpl.exists():
            raise SystemExit(f"template missing: {tpl}")
        dst = out / rec["template_xyz"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tpl, dst)
        copied += 1
    (out / "prepared_rmg_data.yaml").write_text(
        yaml.safe_dump(new_payload, sort_keys=False, allow_unicode=True))
    print(f"wrote {out/'prepared_rmg_data.yaml'}; templates copied: {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
