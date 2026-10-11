#!/usr/bin/env python3
"""Export a per-species relaxed-adsorbate library from the campaign case roots (2026-09-25, R204).

The FTS-module data need "D4" (`docs/reports/2026-09-25-fts-module-data-and-interface-needs.md`) is a
curated structure set for every species of the closure: the pipeline already relaxes each species on
every enumerated site/vacancy group and writes ``rxn/Adsorbates/<species_key>/<site>_vg<k>_opt.xyz``
(extxyz with an embedded ``energy=`` in the comment line — the platform's preferred format).

This tool collects them into a library:

  * one row per (species, arm, case, site, vg) with the absolute model energy and its **relative**
    energy with respect to the best site of the same (species, arm, case) — relative values are
    convention-free (same slab, same model), so they are directly usable without a gas reference;
  * the best structure per (species, arm) is copied into ``<out>/structures/`` with a stable name;
  * ``adsorbate_library.json`` + ``README.md`` carry the provenance (case, source file, sha256) and the
    explicit note that **absolute adsorption energies require the case's slab + gas references**
    (pointer: ``rxn/FS_energy/`` and ``channel_thermo.py``), which this tool deliberately does not
    re-derive.

Usage::

    export_adsorbate_library.py --root <case_root> [--root …] --out <dir> [--dataset <yaml>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import yaml

KEY_RE = re.compile(r"^(?P<name>.+)_(?P<hash>[0-9a-f]{12})$")


def read_extxyz_energy(path: Path) -> float | None:
    """Energy from the extxyz comment line (``… energy=<float> …``), else None."""
    try:
        with path.open() as fh:
            fh.readline()
            comment = fh.readline()
    except OSError:
        return None
    m = re.search(r"(?:^|\s)energy=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", comment)
    return float(m.group(1)) if m else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export a per-species relaxed-adsorbate library")
    ap.add_argument("--root", action="append", required=True, help="campaign case root (repeatable)")
    ap.add_argument("--primary-root", action="append", default=None,
                    help="roots the best-structure pick may come from (repeatable). Defaults to "
                         "--root; use it to keep the pick on the main campaigns when scanning "
                         "older pilot/targeted roots for extra coverage (R210).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=None, help="prepared yaml (maps species_key -> display name)")
    ap.add_argument("--copy-best", action="store_true", help="also copy the best structure per species/arm")
    args = ap.parse_args(argv)

    display: dict[str, str] = {}
    if args.dataset and Path(args.dataset).exists():
        for key, rec in (yaml.safe_load(Path(args.dataset).read_text()) or {}).get("species", {}).items():
            display[str(rec.get("name", key)).lower()] = str(rec.get("name", key))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_files: set[Path] = set()
    for root in args.root:
        for case in sorted(Path(root).glob("*")):
            ads = case / "rxn" / "Adsorbates"
            if not ads.is_dir():
                continue
            arm = "S" if case.name.startswith("S-") else "M" if case.name.startswith("M-") else "?"
            for sp_dir in sorted(p for p in ads.iterdir() if p.is_dir()):
                m = KEY_RE.match(sp_dir.name)
                name_l = m.group("name") if m else sp_dir.name
                name = display.get(name_l, name_l)
                if name == name_l:
                    # R210: species dirs may carry a disambiguating numeric suffix that the dataset
                    # name does not have (e.g. `c2h4_1_9f6f2862e3cb` ↔ dataset name `C2H4`) — strip
                    # trailing `_<digits>` one at a time and retry the display lookup.
                    cand = name_l
                    while name == name_l and "_" in cand:
                        cand = re.sub(r"_\d+$", "", cand)
                        if cand == name_l:
                            break
                        if cand in display:
                            name = display[cand]
                for f in sorted(sp_dir.glob("*_opt.xyz")):
                    if f in seen_files:
                        continue
                    seen_files.add(f)
                    stem = f.stem[: -len("_opt")]
                    site, _, vg = stem.partition("_vg")
                    rows.append({
                        "species": name,
                        "species_key": sp_dir.name,
                        "arm": arm,
                        "case": str(case),
                        "site": int(site) if site.isdigit() else site,
                        "vg": vg,
                        "energy_eV": read_extxyz_energy(f),
                        "file": str(f),
                    })

    # relative energy with respect to the best site of the same (species, arm, case)
    best_by_group: dict[tuple, float] = {}
    for r in rows:
        if r["energy_eV"] is None:
            continue
        key = (r["species"], r["arm"], r["case"])
        best_by_group[key] = min(best_by_group.get(key, r["energy_eV"]), r["energy_eV"])
    for r in rows:
        b = best_by_group.get((r["species"], r["arm"], r["case"]))
        r["rel_energy_eV"] = None if (r["energy_eV"] is None or b is None) else round(r["energy_eV"] - b, 4)

    primary = {str(Path(p)) for p in (args.primary_root or args.root)}
    best: dict[tuple, dict] = {}
    for r in rows:
        if r["energy_eV"] is None:
            continue
        if not any(r["case"].startswith(p) for p in primary):
            continue                      # selection stays on the main campaigns (R210)
        key = (r["species"], r["arm"])
        if key not in best or r["energy_eV"] < best[key]["energy_eV"]:
            best[key] = r
    for r in best.values():
        r["sha256"] = hashlib.sha256(Path(r["file"]).read_bytes()).hexdigest()
        if args.copy_best:
            dst_dir = out / "structures" / r["species"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{r['arm']}_site{r['site']}_vg{r['vg']}.xyz"
            shutil.copyfile(r["file"], dst)
            r["library_file"] = str(dst)

    payload = {
        "schema": "adsorbate-library-v1",
        "roots": args.root,
        "n_rows": len(rows),
        "n_species": len({r["species"] for r in rows}),
        "n_missing_energy": sum(1 for r in rows if r["energy_eV"] is None),
        "best": [best[k] for k in sorted(best)],
        "rows": rows,
        "note": ("Absolute adsorption energies are NOT derived here: they need the case's relaxed "
                 "slab energy plus the gas-phase references (see `rxn/FS_energy/` and "
                 "`Recnet/recnet_adapter/tools/channel_thermo.py`). `rel_energy_eV` is the "
                 "convention-free within-(species, arm, case) offset to the best site. Energies are "
                 "model energies of the two FT²DP-v2.2 arms and must not be mixed across arms."),
    }
    (out / "adsorbate_library.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))

    md = ["# 吸附物结构库（per-species relaxed adsorbates）", "",
          f"- 来源 roots：{', '.join(args.root)}",
          f"- 行数 **{len(rows)}**（{payload['n_species']} 物种；缺能量 {payload['n_missing_energy']} 行）",
          f"- 每 (物种, 臂) 最优结构 **{len(best)}** 条（见 `adsorbate_library.json` 的 `best`）",
          "",
          "> **口径**：`energy_eV` = 该结构在**同一条板**上的模型绝对能量（extxyz 注释内 `energy=`）；",
          "> `rel_energy_eV` = 相对同 (物种, 臂, case) 最优位点的偏移（**与参考无关**，可直接用）；",
          "> **绝对吸附能**需叠加该 case 的弛豫板能量与气相参考（见 `rxn/FS_energy/` 与 `channel_thermo.py`），",
          "> 本库**刻意不代算**以免引入错误参考；两臂能量**不可混比**。",
          "",
          "| 物种 | 臂 | case | site | vg | E (eV) | 相对最优 (eV) |", "|---|---|---|---|---|---|---|"]
    for r in payload["best"]:
        md.append(f"| `{r['species']}` | {r['arm']} | {Path(r['case']).name} | {r['site']} | {r['vg']} | "
                  f"{r['energy_eV']:.4f} | {r['rel_energy_eV']} |")
    (out / "README.md").write_text("\n".join(md) + "\n")
    print(f"[adsorbate-library] rows={len(rows)} species={payload['n_species']} best={len(best)} "
          f"-> {out}/adsorbate_library.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
