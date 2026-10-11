#!/usr/bin/env python3
"""CO-dissociation thermodynamics on the model PES (v2, 2026-09-21).

Per (arm, campaign case): take the pipeline's relaxed adsorbate structures and
compare the *best* adsorption site of each species:

    dE = min_site E(C*) + min_site E(O*) - min_site E(CO*)

This is the standard thermodynamic driving force for CO* -> C* + O* on the same
slab/model; a strongly positive value explains why no first-order C-O saddle was
accepted by the pipeline QC.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recnet_adapter.runtime_config import add_runtime_arguments, runtime_from_args  # noqa: E402
from recnet_adapter import AdapterError  # noqa: E402


def eval_energy(model, path: Path, head=None):
    from ase.io import read
    from recnet_adapter.model_selection import create_calculator

    atoms = read(path)
    atoms.calc = create_calculator(model, head)
    return float(atoms.get_potential_energy())


def slab_energy(model, case: Path, vg: int, cache: dict, head=None):
    key = (str(case), vg)
    if key in cache:
        return cache[key]
    f = case / f"rxn/FS_energy/slab_opt_vg{vg}.xyz"
    if not f.exists():
        cands = sorted(case.glob(f"rxn/FS_energy/slab_opt_vg{vg}*.xyz"))
        f = cands[0] if cands else None
    if f is None:
        raise FileNotFoundError(f"slab reference missing for vg{vg} in {case}")
    e = eval_energy(model, f, head=head)
    cache[key] = e
    return e


def best_site(model, case: Path, prefix: str, head=None):
    best = None
    for d in sorted(case.glob(f"rxn/Adsorbates/{prefix}*")):
        for f in sorted(d.glob("*_opt.xyz")):
            m = re.match(r"(\d+)_vg(\d+)_opt\.xyz", f.name)
            if not m:
                continue
            try:
                e = eval_energy(model, f, head=head)
            except AdapterError:
                raise
            except Exception as exc:
                print(f"  ! {f.name}: {type(exc).__name__}: {exc}")
                continue
            if best is None or e < best[0]:
                best = (e, int(m.group(1)), int(m.group(2)), f)
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default="/tmp/co_thermo2")
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    runtime = runtime_from_args(args)
    if not runtime.models:
        parser.error("select at least one model with --model, --model-s/m, or --project-config")
    from recnet_adapter.model_selection import validate_model_head
    for arm, model in runtime.models.items():
        validate_model_head(model, runtime.heads[arm])
    cases_by_arm = {
        arm: [runtime.root / f"recnet-runs/campaign-c2/{arm}-c{i}" for i in range(4)]
        for arm in runtime.models
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for arm, cases in cases_by_arm.items():
        model = runtime.models[arm]
        for case in cases:
            if not case.is_dir():
                print(f"skip {case} (missing)")
                continue
            bc = best_site(model, case, "co_", head=runtime.heads[arm])
            bc_ = best_site(model, case, "c_", head=runtime.heads[arm])
            bo = best_site(model, case, "o_", head=runtime.heads[arm])
            if not (bc and bc_ and bo):
                print(f"{arm} {case.name}: incomplete (co={bool(bc)} c={bool(bc_)} o={bool(bo)})")
                continue
            cache = {}
            e_ads_co = bc[0] - slab_energy(model, case, bc[2], cache, head=runtime.heads[arm])
            e_ads_c = bc_[0] - slab_energy(model, case, bc_[2], cache, head=runtime.heads[arm])
            e_ads_o = bo[0] - slab_energy(model, case, bo[2], cache, head=runtime.heads[arm])
            de = e_ads_c + e_ads_o - e_ads_co
            rec = {"arm": arm, "case": case.name,
                   "E_ads_CO": e_ads_co, "site_CO": bc[1], "vg_CO": bc[2],
                   "E_ads_C": e_ads_c, "site_C": bc_[1], "vg_C": bc_[2],
                   "E_ads_O": e_ads_o, "site_O": bo[1], "vg_O": bo[2],
                   "dE_diss": de}
            results.append(rec)
            print(f"{arm} {case.name}: dE(C+O - CO) = {de:+.3f} eV "
                  f"(CO@site{bc[1]}/vg{bc[2]}, C@site{bc_[1]}, O@site{bo[1]})", flush=True)
    (out / "co_thermo.json").write_text(json.dumps({"results": results}, indent=1))
    if results:
        v = sorted(r["dE_diss"] for r in results)
        print(f"\nsummary: n={len(v)} min={v[0]:+.3f} median={v[len(v)//2]:+.3f} max={v[-1]:+.3f} eV")
    print("wrote", out / "co_thermo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
