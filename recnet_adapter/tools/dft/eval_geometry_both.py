#!/usr/bin/env python3
"""Evaluate one geometry with selected models and compare against a reference.

Usage: eval_geometry_both.py <geometry.xyz|STRU> <reference.xyz>
Prints dE = E(geometry) - E(reference) per selected model at the same cell/composition.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recnet_adapter.runtime_config import add_runtime_arguments, runtime_from_args  # noqa: E402


def energy(model, head, atoms):
    from recnet_adapter.model_selection import create_calculator

    a = atoms.copy()
    a.calc = create_calculator(model, head)
    return float(a.get_potential_energy())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry")
    parser.add_argument("reference")
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    runtime = runtime_from_args(args)
    if not runtime.models:
        parser.error("select at least one model with --model, --model-s/m, or --project-config")
    from recnet_adapter.model_selection import validate_model_head
    for arm, model in runtime.models.items():
        validate_model_head(model, runtime.heads[arm])
    from ase.io import read
    import numpy as np

    geo, ref = args.geometry, args.reference
    src = Path(geo)
    try:
        atoms = read(src, format="abacus")
    except Exception:
        atoms = read(src)
    sym = atoms.get_chemical_symbols()
    ci = max(i for i, s in enumerate(sym) if s == "C")
    oi = max(i for i, s in enumerate(sym) if s == "O")
    d = float(np.linalg.norm(atoms.positions[ci] - atoms.positions[oi]))
    print(f"geometry: {src}  d(C-O) = {d:.3f} Å")
    ref_atoms = read(Path(ref))
    for arm, model in runtime.models.items():
        e_geo = energy(model, runtime.heads[arm], atoms)
        e_ref = energy(model, runtime.heads[arm], ref_atoms)
        print(f"{arm}: E(geom)={e_geo:.4f}  E(ref={Path(ref).name})={e_ref:.4f}  "
              f"dE = {e_geo - e_ref:+.3f} eV  ({'model also goes downhill' if e_geo < e_ref else 'model goes uphill'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
