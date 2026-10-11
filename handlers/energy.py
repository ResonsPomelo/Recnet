"""
Energy evaluation stage: compute final-state energies (IS, TS, FS) and
adsorption energies with optional ZPE / free-energy corrections.
"""
import hashlib
import os
import shutil

import numpy as np
import yaml
from ase.io import read, write
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, IdealGasThermo

from deepmd.calculator import DP

from .context import WorkflowContext, MODEL
from utils import config as cfgmod


class EnergyHandler:
    """Evaluates final-state energies and adsorption energies."""

    def __init__(self, ctx: WorkflowContext):
        self.ctx = ctx

    def run_final_state_energy(self):
        return get_final_state_energy(self.ctx)

    def run_adsorption_energy(self):
        return get_ads_energy(self.ctx)


def get_final_state_energy(ctx: WorkflowContext):
    """Evaluate IS, TS, FS energies with slab subtraction and optional thermo corrections."""
    records = ctx._load_ts_records()
    if len(records) == 0:
        print("No TS records found; skip final-state energy evaluation.")
        return []

    out_root = os.path.join(ctx.path, "rxn", "FS_energy")
    os.makedirs(out_root, exist_ok=True)
    vib_root = os.path.join(out_root, "vib_jobs")
    if ctx.enable_thermo_corrections:
        os.makedirs(vib_root, exist_ok=True)
    temperature_K = ctx.temperature

    if not ctx.enable_thermo_corrections:
        print("Thermo corrections are disabled; all ZPE/G corrections will be 0.")

    energy_cache = {}
    vib_corr_cache = {}

    def _energy_from_xyz(xyz_path):
        xyz_path = os.path.abspath(xyz_path)
        if xyz_path in energy_cache:
            return energy_cache[xyz_path]
        atoms = read(xyz_path)
        atoms.calc = DP(model=MODEL)
        atoms.set_constraint(ctx._build_constraints(atoms, with_internal_bonds=False))
        e = float(atoms.get_potential_energy())
        energy_cache[xyz_path] = e
        return e

    def _energy_from_gas_species(sp_id):
        cache_key = f"gas::{sp_id}"
        if cache_key in energy_cache:
            return energy_cache[cache_key]
        if sp_id not in ctx.ads_templates:
            raise KeyError(f"Unknown species id for gas energy: {sp_id}")
        gas_atoms = ctx.ads_templates[sp_id]["atoms"].copy()
        gas_atoms.set_cell([20.0, 20.0, 20.0])
        gas_atoms.set_pbc([False, False, False])
        gas_atoms.center()
        gas_atoms.calc = DP(model=MODEL)
        e = float(gas_atoms.get_potential_energy())
        energy_cache[cache_key] = e
        return e

    def _vib_corrections_for_structure(xyz_path):
        cache_key = os.path.abspath(xyz_path)
        if cache_key in vib_corr_cache:
            return vib_corr_cache[cache_key]
        if not ctx.enable_thermo_corrections:
            out = {"zpe": 0.0, "g_corr": 0.0, "used_prefix": 0, "has_vib": False}
            vib_corr_cache[cache_key] = out
            return out
        atoms = read(xyz_path)
        atoms.calc = DP(model=MODEL)
        atoms.set_constraint(ctx._build_constraints(atoms, with_internal_bonds=False))
        digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
        base = os.path.splitext(os.path.basename(cache_key))[0]
        vib_name = os.path.join(vib_root, f"{base}_{digest}")
        g_corr, zpe, used_prefix = ctx._thermo_analysis(atoms, temperature_K, name=vib_name)
        out = {"zpe": float(zpe), "g_corr": float(g_corr), "used_prefix": used_prefix, "has_vib": bool(used_prefix != 0)}
        vib_corr_cache[cache_key] = out
        return out

    def _find_adsorbate_energy_file(sp_id, site):
        stem_tagged = ctx._tagged_stem(f"{site}")
        for ads_dir in ctx._adsorbate_dir_candidates(sp_id):
            candidates = [
                os.path.join(ads_dir, f"{stem_tagged}_opt.xyz"),
                os.path.join(ads_dir, f"{site}_opt.xyz"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
        return None

    def _find_adsorbate_energy_file_global_min(sp_id):
        valid_sites = ctx.valid_sites.get(sp_id, []) if hasattr(ctx, "valid_sites") else []
        if len(valid_sites) > 0:
            best_path = _find_adsorbate_energy_file(sp_id, valid_sites[0])
            if best_path is not None:
                return best_path
        opt_files = []
        for ads_dir in ctx._adsorbate_dir_candidates(sp_id):
            if not os.path.isdir(ads_dir):
                continue
            opt_files.extend([
                os.path.join(ads_dir, name)
                for name in os.listdir(ads_dir)
                if name.endswith("_opt.xyz")
            ])
        if len(opt_files) == 0:
            return None
        best_path = None
        best_energy = None
        for xyz_path in opt_files:
            intact, reason = ctx._is_adsorbate_structure_intact(xyz_path, sp_id)
            try:
                e_ads = _energy_from_xyz(xyz_path)
            except Exception:
                continue
            if (best_energy is None) or (e_ads < best_energy):
                best_energy = e_ads
                best_path = xyz_path
        return best_path

    def _collect_state_items(species_ids, site, tag, state_name, far_separated=False):
        items = []
        for sp_id in species_ids:
            if far_separated:
                xyz_path = _find_adsorbate_energy_file_global_min(sp_id)
            else:
                xyz_path = _find_adsorbate_energy_file(sp_id, site)
            if xyz_path is None:
                msg = f"Skip {tag}: missing {state_name} adsorbate file for {sp_id}"
                print(msg)
                return None
            try:
                e_ads = _energy_from_xyz(xyz_path)
            except Exception as exc:
                print(f"Skip {tag}: failed to evaluate {xyz_path}: {exc}")
                return None
            vib_corr = _vib_corrections_for_structure(xyz_path)
            items.append({
                "species": sp_id,
                "xyz": os.path.abspath(xyz_path),
                "energy": float(e_ads),
                "zpe_correction": float(vib_corr["zpe"]),
                "g_correction": float(vib_corr["g_corr"]),
                "vib_source": vib_corr["used_prefix"],
            })
        return items

    def _state_energy_from_items(items, e_slab_local):
        n_items = len(items)
        e_sum = float(sum(x["energy"] for x in items))
        return float(e_sum - max(0, n_items - 1) * e_slab_local)

    def _state_correction_from_items(items, key, slab_corr=0.0):
        n_items = len(items)
        corr_sum = float(sum(float(x.get(key, 0.0)) for x in items))
        return float(corr_sum - max(0, n_items - 1) * float(slab_corr))

    # ---- Slab optimization ----
    slab_opt_xyz = os.path.join(out_root, ctx._tagged_stem("slab_opt") + ".xyz")
    if os.path.exists(slab_opt_xyz):
        slab_atoms = read(slab_opt_xyz)
        slab_atoms.calc = DP(model=MODEL)
        slab_atoms.set_constraint(ctx._build_constraints(slab_atoms, with_internal_bonds=False))
        e_slab = float(slab_atoms.get_potential_energy())
        print(f"Reuse optimized slab: {slab_opt_xyz}")
    else:
        slab_atoms = ctx.slab.stru.copy()
        slab_atoms.calc = DP(model=MODEL)
        slab_atoms.set_constraint(ctx._build_constraints(slab_atoms, with_internal_bonds=False))
        slab_optimizer = QuasiNewton(slab_atoms)
        slab_optimizer.run(fmax=0.05, steps=100)
        e_slab = float(slab_atoms.get_potential_energy())
        write(slab_opt_xyz, slab_atoms)

    slab_xyz_for_vib = os.path.join(out_root, ctx._tagged_stem("slab_for_vib") + ".xyz")
    write(slab_xyz_for_vib, slab_atoms)
    slab_vib_corr = _vib_corrections_for_structure(slab_xyz_for_vib)

    results = []
    seen_tags = set()
    for rec in records:
        tag = rec.get("tag", f"rxn_{rec['rxn_idx']}_site_{rec['site']}")
        if tag in seen_tags:
            continue
        seen_tags.add(tag)

        rxn_idx = ctx._resolve_record_rxn_index(rec)
        site = int(rec["site"])
        if rxn_idx is None or rxn_idx < 0 or rxn_idx >= len(ctx.rxns_dict):
            print(f"Skip {tag}: cannot map record to current reaction list")
            continue

        rxn = ctx.rxns_dict[rxn_idx]
        reactant_species = list(rxn.get("reactant_species", []))
        product_species = list(rxn.get("product_species", []))
        if len(reactant_species) == 0:
            print(f"Skip {tag}: reactant_species is empty")
            continue
        if len(product_species) < 2:
            print(f"Skip {tag}: product_species count < 2 ({product_species})")
            continue

        is_items = _collect_state_items(reactant_species, site, tag, "IS", far_separated=True)
        if is_items is None:
            continue

        fs_items = _collect_state_items(product_species[:2], site, tag, "FS", far_separated=True)
        if fs_items is None:
            continue

        ts_xyz_path = rec.get("ts_xyz")
        if not ts_xyz_path or (not os.path.exists(ts_xyz_path)):
            print(f"Skip {tag}: missing TS xyz file: {ts_xyz_path}")
            continue
        try:
            e_ts = _energy_from_xyz(ts_xyz_path)
        except Exception as exc:
            print(f"Skip {tag}: failed to evaluate TS energy from {ts_xyz_path}: {exc}")
            continue

        ts_vib_corr = _vib_corrections_for_structure(ts_xyz_path)

        e_is = _state_energy_from_items(is_items, e_slab)
        e_fs = _state_energy_from_items(fs_items, e_slab)
        e_a_is = float(e_ts - e_is)
        e_a_fs = float(e_ts - e_fs)
        e_fs_is = float(e_fs - e_is)

        zpe_is_corr = _state_correction_from_items(is_items, "zpe_correction", slab_corr=slab_vib_corr["zpe"])
        zpe_fs_corr = _state_correction_from_items(fs_items, "zpe_correction", slab_corr=slab_vib_corr["zpe"])
        zpe_ts_corr = float(ts_vib_corr["zpe"])

        g_is_corr = _state_correction_from_items(is_items, "g_correction", slab_corr=slab_vib_corr["g_corr"])
        g_fs_corr = _state_correction_from_items(fs_items, "g_correction", slab_corr=slab_vib_corr["g_corr"])
        g_ts_corr = float(ts_vib_corr["g_corr"])

        e_is_zpe = float(e_is + zpe_is_corr)
        e_ts_zpe = float(e_ts + zpe_ts_corr)
        e_fs_zpe = float(e_fs + zpe_fs_corr)

        g_is = float(e_is + g_is_corr)
        g_ts = float(e_ts + g_ts_corr)
        g_fs = float(e_fs + g_fs_corr)

        barrier_ts_minus_is_zpe = float(e_ts_zpe - e_is_zpe)
        barrier_ts_minus_fs_zpe = float(e_ts_zpe - e_fs_zpe)
        delta_e_fs_minus_is_zpe = float(e_fs_zpe - e_is_zpe)

        barrier_g_ts_minus_is = float(g_ts - g_is)
        barrier_g_ts_minus_fs = float(g_ts - g_fs)
        delta_g_fs_minus_is = float(g_fs - g_is)

        g_zpe_is = float(e_is + zpe_is_corr + g_is_corr)
        g_zpe_ts = float(e_ts + zpe_ts_corr + g_ts_corr)
        g_zpe_fs = float(e_fs + zpe_fs_corr + g_fs_corr)
        barrier_g_zpe_ts_minus_is = float(g_zpe_ts - g_zpe_is)
        barrier_g_zpe_ts_minus_fs = float(g_zpe_ts - g_zpe_fs)
        delta_g_zpe_fs_minus_is = float(g_zpe_fs - g_zpe_is)

        row = {
            "tag": tag,
            "rxn_idx": rxn_idx,
            "site": site,
            "reactants": is_items,
            "products": fs_items,
            "temperature_K": float(temperature_K),
            "e_slab": e_slab,
            "e_is": e_is,
            "e_ts": float(e_ts),
            "e_fs": e_fs,
            "barrier_ts_minus_is": e_a_is,
            "barrier_ts_minus_fs": e_a_fs,
            "delta_e_fs_minus_is": e_fs_is,
            "zpe_correction_is": zpe_is_corr,
            "zpe_correction_ts": zpe_ts_corr,
            "zpe_correction_fs": zpe_fs_corr,
            "e_is_zpe": e_is_zpe,
            "e_ts_zpe": e_ts_zpe,
            "e_fs_zpe": e_fs_zpe,
            "barrier_ts_minus_is_zpe": barrier_ts_minus_is_zpe,
            "barrier_ts_minus_fs_zpe": barrier_ts_minus_fs_zpe,
            "delta_e_fs_minus_is_zpe": delta_e_fs_minus_is_zpe,
            "g_correction_is": g_is_corr,
            "g_correction_ts": g_ts_corr,
            "g_correction_fs": g_fs_corr,
            "g_is": g_is,
            "g_ts": g_ts,
            "g_fs": g_fs,
            "barrier_g_ts_minus_is": barrier_g_ts_minus_is,
            "barrier_g_ts_minus_fs": barrier_g_ts_minus_fs,
            "delta_g_fs_minus_is": delta_g_fs_minus_is,
            "g_zpe_is": g_zpe_is,
            "g_zpe_ts": g_zpe_ts,
            "g_zpe_fs": g_zpe_fs,
            "barrier_g_zpe_ts_minus_is": barrier_g_zpe_ts_minus_is,
            "barrier_g_zpe_ts_minus_fs": barrier_g_zpe_ts_minus_fs,
            "delta_g_zpe_fs_minus_is": delta_g_zpe_fs_minus_is,
            "slab_zpe_correction": float(slab_vib_corr["zpe"]),
            "slab_g_correction": float(slab_vib_corr["g_corr"]),
            "slab_vib_source": slab_vib_corr["used_prefix"],
            "ts_vib_source": ts_vib_corr["used_prefix"],
            "ts_xyz": os.path.abspath(ts_xyz_path),
        }
        results.append(row)

    results_path = os.path.join(out_root, ctx._tagged_stem("final_state_energy") + ".yaml")
    with open(results_path, "w") as f:
        yaml.safe_dump({"final_state_energy": results}, f, sort_keys=False, allow_unicode=True)

    print(f"Final-state energy evaluation finished: {len(results)} cases. Summary: {results_path}")
    return results


def get_ads_energy(ctx: WorkflowContext):
    """Evaluate adsorption energies for gas-capable species."""
    out_root = os.path.join(ctx.path, "rxn", "FS_energy")
    os.makedirs(out_root, exist_ok=True)
    vib_root = os.path.join(out_root, "vib_jobs")
    if ctx.enable_thermo_corrections:
        os.makedirs(vib_root, exist_ok=True)
    temperature_K = ctx.temperature

    if not ctx.enable_thermo_corrections:
        print("Thermo corrections are disabled; all ZPE/G corrections will be 0.")

    energy_cache = {}
    vib_corr_cache = {}

    def _energy_from_xyz(xyz_path):
        xyz_path = os.path.abspath(xyz_path)
        if xyz_path in energy_cache:
            return energy_cache[xyz_path]
        atoms = read(xyz_path)
        atoms.calc = DP(model=MODEL)
        atoms.set_constraint(ctx._build_constraints(atoms, with_internal_bonds=False))
        e = float(atoms.get_potential_energy())
        energy_cache[xyz_path] = e
        return e

    def _energy_from_gas_species(sp_id):
        cache_key = f"gas::{sp_id}"
        if cache_key in energy_cache:
            return energy_cache[cache_key]
        if sp_id not in ctx.ads_templates:
            raise KeyError(f"Unknown species id for gas energy: {sp_id}")
        gas_atoms = ctx.ads_templates[sp_id]["atoms"].copy()
        gas_atoms.set_cell([20.0, 20.0, 20.0])
        gas_atoms.set_pbc([False, False, False])
        gas_atoms.center()
        gas_atoms.calc = DP(model=MODEL)
        e = float(gas_atoms.get_potential_energy())
        energy_cache[cache_key] = e
        return e

    def _vib_corrections_for_structure(xyz_path):
        cache_key = os.path.abspath(xyz_path)
        if cache_key in vib_corr_cache:
            return vib_corr_cache[cache_key]
        if not ctx.enable_thermo_corrections:
            out = {"zpe": 0.0, "g_corr": 0.0, "used_prefix": 0, "has_vib": False}
            vib_corr_cache[cache_key] = out
            return out
        atoms = read(xyz_path)
        atoms.calc = DP(model=MODEL)
        atoms.set_constraint(ctx._build_constraints(atoms, with_internal_bonds=False))
        digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
        base = os.path.splitext(os.path.basename(cache_key))[0]
        vib_name = os.path.join(vib_root, f"{base}_{digest}")
        g_corr, zpe, used_prefix = ctx._thermo_analysis(atoms, temperature_K, name=vib_name)
        out = {"zpe": float(zpe), "g_corr": float(g_corr), "used_prefix": used_prefix, "has_vib": bool(used_prefix != 0)}
        vib_corr_cache[cache_key] = out
        return out

    def _vib_corrections_for_gas_species(sp_id):
        cache_key = f"gas::{sp_id}"
        if cache_key in vib_corr_cache:
            return vib_corr_cache[cache_key]
        if not ctx.enable_thermo_corrections:
            out = {"zpe": 0.0, "g_corr": 0.0, "used_prefix": 0, "has_vib": False}
            vib_corr_cache[cache_key] = out
            return out
        gas_atoms = ctx.ads_templates[sp_id]["atoms"].copy()
        gas_atoms.set_cell([20.0, 20.0, 20.0])
        gas_atoms.set_pbc([False, False, False])
        gas_atoms.center()
        gas_atoms.calc = DP(model=MODEL)
        digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
        vib_name = os.path.join(vib_root, f"gas_{sp_id}_{digest}")
        g_corr, zpe, used_prefix = _gas_ideal_thermo_corrections(
            ctx, gas_atoms, temperature_K=temperature_K,
            pressure_pa=ctx.gas_pressure_pa, name=vib_name,
            indices=list(range(len(gas_atoms))),
        )
        out = {"zpe": float(zpe), "g_corr": float(g_corr), "used_prefix": used_prefix, "has_vib": bool(used_prefix != 0)}
        vib_corr_cache[cache_key] = out
        return out

    def _find_adsorbate_energy_file(sp_id, site):
        stem_tagged = ctx._tagged_stem(f"{site}")
        for ads_dir in ctx._adsorbate_dir_candidates(sp_id):
            candidates = [
                os.path.join(ads_dir, f"{stem_tagged}_opt.xyz"),
                os.path.join(ads_dir, f"{site}_opt.xyz"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
        return None

    def _find_adsorbate_energy_file_global_min(sp_id):
        valid_sites = ctx.valid_sites.get(sp_id, []) if hasattr(ctx, "valid_sites") else []
        if len(valid_sites) > 0:
            best_path = _find_adsorbate_energy_file(sp_id, valid_sites[0])
            if best_path is not None:
                return best_path
        opt_files = []
        for ads_dir in ctx._adsorbate_dir_candidates(sp_id):
            if not os.path.isdir(ads_dir):
                continue
            opt_files.extend([
                os.path.join(ads_dir, name)
                for name in os.listdir(ads_dir)
                if name.endswith("_opt.xyz")
            ])
        if len(opt_files) == 0:
            return None
        best_path = None
        best_energy = None
        for xyz_path in opt_files:
            intact, reason = ctx._is_adsorbate_structure_intact(xyz_path, sp_id)
            try:
                e_ads = _energy_from_xyz(xyz_path)
            except Exception:
                continue
            if (best_energy is None) or (e_ads < best_energy):
                best_energy = e_ads
                best_path = xyz_path
        return best_path

    # ---- Slab energy ----
    slab_opt_xyz = os.path.join(out_root, ctx._tagged_stem("slab_opt") + ".xyz")
    if os.path.exists(slab_opt_xyz):
        slab_atoms = read(slab_opt_xyz)
        slab_atoms.calc = DP(model=MODEL)
        slab_atoms.set_constraint(ctx._build_constraints(slab_atoms, with_internal_bonds=False))
        e_slab = float(slab_atoms.get_potential_energy())
    else:
        slab_atoms = ctx.slab.stru.copy()
        slab_atoms.calc = DP(model=MODEL)
        slab_atoms.set_constraint(ctx._build_constraints(slab_atoms, with_internal_bonds=False))
        slab_optimizer = QuasiNewton(slab_atoms)
        slab_optimizer.run(fmax=0.05, steps=100)
        e_slab = float(slab_atoms.get_potential_energy())
        write(slab_opt_xyz, slab_atoms)

    slab_xyz_for_vib = os.path.join(out_root, ctx._tagged_stem("slab_for_vib") + ".xyz")
    write(slab_xyz_for_vib, slab_atoms)
    slab_vib_corr = _vib_corrections_for_structure(slab_xyz_for_vib)

    # ---- Gas-capable species list ----
    gas_capable_species = []
    if ctx.gas_species_whitelist is not None and len(ctx.gas_species_whitelist) > 0:
        for sp_id in ctx.gas_species_whitelist:
            if sp_id not in ctx.ads_templates:
                print(f"Skip whitelist species {sp_id}: not found in ads templates")
                continue
            gas_capable_species.append(sp_id)

    for sp_id, info in ctx.ads_templates.items():
        if len(info.get("ad_idx", [])) == 0 and sp_id not in gas_capable_species:
            gas_capable_species.append(sp_id)
    for sp_id, info in ctx.ads_templates.items():
        sp_name = str(info.get("name", "")).strip().upper()
        sp_key = str(sp_id).strip().upper()
        if (sp_key == "CO" or sp_name == "CO") and sp_id not in gas_capable_species:
            gas_capable_species.append(sp_id)

    gas_capable_species = cfgmod._dedupe_keep_order(gas_capable_species)

    # ---- Compute adsorption energies ----
    adsorption_results = []
    for sp_id in gas_capable_species:
        ads_xyz_path = _find_adsorbate_energy_file_global_min(sp_id)
        if ads_xyz_path is None:
            print(f"Skip adsorption energy for {sp_id}: no optimized adsorbate structure found")
            continue
        try:
            e_ads_on_slab = _energy_from_xyz(ads_xyz_path)
            e_gas = _energy_from_gas_species(sp_id)
        except Exception as exc:
            print(f"Skip adsorption energy for {sp_id}: energy evaluation failed: {exc}")
            continue

        ads_vib = _vib_corrections_for_structure(ads_xyz_path)
        gas_vib = _vib_corrections_for_gas_species(sp_id)
        delta_e_ads = float(e_ads_on_slab - e_slab - e_gas)
        delta_zpe_ads = float(ads_vib["zpe"] - slab_vib_corr["zpe"] - gas_vib["zpe"])
        delta_g_corr_ads = float(ads_vib["g_corr"] - slab_vib_corr["g_corr"] - gas_vib["g_corr"])

        adsorption_results.append({
            "species": sp_id,
            "species_name": ctx.ads_templates[sp_id].get("name", sp_id),
            "adsorbate_xyz": os.path.abspath(ads_xyz_path),
            "temperature_K": float(temperature_K),
            "e_ads_on_slab": float(e_ads_on_slab),
            "e_slab": float(e_slab),
            "e_gas": float(e_gas),
            "adsorption_energy": delta_e_ads,
            "delta_zpe_adsorption": delta_zpe_ads,
            "delta_g_correction_adsorption": delta_g_corr_ads,
            "adsorption_energy_zpe": float(delta_e_ads + delta_zpe_ads),
            "adsorption_free_energy": float(delta_e_ads + delta_g_corr_ads),
            "adsorption_free_energy_zpe": float(delta_e_ads + delta_zpe_ads + delta_g_corr_ads),
            "ads_vib_source": ads_vib["used_prefix"],
            "slab_vib_source": slab_vib_corr["used_prefix"],
            "gas_vib_source": gas_vib["used_prefix"],
        })

    adsorption_results_path = os.path.join(out_root, ctx._tagged_stem("adsorption_energy") + ".yaml")
    with open(adsorption_results_path, "w") as f:
        yaml.safe_dump({"adsorption_energy": adsorption_results}, f, sort_keys=False, allow_unicode=True)

    print(f"Adsorption-energy evaluation finished: {len(adsorption_results)} species. Summary: {adsorption_results_path}")
    return adsorption_results


def _infer_gas_geometry(atoms):
    if len(atoms) == 1:
        return "monatomic"
    moments = np.array(atoms.get_moments_of_inertia(), dtype=float)
    if len(moments) == 0:
        return "nonlinear"
    if float(np.min(moments)) < 1e-3:
        return "linear"
    return "nonlinear"


def _infer_gas_symmetry_number(atoms):
    formula = atoms.get_chemical_formula(mode="hill")
    if formula in {"H2", "N2", "O2", "F2", "Cl2", "Br2", "I2"}:
        return 2
    if formula in {"CO2", "H2O"}:
        return 2
    return 1


def _gas_ideal_thermo_corrections(ctx, atoms, temperature_K, pressure_pa, name,
                                    indices=None, delta=0.01, nfree=2):
    if indices is None:
        indices = list(range(len(atoms)))

    name_abs = os.path.abspath(name)
    os.makedirs(os.path.dirname(name_abs), exist_ok=True)

    cache_exists = False
    if os.path.isdir(name_abs):
        cache_entries = set(os.listdir(name_abs))
        cache_exists = (
            os.path.exists(os.path.join(name_abs, "combined.json"))
            or any(entry.startswith("cache.") and entry.endswith(".json") for entry in cache_entries)
        )

    vib = Vibrations(atoms, indices=indices, name=name_abs, delta=delta, nfree=nfree)
    if cache_exists:
        try:
            vib.read()
            print(f"Vibrations cache hit for {name_abs}; loaded existing results.")
        except Exception:
            shutil.rmtree(name_abs)
            vib.run()
    else:
        vib.run()
    vib.summary()

    vib_energies_raw = np.array(vib.get_energies(), dtype=complex)
    vib_energies = []
    for e in vib_energies_raw:
        if np.iscomplexobj(e) and abs(np.imag(e)) > 1e-10:
            continue
        e_real = float(np.real(e))
        if e_real > 1e-12:
            vib_energies.append(e_real)

    if len(vib_energies) == 0:
        return 0.0, 0.0, 0

    geometry = _infer_gas_geometry(atoms)
    symmetry_number = _infer_gas_symmetry_number(atoms)
    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        potentialenergy=0.0,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetry_number,
        spin=0,
    )
    g_corr = thermo.get_gibbs_energy(temperature_K, pressure=pressure_pa)
    zpe = thermo.get_ZPE_correction()
    return float(g_corr), float(zpe), name_abs
