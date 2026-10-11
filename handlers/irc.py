"""
IRC (intrinsic reaction coordinate) stage: run IRC from accepted TS records,
select final-state endpoints, and compute thermo corrections.
"""
import hashlib
import os
import shutil

import numpy as np
import yaml
from ase.geometry import get_distances
from ase.data import covalent_radii
from ase.io import read, write
from ase.optimize import QuasiNewton
import ase

from recnet_backend.calculator import DP

from .context import WorkflowContext, MODEL


class IRCHandler:
    """Runs IRC from TS records and selects final-state endpoints."""

    def __init__(self, ctx: WorkflowContext):
        self.ctx = ctx

    def run(self):
        return get_final_state_IRC(self.ctx)


def get_final_state_IRC(ctx: WorkflowContext):
    """Run IRC from each TS record and pick the clearly dissociated endpoint."""
    try:
        from sella import IRC
    except Exception as exc:
        raise RuntimeError(f"Sella IRC is unavailable: {exc}")

    records = ctx._load_ts_records()
    if len(records) == 0:
        print("No TS records found; skip IRC final-state search.")
        return []

    out_root = os.path.join(ctx.path, "rxn", "IRC_final_states")
    os.makedirs(out_root, exist_ok=True)

    temperature_K = float(ctx.temperature)
    enable_irc_thermo = bool(ctx.enable_irc_thermo_corrections)
    vib_corr_cache = {}

    if enable_irc_thermo:
        vib_root = os.path.join(out_root, "vib_jobs")
        os.makedirs(vib_root, exist_ok=True)
        print(f"IRC thermo corrections enabled at T={temperature_K:.2f} K")
    else:
        print("IRC thermo corrections disabled; all ZPE/G correction terms are set to 0.")

    def _vib_corrections_for_atoms(atoms, cache_key, tag_prefix):
        if not enable_irc_thermo:
            return {"zpe": 0.0, "g_corr": 0.0, "used_prefix": 0, "has_vib": False}

        if cache_key in vib_corr_cache:
            return vib_corr_cache[cache_key]

        vib_atoms = atoms.copy()
        vib_atoms.calc = DP(model=MODEL)
        vib_atoms.set_constraint(ctx._build_constraints(vib_atoms, with_internal_bonds=False))

        digest = hashlib.md5(str(cache_key).encode("utf-8")).hexdigest()[:10]
        vib_name = os.path.join(vib_root, f"{tag_prefix}_{digest}")
        g_corr, zpe, used_prefix = ctx._thermo_analysis(vib_atoms, temperature_K, name=vib_name)
        out = {
            "zpe": float(zpe),
            "g_corr": float(g_corr),
            "used_prefix": used_prefix,
            "has_vib": bool(used_prefix != 0),
        }
        vib_corr_cache[cache_key] = out
        return out

    results = []
    for rec in records:
        tag = rec.get("tag", f"rxn_{rec['rxn_idx']}_site_{rec['site']}")
        case_dir = os.path.join(out_root, tag)
        os.makedirs(case_dir, exist_ok=True)

        ts_atoms = read(rec["ts_xyz"])
        init_atoms = read(rec["initial_xyz"])
        rec_i, rec_j = rec["rec_bond_global"]

        ts_atoms.calc = DP(model=MODEL)
        ts_atoms.set_constraint(ctx._build_constraints(ts_atoms, with_internal_bonds=False))

        forward_traj = os.path.join(case_dir, "irc_forward.traj")
        reverse_traj = os.path.join(case_dir, "irc_reverse.traj")

        ts_fwd = read(rec["ts_xyz"])
        ts_fwd.calc = DP(model=MODEL)
        ts_fwd.set_constraint(ctx._build_constraints(ts_fwd, with_internal_bonds=False))
        irc_forward = IRC(
            ts_fwd,
            trajectory=forward_traj,
            dx=ctx.irc_dx,
            eta=ctx.irc_eta,
            ninner_iter=ctx.irc_ninner_iter,
            logfile=os.path.join(case_dir, "irc_forward.log"),
        )
        irc_forward.run(fmax=ctx.irc_fmax, steps=ctx.irc_steps, direction="forward")

        ts_rev = read(rec["ts_xyz"])
        ts_rev.calc = DP(model=MODEL)
        ts_rev.set_constraint(ctx._build_constraints(ts_rev, with_internal_bonds=False))
        irc_reverse = IRC(
            ts_rev,
            trajectory=reverse_traj,
            dx=ctx.irc_dx,
            eta=ctx.irc_eta,
            ninner_iter=ctx.irc_ninner_iter,
            logfile=os.path.join(case_dir, "irc_reverse.log"),
        )
        irc_reverse.run(fmax=ctx.irc_fmax, steps=ctx.irc_steps, direction="reverse")

        forward_path = read(forward_traj, index=":")
        reverse_path = read(reverse_traj, index=":")
        if len(forward_path) == 0 or len(reverse_path) == 0:
            print(f"IRC path empty for {tag}, skip.")
            continue

        reverse_final = reverse_path[-1]
        forward_final = forward_path[-1]

        reverse_final.calc = DP(model=MODEL)
        forward_final.calc = DP(model=MODEL)

        reverse_final.set_constraint(ctx._build_constraints(reverse_final, with_internal_bonds=False))
        forward_final.set_constraint(ctx._build_constraints(forward_final, with_internal_bonds=False))
        init_atoms.calc = DP(model=MODEL)
        init_atoms.set_constraint(ctx._build_constraints(init_atoms, with_internal_bonds=False))

        if os.path.exists(os.path.join(case_dir, "initial_opt.xyz")):
            init_atoms = read(os.path.join(case_dir, "initial_opt.xyz"))
            init_atoms.calc = DP(model=MODEL)
            init_atoms.set_constraint(ctx._build_constraints(init_atoms, with_internal_bonds=False))
        else:
            init_opt = QuasiNewton(init_atoms)
            try:
                init_opt.run(steps=70, fmax=0.05)
            except ase.calculators.calculator.CalculationFailed:
                print(f"Initial state optimization failed for {tag}; proceed with unoptimized initial state.")
            write(os.path.join(case_dir, "initial_opt.xyz"), init_atoms)

        _, dmat_init = get_distances(init_atoms.get_positions(), None, init_atoms.get_cell(), [True, True, True])
        d_break_init = float(dmat_init[rec_i, rec_j])

        _, dmat_r = get_distances(reverse_final.get_positions(), None, reverse_final.get_cell(), [True, True, True])
        _, dmat_f = get_distances(forward_final.get_positions(), None, forward_final.get_cell(), [True, True, True])
        d_break_r = float(dmat_r[rec_i, rec_j])
        d_break_f = float(dmat_f[rec_i, rec_j])

        z_i = int(ts_atoms[rec_i].number)
        z_j = int(ts_atoms[rec_j].number)
        r_sum = float(covalent_radii[z_i] + covalent_radii[z_j])
        dissoc_threshold = max(1.35 * d_break_init, 1.25 * r_sum, d_break_init + 0.35)

        chosen_branch = "forward" if d_break_f >= d_break_r else "reverse"
        chosen_final = forward_final if chosen_branch == "forward" else reverse_final
        chosen_d_break = d_break_f if chosen_branch == "forward" else d_break_r

        used_perturbation_fallback = False
        perturbation_choice = None

        if max(d_break_f, d_break_r) < dissoc_threshold:
            ts_stem = rec.get("tag", f"rxn_{rec['rxn_idx']}_site_{rec['site']}")
            perturb_candidates = []
            perturb_specs = [
                ("imag_plus", os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_imag_plus.xyz")),
                ("imag_minus", os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_imag_minus.xyz")),
            ]

            for label, path in perturb_specs:
                if not os.path.exists(path):
                    continue
                try:
                    cand = read(path)
                    cand.calc = DP(model=MODEL)
                    cand.set_constraint(ctx._build_constraints(cand, with_internal_bonds=False))
                    _, dmat_c = get_distances(cand.get_positions(), None, cand.get_cell(), [True, True, True])
                    d_break_c = float(dmat_c[rec_i, rec_j])
                    e_c = float(cand.get_potential_energy())
                    perturb_candidates.append({"label": label, "path": path, "atoms": cand,
                                                "d_break": d_break_c, "energy": e_c})
                except Exception:
                    continue

            if perturb_candidates:
                dissoc_like = [c for c in perturb_candidates if c["d_break"] >= dissoc_threshold]
                if dissoc_like:
                    dissoc_like.sort(key=lambda x: (x["d_break"], -x["energy"]), reverse=True)
                    best_pert = dissoc_like[0]
                else:
                    perturb_candidates.sort(key=lambda x: (x["d_break"], -x["energy"]), reverse=True)
                    best_pert = perturb_candidates[0]

                if best_pert["d_break"] > chosen_d_break + 0.05:
                    chosen_final = best_pert["atoms"]
                    chosen_branch = f"perturbation_{best_pert['label']}"
                    chosen_d_break = float(best_pert["d_break"])
                    used_perturbation_fallback = True
                    perturbation_choice = {
                        "label": best_pert["label"],
                        "source_xyz": os.path.abspath(best_pert["path"]),
                        "d_break": float(best_pert["d_break"]),
                        "energy": float(best_pert["energy"]),
                    }
                    write(os.path.join(case_dir, "final_perturbation.xyz"), chosen_final)
                    print(f"{tag}: IRC final states are not clearly dissociated; fallback to {best_pert['label']} perturbation endpoint.")

        write(os.path.join(case_dir, "initial.xyz"), init_atoms)
        write(os.path.join(case_dir, "ts.xyz"), ts_atoms)
        write(os.path.join(case_dir, "final_forward.xyz"), forward_final)
        write(os.path.join(case_dir, "final_reverse.xyz"), reverse_final)
        write(os.path.join(case_dir, "final_selected.xyz"), chosen_final)

        reverse_rev = list(reverse_path)
        reverse_rev.reverse()
        full_path = reverse_rev + forward_path[1:]
        write(os.path.join(case_dir, "irc_full_path.xyz"), full_path)

        result = {
            "tag": tag,
            "temperature_K": float(temperature_K),
            "irc_thermo_corrections_enabled": bool(enable_irc_thermo),
            "initial_xyz": os.path.abspath(os.path.join(case_dir, "initial.xyz")),
            "ts_xyz": os.path.abspath(os.path.join(case_dir, "ts.xyz")),
            "final_forward_xyz": os.path.abspath(os.path.join(case_dir, "final_forward.xyz")),
            "final_reverse_xyz": os.path.abspath(os.path.join(case_dir, "final_reverse.xyz")),
            "final_selected_xyz": os.path.abspath(os.path.join(case_dir, "final_selected.xyz")),
            "chosen_branch": chosen_branch,
            "broken_bond_global": [int(rec_i), int(rec_j)],
            "d_break_initial": d_break_init,
            "dissociation_threshold": dissoc_threshold,
            "d_break_forward": d_break_f,
            "d_break_reverse": d_break_r,
            "d_break_selected": chosen_d_break,
            "e_forward": float(forward_final.get_potential_energy()),
            "e_reverse": float(reverse_final.get_potential_energy()),
            "e_selected": float(chosen_final.get_potential_energy()),
            "e_initial": float(init_atoms.get_potential_energy()),
            "e_ts": float(ts_atoms.get_potential_energy()),
            "used_perturbation_fallback": bool(used_perturbation_fallback),
            "perturbation_choice": perturbation_choice,
        }

        init_vib = _vib_corrections_for_atoms(init_atoms, result["initial_xyz"], f"{tag}_is")
        ts_vib = _vib_corrections_for_atoms(ts_atoms, result["ts_xyz"], f"{tag}_ts")
        selected_vib = _vib_corrections_for_atoms(chosen_final, f"{tag}::selected::{chosen_branch}", f"{tag}_selected")

        result.update({
            "zpe_correction_initial": float(init_vib["zpe"]),
            "zpe_correction_ts": float(ts_vib["zpe"]),
            "zpe_correction_selected": float(selected_vib["zpe"]),
            "g_correction_initial": float(init_vib["g_corr"]),
            "g_correction_ts": float(ts_vib["g_corr"]),
            "g_correction_selected": float(selected_vib["g_corr"]),
            "e_initial_zpe": float(result["e_initial"] + init_vib["zpe"]),
            "e_ts_zpe": float(result["e_ts"] + ts_vib["zpe"]),
            "e_selected_zpe": float(result["e_selected"] + selected_vib["zpe"]),
            "g_initial": float(result["e_initial"] + init_vib["g_corr"]),
            "g_ts": float(result["e_ts"] + ts_vib["g_corr"]),
            "g_selected": float(result["e_selected"] + selected_vib["g_corr"]),
            "barrier_e_ts_minus_initial_zpe": float((result["e_ts"] + ts_vib["zpe"]) - (result["e_initial"] + init_vib["zpe"])),
            "delta_e_selected_minus_initial_zpe": float((result["e_selected"] + selected_vib["zpe"]) - (result["e_initial"] + init_vib["zpe"])),
            "barrier_g_ts_minus_initial": float((result["e_ts"] + ts_vib["g_corr"]) - (result["e_initial"] + init_vib["g_corr"])),
            "delta_g_selected_minus_initial": float((result["e_selected"] + selected_vib["g_corr"]) - (result["e_initial"] + init_vib["g_corr"])),
            "g_zpe_is": float(result["e_initial"] + init_vib["g_corr"] + init_vib["zpe"]),
            "g_zpe_ts": float(result["e_ts"] + ts_vib["g_corr"] + ts_vib["zpe"]),
            "g_zpe_fs": float(result["e_selected"] + selected_vib["g_corr"] + selected_vib["zpe"]),
            "barrier_g_zpe_ts_minus_is": float((result["e_ts"] + ts_vib["g_corr"] + ts_vib["zpe"]) - (result["e_initial"] + init_vib["g_corr"] + init_vib["zpe"])),
            "delta_g_zpe_selected_minus_is": float((result["e_selected"] + selected_vib["g_corr"] + selected_vib["zpe"]) - (result["e_initial"] + init_vib["g_corr"] + init_vib["zpe"])),
            "barrier_g_zpe_ts_minus_fs": float((result["e_ts"] + ts_vib["g_corr"] + ts_vib["zpe"]) - (result["e_selected"] + selected_vib["g_corr"] + selected_vib["zpe"])),
            "vib_source_initial": init_vib["used_prefix"],
            "vib_source_ts": ts_vib["used_prefix"],
            "vib_source_selected": selected_vib["used_prefix"],
        })

        result["e_ts_zpe"] = float(result["e_ts"] + ts_vib["zpe"])
        result["g_ts"] = float(result["e_ts"] + ts_vib["g_corr"])

        with open(os.path.join(case_dir, "irc_summary.yaml"), "w") as f:
            yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
        results.append(result)

    result_path = os.path.join(out_root, "irc_results.yaml")
    with open(result_path, "w") as f:
        yaml.safe_dump({"irc_results": results}, f, sort_keys=False, allow_unicode=True)

    print(f"IRC final-state search finished: {len(results)} cases. Summary: {result_path}")
    return results
