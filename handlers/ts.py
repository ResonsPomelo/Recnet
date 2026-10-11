"""
TS (transition state) stage: build TS guesses, run CCQN saddle optimization,
validate imaginary frequencies, and archive accepted TS records.
"""
import os
import traceback

import numpy as np
import yaml
from ase.geometry import get_distances
from ase.data import covalent_radii
from ase.io import read, write
from ase.optimize import MDMin, QuasiNewton
from ase.vibrations import Vibrations
from sella import Sella
from deepmd.calculator import DP
from ccqn import CCQN

from .context import WorkflowContext, MODEL
from utils import config as cfgmod
from utils import geometry as geom
from utils import constraints as constraint_utils


class TSHandler:
    """Handles transition-state search — build guesses, run CCQN, validate."""

    def __init__(self, ctx: WorkflowContext):
        self.ctx = ctx

    def run(self):
        generate_rxn_ts_guesses_ccqn(self.ctx)


def generate_rxn_ts_guesses_ccqn(ctx: WorkflowContext):
    """For each reaction, build TS guesses per site, run CCQN, and validate."""
    slab = ctx.slab
    os.makedirs(os.path.join(ctx.path, "rxn", "TS_guesses"), exist_ok=True)

    old_records = ctx._load_ts_records()
    records_by_tag = {rec.get("tag"): rec for rec in old_records if isinstance(rec, dict) and rec.get("tag")}
    ctx.ts_records = list(records_by_tag.values())

    def _drop_record_by_tag(tag):
        ctx.ts_records = [rec for rec in ctx.ts_records if rec.get("tag") != tag]

    def _bond_key(i, j):
        a, b = int(i), int(j)
        return (a, b) if a <= b else (b, a)

    for idx, rxn in enumerate(ctx.rxns_dict):
        preferred_sites = list(rxn.get("valid_reactant_sites", []))
        all_sites = list(rxn.get("valid_reactant_sites_all", preferred_sites))
        sites = cfgmod._dedupe_keep_order(preferred_sites + [s for s in all_sites if s not in preferred_sites])
        backup_sites = set(s for s in sites if s not in preferred_sites)
        accepted_any_site = False
        rec_bond = rxn["broken_bond"]
        sp_id = rxn["reactant_species"][0]
        rxn_key = ctx.rxn_key_by_index[idx]
        myslab = slab.stru.copy()
        template = ctx.ads_templates[sp_id]

        for site in sites:
            if site in backup_sites and accepted_any_site:
                continue
            try:
                ts_stem = ctx._tagged_stem(f"{rxn_key}_site_{site}")
                legacy_ts_stem = ctx._tagged_stem(f"rxn_{idx}_site_{site}")
                ts_tag = ts_stem
                ts_guess_xyz = os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}.xyz")
                legacy_ts_guess_xyz = os.path.join(ctx.path, "rxn", "TS_guesses", f"{legacy_ts_stem}.xyz")
                ts_opt_xyz = os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_opt.xyz")
                legacy_ts_opt_xyz = os.path.join(ctx.path, "rxn", "TS_guesses", f"{legacy_ts_stem}_opt.xyz")
                out_xyz = os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_opt_ccqn.xyz")
                legacy_out_xyz = os.path.join(ctx.path, "rxn", "TS_guesses", f"{legacy_ts_stem}_opt_ccqn.xyz")
                summary_log = ctx._tagged_stem(f"optimization_summary_{rxn_key}_site_{site}") + ".log"
                legacy_summary_log = ctx._tagged_stem(f"optimization_summary_{idx}_site_{site}") + ".log"

                reuse_out_xyz = out_xyz
                if (not os.path.exists(reuse_out_xyz)) and os.path.exists(legacy_out_xyz):
                    reuse_out_xyz = legacy_out_xyz

                reuse_summary_log = summary_log if os.path.exists(summary_log) else legacy_summary_log
                if not os.path.exists(reuse_summary_log):
                    reuse_summary_log = summary_log

                reuse_initial_xyz = ts_opt_xyz
                if not os.path.exists(reuse_initial_xyz) and os.path.exists(legacy_ts_opt_xyz):
                    reuse_initial_xyz = legacy_ts_opt_xyz
                if not os.path.exists(reuse_initial_xyz) and os.path.exists(ts_guess_xyz):
                    reuse_initial_xyz = ts_guess_xyz
                if not os.path.exists(reuse_initial_xyz) and os.path.exists(legacy_ts_guess_xyz):
                    reuse_initial_xyz = legacy_ts_guess_xyz
                if not os.path.exists(reuse_initial_xyz):
                    reuse_initial_xyz = reuse_out_xyz

                if os.path.exists(reuse_out_xyz):
                    print(f"Reuse existing TS result for {ts_stem}: {reuse_out_xyz}")
                    if not os.path.exists(reuse_summary_log):
                        with open(reuse_summary_log, "w") as f:
                            f.write("Skipped generate_rxn_ts_guesses_ccqn recomputation: existing _opt_ccqn structure was reused.\n")
                    if ts_tag not in records_by_tag:
                        ctx._save_ts_record(
                            rxn_idx=idx, site=site, sp_id=sp_id, rxn_key=rxn_key,
                            rec_bond=rec_bond, slab_atom_count=len(myslab),
                            initial_xyz=reuse_initial_xyz, ts_xyz=reuse_out_xyz,
                            summary_log=reuse_summary_log, tag=ts_tag,
                        )
                        records_by_tag[ts_tag] = ctx.ts_records[-1]
                    continue

                pos = slab.get_site(site)
                print(f"Processing rxn_{idx} at site {site} with rec_bond {rec_bond} and sp_id {sp_id}")

                stru_seed = template["atoms"].copy()
                template_bonds = {
                    _bond_key(i, j)
                    for i, j in geom.get_bond_connections(stru_seed, shift=0, cutoff=1.2, bond_type="nosurf")
                }
                expected_broken_bond = _bond_key(rec_bond[0], rec_bond[1])

                _, distances = get_distances(
                    stru_seed.get_positions(), None, stru_seed.get_cell(), [True, True, True]
                )
                d = distances[rec_bond[0], rec_bond[1]]
                msg_header = f"distance between broken bond atoms before adjustment: {d:.2f} Å\n"

                atom_bond_params_list = [{
                    "ind1": rec_bond[0] + len(myslab),
                    "ind2": rec_bond[1] + len(myslab),
                    "k": 2.0,
                    "deq": 1.1 * d,
                }]

                # Pre-orient broken-bond torsion
                torsion_info = _orient_broken_bond_torsion(
                    ctx, stru_seed, rec_bond=rec_bond,
                    ads_indices=template["ad_idx"],
                )
                msg_seed = msg_header
                if torsion_info:
                    torsion_ok, torsion_atom_idx, torsion_angle = torsion_info
                    if torsion_ok:
                        msg_seed += f"pre-rotated broken-bond torsion: atom {torsion_atom_idx} by {torsion_angle:.1f} deg\n"

                # Inner-angle adjustment
                angle_before, angle_rot = geom.rotate_about_ads_inner_angle(
                    stru_seed, rec_bond=rec_bond,
                    ads_indices=template["ad_idx"],
                    target_min_angle_deg=50.0,
                    surface_normal=ctx.surface_normal,
                )
                if angle_rot > 0.0:
                    msg_seed += (
                        f"rotated around ads atom by inner-angle rule: "
                        f"{angle_before:.2f}° -> >= 50.00°, applied {angle_rot:.2f}°\n"
                    )

                # Azimuth enumeration
                azimuth_candidates = [0.0]
                if ctx.enable_rotation_enum_ts:
                    site_bond_params_tmp, center_bond_params_tmp = ctx._build_site_and_center_anchors(
                        template, pos, len(myslab)
                    )
                    _, best_angle, best_energy = _enumerate_azimuth_orientation(
                        ctx, base_stru=stru_seed.copy(),
                        ads_idx=template["ad_idx"], myslab=myslab, pos=pos,
                        site_bond_params_list=site_bond_params_tmp,
                        center_bond_params_list=center_bond_params_tmp,
                        atom_bond_params_list=atom_bond_params_list,
                    )
                    azimuth_candidates = [float(best_angle)] + [
                        float(a) for a in ctx.rotation_angle_candidates if float(a) != float(best_angle)
                    ]
                    if best_energy is not None:
                        msg_seed += f"azimuth enumeration selected initial angle {best_angle:.1f}° with E={best_energy:.6f} eV\n"
                    else:
                        msg_seed += "azimuth enumeration failed; start from default angle 0°\n"

                accepted = False
                attempt_messages = []

                for attempt_idx, az_angle in enumerate(azimuth_candidates, start=1):
                    attempt_msg = msg_seed
                    attempt_msg += f"Attempt {attempt_idx}/{len(azimuth_candidates)}: azimuth={az_angle:.1f}°\n"

                    site_bond_params_list, center_bond_params_list = ctx._build_site_and_center_anchors(
                        template, pos, len(myslab)
                    )

                    stru = stru_seed.copy()
                    if abs(float(az_angle)) > 1e-8:
                        geom.rotate_about_ads_vertical(
                            stru=stru, ads_idx=template["ad_idx"],
                            angle_deg=float(az_angle), surface_normal=ctx.surface_normal,
                        )

                    stru.translate(np.array(pos, dtype=float) + ctx._site_lift_vector())
                    ads = myslab + stru
                    write(ts_guess_xyz, ads)

                    endpoint_anchor_added = _add_reactive_endpoint_site_anchor(
                        ctx, stru=stru, rec_bond=rec_bond,
                        ads_indices=template["ad_idx"], site_pos=pos,
                        site_bond_params_list=site_bond_params_list,
                        slab_atom_count=len(myslab),
                    )
                    if endpoint_anchor_added:
                        attempt_msg += "added weak endpoint-site anchor during pre-TS optimization\n"

                    calc = constraint_utils.HarmonicallyForcedDP(
                        model=MODEL,
                        atom_bond_potentials=atom_bond_params_list,
                        site_bond_potentials=site_bond_params_list,
                        center_bond_potentials=center_bond_params_list,
                    )
                    ads.calc = calc
                    ads.set_constraint(ctx._build_constraints(ads, with_internal_bonds=False))

                    try:
                        opt_md = MDMin(ads, dt=0.05)
                        opt_md.run(fmax=0.5, steps=70)
                        opt_qn = QuasiNewton(ads)
                        opt_qn.run(fmax=0.05, steps=70)
                    except Exception as exc:
                        attempt_msg += f"Optimization failed: {exc}\n"
                        attempt_messages.append(attempt_msg)
                        continue

                    ads.calc = DP(model=MODEL)
                    _, distances = get_distances(
                        ads.get_positions(), None, stru.get_cell(), [True, True, True]
                    )
                    rec_bond_global = (rec_bond[0] + len(myslab), rec_bond[1] + len(myslab))
                    d_opt = distances[rec_bond_global[0], rec_bond_global[1]]
                    attempt_msg += f"distance between broken bond atoms after opt: {d_opt:.2f} Å\n"

                    z_i = int(ads[rec_bond_global[0]].number)
                    z_j = int(ads[rec_bond_global[1]].number)
                    d_ccqn_max = _ts_bond_upper_bound(ctx, d_ref=d, atom_i_number=z_i, atom_j_number=z_j)
                    attempt_msg += f"TS reactive-bond max length threshold: {d_ccqn_max:.2f} Å\n"

                    write(os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_opt.xyz"), ads)
                    ads_pre_ccqn = ads.copy()

                    ads.calc = DP(model=MODEL)
                    opt_ccqn = CCQN(
                        ads,
                        logfile=os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_ccqn_try{attempt_idx}.log"),
                        e_vector_method="ic",
                        reactive_bonds=[(rec_bond[0] + len(myslab), rec_bond[1] + len(myslab))],
                        saddle_optimizer="sella",
                        sella_kwargs={"delta0": 0.05},
                    )
                    opt_ccqn.run(fmax=0.05, steps=100)
                    write(out_xyz, ads)

                    _, distances = get_distances(
                        ads.get_positions(), None, stru.get_cell(), [True, True, True]
                    )
                    d_ccqn = distances[rec_bond_global[0], rec_bond_global[1]]
                    attempt_msg += f"distance between broken bond atoms after ccqn: {d_ccqn:.2f} Å\n"

                    overstretched, _ = _is_ts_bond_overstretched(
                        ctx, d_now=d_ccqn, d_ref=d, atom_i_number=z_i, atom_j_number=z_j,
                    )
                    if overstretched:
                        attempt_msg += "Reactive bond is over-elongated after CCQN; rollback to pre-CCQN structure and rerun conservative CCQN.\n"
                        ads = ads_pre_ccqn.copy()
                        ads.calc = DP(model=MODEL)
                        ads.set_constraint(ctx._build_constraints(ads, with_internal_bonds=False))
                        opt_ccqn_retry = CCQN(
                            ads,
                            logfile=os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_ccqn_retry_try{attempt_idx}.log"),
                            e_vector_method="ic",
                            reactive_bonds=[rec_bond_global],
                            saddle_optimizer="sella",
                            sella_kwargs={"delta0": 0.02},
                        )
                        opt_ccqn_retry.run(fmax=0.04, steps=70)
                        write(out_xyz, ads)

                        _, distances = get_distances(
                            ads.get_positions(), None, stru.get_cell(), [True, True, True]
                        )
                        d_ccqn_retry = distances[rec_bond_global[0], rec_bond_global[1]]
                        attempt_msg += f"distance between broken bond atoms after rollback ccqn: {d_ccqn_retry:.2f} Å\n"

                        overstretched_retry, _ = _is_ts_bond_overstretched(
                            ctx, d_now=d_ccqn_retry, d_ref=d, atom_i_number=z_i, atom_j_number=z_j,
                        )
                        if overstretched_retry:
                            attempt_msg += "Rejected this orientation: reactive bond is still over-elongated after rollback retry.\n"
                            attempt_messages.append(attempt_msg)
                            continue
                        d_ccqn = d_ccqn_retry
                        attempt_msg += "Rollback retry recovered acceptable TS bond length.\n"

                    d_ccqn_min = _ts_bond_dissociation_threshold(
                        ctx, d_ref=d, atom_i_number=z_i, atom_j_number=z_j,
                    )
                    attempt_msg += f"TS reactive-bond dissociation min threshold: {d_ccqn_min:.2f} Å\n"
                    if d_ccqn < d_ccqn_min:
                        attempt_msg += "Rejected this orientation: reactive bond is not sufficiently dissociated after CCQN; skip vibration checks.\n"
                        attempt_messages.append(attempt_msg)
                        continue

                    ads.calc = DP(model=MODEL)
                    vib_prefix = os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_vib_try{attempt_idx}")

                    def _run_vibration_with_cleanup(prefix, indices=None):
                        if os.path.exists(prefix):
                            import shutil
                            shutil.rmtree(prefix)
                        if indices is None:
                            indices = ctx._get_vib_indices(ads)
                        vib_local = Vibrations(ads, name=prefix, indices=indices)
                        vib_local.run()
                        vib_local.summary()
                        freqs_local = vib_local.get_frequencies()
                        imag_local = [
                            f for f in freqs_local
                            if (np.iscomplexobj(f) and abs(np.imag(f)) > 1e-12)
                            or (np.isrealobj(f) and np.real(f) < 0.0)
                        ]
                        return vib_local, freqs_local, imag_local

                    vib, freqs, imag_freqs = _run_vibration_with_cleanup(vib_prefix)
                    attempt_msg += f"Initial imaginary frequency count: {len(imag_freqs)}\n"

                    if len(imag_freqs) != 1:
                        attempt_msg += "Imaginary frequency count is not 1; run an extra Sella saddle optimization and recompute vibrations.\n"
                        try:
                            ads.calc = DP(model=MODEL)
                            ads.set_constraint(ctx._build_constraints(ads, with_internal_bonds=False))
                            retry_traj = os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_sella_retry_try{attempt_idx}.traj")
                            retry_log = os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_sella_retry_try{attempt_idx}.log")
                            opt_retry = Sella(ads, trajectory=retry_traj, logfile=retry_log)
                            opt_retry.run(fmax=0.05, steps=80)
                            retry_xyz = os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_opt_sella_retry_try{attempt_idx}.xyz")
                            write(retry_xyz, ads)
                            _, distances = get_distances(
                                ads.get_positions(), None, stru.get_cell(), [True, True, True]
                            )
                            d_retry = distances[rec_bond[0] + len(myslab), rec_bond[1] + len(myslab)]
                            attempt_msg += f"distance between broken bond atoms after extra sella: {d_retry:.2f} Å\n"
                            vib, freqs, imag_freqs = _run_vibration_with_cleanup(vib_prefix)
                            attempt_msg += f"Imaginary frequency count after extra sella: {len(imag_freqs)}\n"
                        except Exception as exc:
                            attempt_msg += f"Extra sella saddle optimization failed: {exc}\n"

                    significant_imag_freqs = _significant_imag_frequencies(ctx, freqs)
                    attempt_msg += (
                        f"Significant imaginary frequency count "
                        f"(|nu| >= {ctx.imag_freq_significant_cutoff:.1f} cm^-1): "
                        f"{len(significant_imag_freqs)}\n"
                    )
                    if len(significant_imag_freqs) != 1:
                        attempt_msg += "Rejected this orientation: significant imaginary frequency count is not 1 after TS refinement.\n"
                        attempt_messages.append(attempt_msg)
                        continue

                    ads_only = ads[len(myslab):]
                    final_bonds = {
                        _bond_key(i, j)
                        for i, j in geom.get_bond_connections(ads_only, shift=0, cutoff=1.2, bond_type="nosurf")
                    }
                    broken_bonds = sorted(template_bonds - final_bonds)
                    unexpected_broken_bonds = [b for b in broken_bonds if b != expected_broken_bond]
                    has_unexpected_break = len(unexpected_broken_bonds) > 0

                    if has_unexpected_break:
                        attempt_msg += f"Unexpected broken bonds detected after CCQN: {unexpected_broken_bonds}\n"

                    if len(imag_freqs) != 1 and has_unexpected_break:
                        attempt_msg += "Rejected this orientation: non-single imaginary frequency and unexpected bond breaking were both observed.\n"
                        attempt_messages.append(attempt_msg)
                        continue

                    if imag_freqs:
                        attempt_msg += f"Imaginary frequencies found: {imag_freqs}\n"
                    else:
                        attempt_msg += "No imaginary frequencies found (not a TS candidate).\n"

                    imag_report = _get_imag_mode_report(
                        ctx, ads=ads, vib=vib,
                        rec_bond_global=rec_bond_global,
                        slab_atom_count=len(myslab),
                    )
                    if imag_report is not None:
                        for m in imag_report["all"]:
                            attempt_msg += (
                                f"Imag mode {m['mode_idx']}: f={m['freq']}, "
                                f"bond_proj={m['bond_proj']:.3f}, "
                                f"ads_drift_proj={m['drift_proj']:.3f}, "
                                f"max_atom={m['max_atom_symbol']}{m['max_atom']}\n"
                            )
                        primary = imag_report["primary"]
                        if primary["bond_proj"] >= primary["drift_proj"]:
                            attempt_msg += "Primary imaginary mode is bond-breaking dominated.\n"
                        else:
                            attempt_msg += "Warning: primary imaginary mode is adsorbate-drift dominated.\n"

                        drift_dominated = (
                            primary["bond_proj"] + ctx.ts_primary_mode_bond_margin < primary["drift_proj"]
                        )
                        weak_bond_character = primary["bond_proj"] < ctx.ts_primary_mode_min_bond_proj
                        if drift_dominated or weak_bond_character:
                            attempt_msg += "Rejected this orientation: primary imaginary mode does not show sufficiently strong bond-breaking character.\n"
                            attempt_messages.append(attempt_msg)
                            continue

                        if ctx.enable_imag_mode_check:
                            plus_xyz = os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_imag_plus_try{attempt_idx}.xyz")
                            minus_xyz = os.path.join(ctx.path, "rxn", "TS_guesses", f"{ts_stem}_imag_minus_try{attempt_idx}.xyz")
                            plus_info = _displace_and_relax_imag_mode(
                                ctx, atoms=ads, mode=primary["mode"], sign=+1.0,
                                out_xyz=plus_xyz, rec_bond_global=rec_bond_global, site_pos=pos,
                            )
                            minus_info = _displace_and_relax_imag_mode(
                                ctx, atoms=ads, mode=primary["mode"], sign=-1.0,
                                out_xyz=minus_xyz, rec_bond_global=rec_bond_global, site_pos=pos,
                            )
                            if plus_info is not None and minus_info is not None:
                                attempt_msg += (
                                    f"Imag+ endpoint: E={plus_info['energy']:.6f} eV, "
                                    f"d_break={plus_info['broken_bond']:.3f} A, "
                                    f"h_ads={plus_info['ads_height']:.3f} A\n"
                                )
                                attempt_msg += (
                                    f"Imag- endpoint: E={minus_info['energy']:.6f} eV, "
                                    f"d_break={minus_info['broken_bond']:.3f} A, "
                                    f"h_ads={minus_info['ads_height']:.3f} A\n"
                                )
                    else:
                        attempt_msg += "Imaginary mode vectors unavailable for projection analysis.\n"

                    attempt_messages.append(attempt_msg)
                    accepted = True
                    break

                msg = "\n".join(attempt_messages) + "\n"
                with open(summary_log, "w") as f:
                    f.write(msg)

                if not accepted:
                    print(f"All orientations rejected for rxn_{idx}_site_{site}; try next candidate site.")
                    continue

                accepted_any_site = True

                if ts_tag in records_by_tag:
                    _drop_record_by_tag(ts_tag)
                    records_by_tag.pop(ts_tag, None)

                ctx._save_ts_record(
                    rxn_idx=idx, site=site, sp_id=sp_id, rxn_key=rxn_key,
                    rec_bond=rec_bond, slab_atom_count=len(myslab),
                    initial_xyz=ts_opt_xyz if os.path.exists(ts_opt_xyz) else ts_guess_xyz,
                    ts_xyz=out_xyz, summary_log=summary_log, tag=ts_tag,
                )
                records_by_tag[ts_tag] = ctx.ts_records[-1]
            except Exception as exc:
                print(f"Error for rxn_{idx}_site_{site}, skip this structure: {exc}")
                traceback.print_exc()
                continue

        if (not accepted_any_site) and len(backup_sites) > 0:
            print(f"No TS accepted on preferred top-x sites for rxn_{idx}; backup sites were also attempted but all failed.")

    ctx._flush_ts_records()


# ---------------------------------------------------------------------- #
#  TS-specific helper functions                                          #
# ---------------------------------------------------------------------- #

def _enumerate_azimuth_orientation(
    ctx, base_stru, ads_idx, myslab, pos,
    site_bond_params_list, center_bond_params_list,
    atom_bond_params_list=None,
):
    best_stru = base_stru
    best_angle = 0
    best_energy = None

    for az in ctx.rotation_angle_candidates:
        cand = base_stru.copy()
        if az != 0:
            geom.rotate_about_ads_vertical(cand, ads_idx, az, surface_normal=ctx.surface_normal)
        cand_shifted = cand.copy()
        cand_shifted.translate(np.array(pos, dtype=float) + ctx._site_lift_vector())
        cand_ads = myslab + cand_shifted
        cand_ads.set_constraint(ctx._build_constraints(cand_ads, surf_atom_num=len(myslab), with_internal_bonds=False))
        cand_ads.calc = constraint_utils.HarmonicallyForcedDP(
            model=MODEL,
            atom_bond_potentials=atom_bond_params_list,
            site_bond_potentials=site_bond_params_list,
            center_bond_potentials=center_bond_params_list,
        )
        try:
            e_cand = float(cand_ads.get_potential_energy())
        except Exception:
            continue
        if best_energy is None or e_cand < best_energy:
            best_energy = e_cand
            best_angle = az
            best_stru = cand
    return best_stru, best_angle, best_energy


def _find_heavy_neighbor(stru, atom_idx, exclude_idx=None, cutoff_scale=1.25):
    symbols = stru.get_chemical_symbols()
    pos = stru.get_positions()
    target_num = stru[atom_idx].number

    best_idx = None
    best_ratio = None
    for j in range(len(stru)):
        if j == atom_idx or j == exclude_idx:
            continue
        if symbols[j] == "H":
            continue
        d = np.linalg.norm(pos[j] - pos[atom_idx])
        r_sum = covalent_radii[target_num] + covalent_radii[stru[j].number]
        if r_sum <= 1e-12:
            continue
        ratio = d / r_sum
        if ratio <= cutoff_scale and (best_ratio is None or ratio < best_ratio):
            best_idx = j
            best_ratio = ratio
    return best_idx


def _get_rotating_group_indices(stru, moving_idx, pivot_idx):
    n_atoms = len(stru)
    if n_atoms == 0:
        return []
    adjacency = [[] for _ in range(n_atoms)]
    for a, b in geom.get_bond_connections(stru, shift=0, cutoff=1.2, bond_type="nosurf"):
        adjacency[a].append(b)
        adjacency[b].append(a)
    visited = {pivot_idx}
    stack = [moving_idx]
    group = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        group.append(node)
        for nb in adjacency[node]:
            if nb not in visited:
                stack.append(nb)
    return group


def _orient_broken_bond_torsion(ctx, stru, rec_bond, ads_indices=None):
    if len(rec_bond) != 2:
        return False, None, 0.0

    i, j = rec_bond
    pos = stru.get_positions()
    surface_normal = ctx.surface_normal

    normal = np.array(surface_normal, dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-8:
        return False, None, 0.0
    normal_unit = normal / normal_norm

    ads_set = set(ads_indices or [])
    bonded_pairs = ctx._adsorbate_bond_set(stru)

    def _passes_contact_constraints(cand_pos, moving_group):
        n_atoms = len(stru)
        for a in moving_group:
            for b in range(n_atoms):
                if a == b:
                    continue
                pair = (a, b) if a <= b else (b, a)
                if pair in bonded_pairs:
                    continue
                d_ab = float(np.linalg.norm(cand_pos[a] - cand_pos[b]))
                r_sum = float(covalent_radii[int(stru[a].number)] + covalent_radii[int(stru[b].number)])
                d_min = max(0.75, 0.60 * r_sum)
                if d_ab < d_min:
                    return False
        return True

    ref_ids = [k for k in range(len(stru)) if k not in rec_bond]
    if not ref_ids:
        ref_ids = list(range(len(stru)))
    ref_center = np.mean(pos[ref_ids], axis=0)

    def _score(moved_point, pivot_idx):
        bond_vec = moved_point - pos[pivot_idx]
        bond_norm = np.linalg.norm(bond_vec)
        if bond_norm < 1e-12:
            return -1e9
        bond_unit = bond_vec / bond_norm
        parallel_score = 1.0 - abs(np.dot(bond_unit, normal_unit))
        rc_vec = moved_point - ref_center
        rc_norm = np.linalg.norm(rc_vec)
        if rc_norm < 1e-12:
            lateral_score = 0.0
        else:
            rc_plane = rc_vec - np.dot(rc_vec, normal_unit) * normal_unit
            lateral_score = np.linalg.norm(rc_plane) / rc_norm
        keep_ads_toward_surface_score = 0.0
        if ads_set:
            ad_center = np.mean(pos[list(ads_set)], axis=0)
            outward_delta = np.dot(moved_point - ad_center, normal_unit)
            keep_ads_toward_surface_score = np.tanh(outward_delta)
        return 0.65 * parallel_score + 0.20 * lateral_score + 0.15 * keep_ads_toward_surface_score

    candidates = []
    for moving_idx, pivot_idx in [(i, j), (j, i)]:
        if moving_idx in ads_set:
            continue
        rotating_group = _get_rotating_group_indices(stru, moving_idx, pivot_idx)
        if not rotating_group:
            continue
        if any(idx in ads_set for idx in rotating_group):
            continue
        axis_end = _find_heavy_neighbor(stru, pivot_idx, exclude_idx=moving_idx)
        if axis_end is None:
            continue
        axis_vec = pos[axis_end] - pos[pivot_idx]
        axis_norm = np.linalg.norm(axis_vec)
        if axis_norm < 1e-8:
            continue
        axis_unit = axis_vec / axis_norm

        base_point = pos[moving_idx].copy()
        for ang in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            cand_point = geom.rotate_atom_around_axis(base_point, pos[pivot_idx], axis_unit, ang)
            cand_pos = pos.copy()
            for r_idx in rotating_group:
                cand_pos[r_idx] = geom.rotate_atom_around_axis(pos[r_idx].copy(), pos[pivot_idx], axis_unit, ang)
            if not _passes_contact_constraints(cand_pos, rotating_group):
                continue
            score = _score(cand_point, pivot_idx)
            candidates.append((score, moving_idx, pivot_idx, axis_unit, ang, rotating_group))

    if not candidates:
        return False, None, 0.0
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_moving_idx, best_pivot_idx, best_axis_unit, best_ang, best_group = candidates[0]
    if abs(best_ang) > 1e-8:
        for r_idx in best_group:
            pos[r_idx] = geom.rotate_atom_around_axis(pos[r_idx].copy(), pos[best_pivot_idx], best_axis_unit, best_ang)
        stru.set_positions(pos)
    return True, int(best_moving_idx), float(best_ang)


def _add_reactive_endpoint_site_anchor(ctx, stru, rec_bond, ads_indices, site_pos,
                                        site_bond_params_list, slab_atom_count,
                                        k=0.35, deq=0.6):
    if len(rec_bond) != 2:
        return False
    ads_set = set(ads_indices or [])
    candidates = [idx for idx in rec_bond if 0 <= idx < len(stru)]
    if not candidates:
        return False
    normal = np.array(ctx.surface_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    pos = stru.get_positions()
    ad_center = np.mean(pos[list(ads_set)], axis=0) if ads_set else np.zeros(3)

    def _score(idx):
        penalty_ads = -100.0 if idx in ads_set else 0.0
        outward = float(np.dot(pos[idx] - ad_center, normal))
        return penalty_ads + outward

    anchor_local_idx = max(candidates, key=_score)
    anchor_global_idx = anchor_local_idx + slab_atom_count
    if any(p.get("ind") == anchor_global_idx for p in site_bond_params_list):
        return False
    site_bond_params_list.append({"site_pos": site_pos, "ind": anchor_global_idx, "k": k, "deq": deq})
    return True


def _get_imag_mode_report(ctx, ads, vib, rec_bond_global, slab_atom_count):
    freqs = np.array(vib.get_frequencies(), dtype=complex)
    imag_indices = [
        i for i, f in enumerate(freqs)
        if (np.iscomplexobj(f) and abs(np.imag(f)) > 1e-12) or (np.isrealobj(f) and np.real(f) < 0.0)
    ]
    if len(imag_indices) == 0:
        return None

    normal = np.array(ctx.surface_normal, dtype=float)
    normal /= max(np.linalg.norm(normal), 1e-12)
    n_atoms = len(ads)
    ads_indices = list(range(slab_atom_count, n_atoms))

    i, j = rec_bond_global
    pos = ads.get_positions()
    bond_vec = pos[j] - pos[i]
    bond_norm = np.linalg.norm(bond_vec)
    bond_unit = bond_vec / bond_norm if bond_norm > 1e-12 else np.array([1.0, 0.0, 0.0])

    reaction_vec = np.zeros((n_atoms, 3), dtype=float)
    reaction_vec[i] = -bond_unit
    reaction_vec[j] = bond_unit
    reaction_norm = np.linalg.norm(reaction_vec)

    ads_trans_vec = np.zeros((n_atoms, 3), dtype=float)
    for idx in ads_indices:
        ads_trans_vec[idx] = normal
    ads_trans_norm = np.linalg.norm(ads_trans_vec)

    mode_reports = []
    for mode_idx in imag_indices:
        mode = np.array(vib.get_mode(mode_idx), dtype=float)
        mode_norm = np.linalg.norm(mode)
        if mode_norm < 1e-14:
            continue
        atom_amp = np.linalg.norm(mode, axis=1)
        max_atom = int(np.argmax(atom_amp))
        bond_proj = abs(float(np.sum(mode * reaction_vec))) / (mode_norm * reaction_norm) if reaction_norm > 1e-14 else 0.0
        drift_proj = abs(float(np.sum(mode * ads_trans_vec))) / (mode_norm * ads_trans_norm) if ads_trans_norm > 1e-14 else 0.0
        mode_reports.append({
            "mode_idx": int(mode_idx),
            "freq": freqs[mode_idx],
            "bond_proj": float(bond_proj),
            "drift_proj": float(drift_proj),
            "max_atom": max_atom,
            "max_atom_symbol": ads[max_atom].symbol,
            "mode": mode,
        })

    if len(mode_reports) == 0:
        return None
    primary = min(mode_reports, key=lambda x: np.real(x["freq"]))
    return {"all": mode_reports, "primary": primary}


def _displace_and_relax_imag_mode(ctx, atoms, mode, sign, out_xyz, rec_bond_global, site_pos, fmax=0.10):
    mode_norm = np.linalg.norm(mode)
    if mode_norm < 1e-14:
        return None
    disp_atoms = atoms.copy()
    scale = sign * ctx.imag_mode_displacement
    disp_atoms.set_positions(disp_atoms.get_positions() + scale * (mode / mode_norm))
    write(out_xyz.replace(".xyz", "_init.xyz"), disp_atoms)
    disp_atoms.calc = DP(model=MODEL)
    disp_atoms.set_constraint(ctx._build_constraints(disp_atoms, with_internal_bonds=False))
    try:
        opt = MDMin(disp_atoms, dt=0.05)
        opt.run(fmax=fmax, steps=ctx.imag_mode_relax_steps)
    except Exception:
        pass
    write(out_xyz, disp_atoms)

    i, j = rec_bond_global
    _, distances = get_distances(disp_atoms.get_positions(), None, disp_atoms.get_cell(), [True, True, True])
    d_broken = float(distances[i, j])
    normal = np.array(ctx.surface_normal, dtype=float)
    normal /= max(np.linalg.norm(normal), 1e-12)
    ads_positions = disp_atoms.get_positions()[len(ctx.slab.stru):]
    ads_center = np.mean(ads_positions, axis=0)
    height = float(np.dot(ads_center - np.array(site_pos, dtype=float), normal))
    return {"broken_bond": d_broken, "ads_height": height, "energy": float(disp_atoms.get_potential_energy())}


def _ts_bond_upper_bound(ctx, d_ref, atom_i_number, atom_j_number):
    d_ref = float(d_ref)
    r_sum = float(covalent_radii[int(atom_i_number)] + covalent_radii[int(atom_j_number)])
    base_limit = max(ctx.ts_bond_max_scale_ref * d_ref, ctx.ts_bond_max_scale_covalent * r_sum)
    additive_cap = d_ref + ctx.ts_bond_max_additive_cap
    return min(base_limit, additive_cap)


def _is_ts_bond_overstretched(ctx, d_now, d_ref, atom_i_number, atom_j_number):
    d_max = _ts_bond_upper_bound(ctx, d_ref, atom_i_number, atom_j_number)
    return bool(float(d_now) > float(d_max)), float(d_max)


def _ts_bond_dissociation_threshold(ctx, d_ref, atom_i_number, atom_j_number):
    d_ref = float(d_ref)
    r_sum = float(covalent_radii[int(atom_i_number)] + covalent_radii[int(atom_j_number)])
    return min(
        ctx.ts_endpoint_dissoc_scale_ref * d_ref,
        ctx.ts_endpoint_dissoc_scale_covalent * r_sum,
        d_ref + ctx.ts_endpoint_dissoc_additive,
    )


def _imag_frequency_magnitude(freq):
    if np.iscomplexobj(freq):
        return abs(float(np.imag(freq)))
    val = float(np.real(freq))
    return abs(val) if val < 0.0 else 0.0


def _significant_imag_frequencies(ctx, freqs):
    cutoff = float(ctx.imag_freq_significant_cutoff)
    return [f for f in freqs if _imag_frequency_magnitude(f) >= cutoff]
