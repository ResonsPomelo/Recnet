"""
Adsorption stage: place adsorbates on surface sites, relax, filter intact structures.
"""
import os

import numpy as np
from ase.io import read, write
from ase.optimize import MDMin, QuasiNewton
import ase

from deepmd.calculator import DP
from .context import WorkflowContext, MODEL
from utils import constraints as constraint_utils


class AdsorptionHandler:
    """Handles the initial adsorbate guess generation and ranking."""

    def __init__(self, ctx: WorkflowContext):
        self.ctx = ctx

    def run(self):
        """For each species, place it on every unique site, relax, rank by energy."""
        ctx = self.ctx
        slab = ctx.slab
        os.makedirs(os.path.join(ctx.path, "rxn", "Adsorbates"), exist_ok=True)

        ctx.valid_sites = {}
        ctx.valid_sites_all = {}

        def _opt(path, prefix, site_bond_params_list, atom_bond_params_list=None,
                 surf_atom_num=0, constraints_in=None, center_bond_params_list=None):
            struct = read(os.path.join(path, prefix + ".xyz"))
            if constraints_in is None:
                constraints = ctx._build_constraints(struct, surf_atom_num=surf_atom_num)
            else:
                constraints = constraints_in
            struct.set_constraint(constraints)
            constraints = ctx._build_constraints(struct, surf_atom_num=surf_atom_num)

            if not os.path.exists(os.path.join(path, prefix + "_opt.xyz")):
                calc = constraint_utils.HarmonicallyForcedDP(
                    model=MODEL,
                    atom_bond_potentials=atom_bond_params_list,
                    site_bond_potentials=site_bond_params_list,
                    center_bond_potentials=center_bond_params_list,
                )
                struct.calc = calc
                try:
                    opt_md = MDMin(struct, dt=0.05)
                    opt_md.run(fmax=0.5, steps=70)
                    write(os.path.join(path, prefix + "_weakopt.xyz"), struct)
                    calc = DP(model=MODEL)
                    struct.set_constraint(constraints)
                    struct.calc = calc
                    opt_qn = QuasiNewton(struct)
                    opt_qn.run(fmax=0.05, steps=70)
                except ase.calculators.calculator.CalculationFailed:
                    write(os.path.join(path, prefix + "_opt_fail.xyz"), struct)
                    return None
                write(os.path.join(path, prefix + "_opt.xyz"), struct)
            else:
                struct = read(os.path.join(path, prefix + "_opt.xyz"))
                struct.set_constraint(constraints)
                struct.calc = DP(model=MODEL)
            try:
                return float(struct.get_potential_energy())
            except Exception:
                return None

        for sp_id, data in ctx.ads_templates.items():
            ads_dir = ctx._adsorbate_dir_candidates(sp_id)[0]
            os.makedirs(ads_dir, exist_ok=True)
            sites = slab.unique_sites["idx"]
            myslab = slab.stru.copy()
            energy = {}

            for site in sites:
                path = ads_dir
                prefix = f"{site}"
                tagged_prefix = ctx._tagged_stem(prefix)
                pos = slab.get_site(site)
                stru = data["atoms"].copy()

                site_bond_params_list, center_bond_params_list = ctx._build_site_and_center_anchors(
                    data, pos, len(myslab)
                )

                if ctx.enable_rotation_enum_ads:
                    stru, best_angle, best_energy = self._enumerate_azimuth_orientation(
                        base_stru=stru, ads_idx=data["ad_idx"], myslab=myslab, pos=pos,
                        site_bond_params_list=site_bond_params_list,
                        center_bond_params_list=center_bond_params_list,
                        atom_bond_params_list=[],
                    )
                    if best_energy is not None:
                        print(f"Ads enum {sp_id} site {site}: angle={best_angle:.1f} deg, E={best_energy:.6f} eV")

                stru.translate(np.array(pos, dtype=float) + ctx._site_lift_vector())
                ads = myslab + stru
                write(os.path.join(path, f"{tagged_prefix}.xyz"), ads)
                e_site = _opt(path, tagged_prefix, site_bond_params_list, [],
                              len(myslab), center_bond_params_list=center_bond_params_list)

                if e_site is not None:
                    opt_xyz = os.path.join(path, f"{tagged_prefix}_opt.xyz")
                    if os.path.exists(opt_xyz):
                        intact, reason = ctx._is_adsorbate_structure_intact(opt_xyz, sp_id)
                        if not intact:
                            print(f"Reject site {site} for {sp_id}: dissociated adsorbate after opt ({reason})")
                            e_site = None
                        else:
                            opt_atoms = read(opt_xyz)
                            close_ok, height = self._is_adsorbate_close_to_surface(
                                opt_atoms, sp_id, pos, max_height=2,
                            )
                            if not close_ok:
                                print(f"Site {site} for {sp_id} is too far from surface (height={height:.3f} A). Run rescue optimization.")
                                e_site = self._rescue_site(sp_id, data, site, pos, myslab,
                                                           opt_xyz, opt_atoms, height)
                energy[site] = e_site

            valid_energy = {k: v for k, v in energy.items() if v is not None}
            valid_sites = self._select_valid_sites(valid_energy, sites, ads_dir, sp_id, myslab)
            valid_sites_all = list(valid_sites)
            valid_sites_preferred = valid_sites_all[:ctx.top_x] if (ctx.top_x and ctx.top_x > 0) else valid_sites_all

            ctx.valid_sites_all[sp_id] = valid_sites_all
            ctx.valid_sites[sp_id] = valid_sites_preferred
            print(f"Valid sites for {sp_id}: preferred={valid_sites_preferred}, all={valid_sites_all}")

        for rxn in ctx.rxns_dict:
            rxn["valid_reactant_sites"] = []
            rxn["valid_reactant_sites_all"] = []
            rxn["valid_product_sites"] = []
            for sp_id in rxn["reactant_species"]:
                rxn["valid_reactant_sites"] = ctx.valid_sites.get(sp_id, slab.unique_sites["idx"])
                rxn["valid_reactant_sites_all"] = ctx.valid_sites_all.get(sp_id, slab.unique_sites["idx"])
            for sp_id in rxn["product_species"]:
                rxn["valid_product_sites"] = ctx.valid_sites.get(sp_id, slab.unique_sites["idx"])

    # ---- rescue & ranking helpers -------------------------------------------

    def _rescue_site(self, sp_id, data, site, pos, myslab, opt_xyz, opt_atoms, height):
        ctx = self.ctx
        rescue_atoms = opt_atoms.copy()
        normal = np.array(ctx.surface_normal, dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            normal = np.array(ctx.surface_normal, dtype=float)
            normal_norm = np.linalg.norm(normal)
        normal /= normal_norm

        target_height = 1.3
        pull_dist = max(0.0, float(height) - target_height)
        if pull_dist > 1e-8:
            rescue_pos = rescue_atoms.get_positions()
            rescue_pos[len(myslab):] -= pull_dist * normal
            rescue_atoms.set_positions(rescue_pos)

        rescue_site_params, rescue_center_params = ctx._build_site_and_center_anchors(data, pos, len(myslab))
        for p in rescue_site_params:
            p["k"] = max(float(p.get("k", 0.5)), 1.2)
        for p in rescue_center_params:
            p["k"] = max(float(p.get("k", 0.5)), 0.8)

        rescue_atoms.set_constraint(
            ctx._build_constraints(rescue_atoms, surf_atom_num=len(myslab), with_internal_bonds=False)
        )
        rescue_atoms.calc = constraint_utils.HarmonicallyForcedDP(
            model=MODEL, atom_bond_potentials=[],
            site_bond_potentials=rescue_site_params,
            center_bond_potentials=rescue_center_params,
        )
        try:
            rescue_md = MDMin(rescue_atoms, dt=0.05)
            rescue_md.run(fmax=0.4, steps=80)
            rescue_qn = QuasiNewton(rescue_atoms)
            rescue_qn.run(fmax=0.08, steps=80)
        except ase.calculators.calculator.CalculationFailed:
            pass

        rescue_atoms.calc = DP(model=MODEL)
        rescue_atoms.set_constraint(
            ctx._build_constraints(rescue_atoms, surf_atom_num=len(myslab), with_internal_bonds=False)
        )
        try:
            release_qn = QuasiNewton(rescue_atoms)
            release_qn.run(fmax=0.05, steps=60)
        except ase.calculators.calculator.CalculationFailed:
            pass

        write(opt_xyz, rescue_atoms)
        try:
            e_site = float(rescue_atoms.get_potential_energy())
        except Exception:
            return None

        if e_site is not None:
            intact2, reason2 = ctx._is_adsorbate_structure_intact(opt_xyz, sp_id)
            close_ok2, height2 = self._is_adsorbate_close_to_surface(rescue_atoms, sp_id, pos, max_height=2.2)
            if (not intact2) or (not close_ok2):
                print(f"Reject site {site} for {sp_id} after rescue: intact={intact2}, height={height2}, reason={reason2}")
                return None
        return e_site

    def _select_valid_sites(self, valid_energy, sites, ads_dir, sp_id, myslab):
        ctx = self.ctx
        if valid_energy:
            valid_sorted = dict(sorted(valid_energy.items(), key=lambda item: item[1]))
            return list(valid_sorted.keys())

        print(f"No intact adsorption structures found for {sp_id}; skip bondfix retry and force-select one fallback site.")
        fallback_energy = {}
        for site in sites:
            tagged_prefix = ctx._tagged_stem(f"{site}")
            opt_xyz = os.path.join(ads_dir, f"{tagged_prefix}_opt.xyz")
            if not os.path.exists(opt_xyz):
                continue
            try:
                atoms = read(opt_xyz)
                atoms.set_constraint(ctx._build_constraints(atoms, surf_atom_num=len(myslab), with_internal_bonds=False))
                atoms.calc = DP(model=MODEL)
                fallback_energy[site] = float(atoms.get_potential_energy())
            except Exception:
                continue

        if fallback_energy:
            chosen_site = min(fallback_energy, key=fallback_energy.get)
            print(f"Fallback for {sp_id}: select site {chosen_site} (E={fallback_energy[chosen_site]:.6f} eV) without bondfix retry.")
            return [chosen_site]
        if sites:
            print(f"Fallback for {sp_id}: no readable *_opt.xyz; force-select first site {sites[0]}.")
            return [sites[0]]
        print(f"Fallback for {sp_id}: no available site to select.")
        return []

    # ---- orientation helpers -----------------------------------------------

    def _enumerate_azimuth_orientation(
        self, base_stru, ads_idx, myslab, pos,
        site_bond_params_list, center_bond_params_list,
        atom_bond_params_list=None,
    ):
        from utils import geometry as geom
        ctx = self.ctx

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

    def _is_adsorbate_close_to_surface(self, atoms, sp_id, site_pos, max_height=2.2):
        ctx = self.ctx
        height = ctx._adsorbate_height_from_site(atoms, sp_id, site_pos)
        if height is None:
            return False, None
        return bool(height <= float(max_height)), float(height)

