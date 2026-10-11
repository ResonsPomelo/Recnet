"""Adsorption-rejected sites remain TS backups without affecting adsorption ranking."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from handlers import adsorption
from handlers.ts import generate_rxn_ts_guesses_ccqn
import synth_case


class EnergyProbe(Calculator):
    implemented_properties = ["energy"]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": float(atoms.info["probe_energy"])}


def make_context(tmp_path, sites, dissociated, monkeypatch):
    ads_dir = tmp_path / "rxn" / "Adsorbates" / "sp"
    ads_dir.mkdir(parents=True)
    for site in set(sites):
        (ads_dir / f"{site}_opt.xyz").touch()

    def fake_read(path):
        site = int(Path(path).name.split("_")[0].split(".")[0])
        atoms = Atoms("FeH2", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 1.8]], cell=[8, 8, 8])
        atoms.info["probe_energy"] = float(site)
        return atoms

    monkeypatch.setattr(adsorption, "read", fake_read)
    monkeypatch.setattr(adsorption, "write", lambda *args, **kwargs: None)
    monkeypatch.setattr(adsorption, "DP", EnergyProbe)

    slab = SimpleNamespace(
        unique_sites={"idx": sites},
        stru=Atoms("Fe", positions=[[0, 0, 0]], cell=[8, 8, 8]),
        get_site=lambda site: [float(site), 0, 0],
    )
    rxn = {"reactant_species": ["sp"], "product_species": [], "broken_bond": [0, 1]}
    ctx = SimpleNamespace(
        path=str(tmp_path), slab=slab,
        ads_templates={"sp": {"atoms": Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.8]]), "ad_idx": [0, 1]}},
        rxns_dict=[rxn], top_x=1, enable_rotation_enum_ads=False,
        _adsorbate_dir_candidates=lambda sp: [str(ads_dir)],
        _build_site_and_center_anchors=lambda *args: ([], []),
        _build_constraints=lambda *args, **kwargs: [],
        _tagged_stem=lambda stem: stem,
        _site_lift_vector=lambda: np.zeros(3),
        _is_adsorbate_structure_intact=lambda path, sp: (
            int(Path(path).name.split("_")[0]) not in dissociated, "dissociated"
        ),
        _load_ts_records=lambda: [],
        _flush_ts_records=lambda: None,
        rxn_key_by_index=["reaction-key"],
    )
    handler = adsorption.AdsorptionHandler(ctx)
    monkeypatch.setattr(handler, "_is_adsorbate_close_to_surface", lambda *args, **kwargs: (True, 1.0))
    return ctx, handler


def test_dissociated_sites_are_ts_backups_after_intact_ranked_site(tmp_path, monkeypatch):
    ctx, handler = make_context(tmp_path, [2, 1, 3], {2, 3}, monkeypatch)
    handler.run()
    rxn = ctx.rxns_dict[0]
    assert ctx.valid_sites == {"sp": [1]}
    assert ctx.valid_sites_all == {"sp": [1]}
    assert ctx.dissociated_sites == {"sp": [2, 3]}
    assert rxn["valid_reactant_sites"] == [1]
    assert rxn["valid_reactant_sites_all"] == [1, 2, 3]

    # Exercise the real TS site's selection/iteration without entering any
    # optimizer. Each get_site failure is contained by the TS retry loop.
    attempted = []

    def reject_site(site):
        attempted.append(site)
        raise RuntimeError("stop before scientific calculation")

    ctx.slab.get_site = reject_site
    generate_rxn_ts_guesses_ccqn(ctx)
    assert attempted == [1, 2, 3]


def test_backup_sites_are_deduplicated_in_adsorption_attempt_order(tmp_path, monkeypatch):
    ctx, handler = make_context(tmp_path, [3, 1, 3, 2, 2], {3, 2}, monkeypatch)
    handler.run()
    assert ctx.dissociated_sites == {"sp": [3, 3, 2, 2]}
    assert ctx.valid_sites_all == {"sp": [1]}
    assert ctx.rxns_dict[0]["valid_reactant_sites_all"] == [1, 3, 2]


def test_empty_site_candidates_stay_empty(tmp_path, monkeypatch):
    ctx, handler = make_context(tmp_path, [], set(), monkeypatch)
    handler.run()
    assert ctx.dissociated_sites == {}
    assert ctx.valid_sites == {"sp": []}
    assert ctx.valid_sites_all == {"sp": []}
    assert ctx.rxns_dict[0]["valid_reactant_sites_all"] == []


def test_real_context_initializes_empty_backup_registry(case):
    ctx = synth_case.make_context(case)
    assert ctx.dissociated_sites == {}
