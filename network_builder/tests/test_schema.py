"""Schema alignment with the cluster gold standard and artefact consistency.

Gold standard:
``SAI:/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/recnet-runs/ft2dpv22-demo-S/
prepared_data/prepared_rmg_data.yaml`` (read 2026-09-21):

    rxns:
    - reactant: CO*
      product: C* O*
      broken_bond: [0, 1]
      reactant_species: [sp_000]
      product_species: [sp_001, sp_002]
    species:
      sp_000: {name: CO, adjlist: '', template_xyz: ads_templates/sp_000.xyz, ad_idx: [0]}
"""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from network_builder.build import build_dataset
from network_builder.species import SEED

from conftest import NETWORK_INPUTS, YAML_NAME

RXN_FIELDS = {"reactant", "product", "broken_bond", "reactant_species", "product_species"}
SPECIES_FIELDS = {"name", "adjlist", "template_xyz", "ad_idx"}


def test_generated_yaml_matches_gold_standard_shape(manifest):
    payload = yaml.safe_load(open(manifest["_yaml_path"], encoding="utf-8"))
    assert set(payload) == {"rxns", "species"}
    assert payload["rxns"], "no reactions generated"
    for rxn in payload["rxns"]:
        assert set(rxn) == RXN_FIELDS, f"unexpected reaction fields: {set(rxn)}"
        assert isinstance(rxn["reactant"], str) and isinstance(rxn["product"], str)
        assert isinstance(rxn["broken_bond"], list) and len(rxn["broken_bond"]) == 2
        assert all(isinstance(i, int) for i in rxn["broken_bond"])
        assert len(rxn["reactant_species"]) == 1
        assert len(rxn["product_species"]) >= 2  # handlers/energy.py requirement
    for sp_id, record in payload["species"].items():
        assert sp_id.startswith("sp_")
        assert set(record) == SPECIES_FIELDS, f"unexpected species fields: {set(record)}"
        assert record["adjlist"] == ""  # gold standard clears this field
        assert isinstance(record["ad_idx"], list)
        assert all(isinstance(i, int) for i in record["ad_idx"])
        assert record["template_xyz"] == f"ads_templates/{sp_id}.xyz"


def test_reaction_references_resolve(manifest):
    payload = yaml.safe_load(open(manifest["_yaml_path"], encoding="utf-8"))
    species = payload["species"]
    for rxn in payload["rxns"]:
        reactant_id = rxn["reactant_species"][0]
        assert reactant_id in species
        for sp_id in rxn["product_species"]:
            assert sp_id in species, f"{sp_id} referenced but not defined"
        template = species[reactant_id]["template_xyz"]
        assert max(rxn["broken_bond"]) < _template_natoms(manifest, template)


def _template_natoms(manifest, template: str) -> int:
    import os

    path = os.path.join(os.path.dirname(manifest["_yaml_path"]), template)
    with open(path, encoding="utf-8") as handle:
        return int(handle.readline().strip())


def test_templates_exist_and_are_extxyz(manifest):
    import pathlib

    template_dir = pathlib.Path(manifest["_yaml_path"]).parent
    records = manifest["files"]["templates"]
    assert len(records) == manifest["counts"]["species_entries"]
    for record in records:
        path = template_dir / record["path"]
        assert path.exists(), f"missing template {path}"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert int(lines[0]) == record["natoms"] == len(lines) - 2
        assert lines[1] == 'Properties=species:S:1:pos:R:3 pbc="F F F"'
        assert all(len(line.split()) == 4 for line in lines[2:])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_gas_entries_serialise_with_empty_ad_idx(manifest):
    payload = yaml.safe_load(open(manifest["_yaml_path"], encoding="utf-8"))
    by_key = {entry["key"]: entry["sp_id"] for entry in manifest["species_entries"]}
    for name in ("H2", "H2O", "CH4", "C2H4", "C2H6", "CO_GAS"):
        sp_id = by_key[name]
        assert payload["species"][sp_id]["ad_idx"] == []


def test_manifest_checks_pass(manifest):
    checks = manifest["checks"]
    balance = checks["mass_conservation"]
    assert balance["checked"] == manifest["counts"]["rxns"]
    assert balance["failures"] == []
    assert all(record["elements_conserved"] for record in balance["per_channel"])
    assert all(record["one_bond_broken"] for record in balance["per_channel"])
    visibility = checks["broken_bond_visible_to_pipeline"]
    assert visibility["checked"] == manifest["counts"]["rxns"]
    assert visibility["failures"] == []
    assert checks["one_bond_broken_all_channels"]
    assert checks["products_at_least_two_all_channels"]
    assert checks["fragment_closure"]["dropped_count"] == len(
        checks["fragment_closure"]["dropped_candidates"]
    )


def test_manifest_records_element_counts_per_channel(manifest):
    records = manifest["checks"]["mass_conservation"]["per_channel"]
    assert len(records) == manifest["counts"]["rxns"]
    for record in records:
        assert record["reactant_elements"] == record["product_elements"]
        assert set(record["reactant_elements"]) <= {"C", "H", "O"}


def test_manifest_records_environment_and_hashes(manifest):
    environment = manifest["environment"]
    assert environment["etkdg_seed"] == SEED
    for field in ("python", "rdkit", "ase", "numpy", "pyyaml"):
        assert environment[field] not in ("", "unknown")
    assert len(manifest["files"]["yaml"]["sha256"]) == 64
    assert manifest["files"]["yaml"]["bytes"] > 0


def test_rebuild_is_byte_identical(tmp_path):
    first = build_dataset(out_dir=tmp_path / "a", yaml_name=YAML_NAME, seed=SEED)
    second = build_dataset(out_dir=tmp_path / "b", yaml_name=YAML_NAME, seed=SEED)
    assert first["files"]["yaml"]["sha256"] == second["files"]["yaml"]["sha256"]
    assert first["channels"] == second["channels"]
    assert first["files"]["templates"] == second["files"]["templates"]


@pytest.mark.skipif(
    not (NETWORK_INPUTS / YAML_NAME).exists(),
    reason="committed artefacts not generated yet",
)
def test_committed_artefacts_are_current(tmp_path):
    """The checked-in yaml/MANIFEST must match a fresh build (no stale inputs)."""
    fresh = build_dataset(out_dir=tmp_path, yaml_name=YAML_NAME, seed=SEED)
    committed_yaml = (NETWORK_INPUTS / YAML_NAME).read_text(encoding="utf-8")
    fresh_yaml = open(fresh["_yaml_path"], encoding="utf-8").read()
    assert hashlib.sha256(committed_yaml.encode()).hexdigest() == \
        hashlib.sha256(fresh_yaml.encode()).hexdigest()

    committed_manifest = json.loads((NETWORK_INPUTS / "MANIFEST.json").read_text(encoding="utf-8"))
    for key in ("counts", "channels", "paper_coverage", "species_entries"):
        assert committed_manifest[key] == fresh[key], f"MANIFEST {key} is stale"


def test_pipeline_lookup_sequence_succeeds(manifest, pipeline_geometry):
    """Replay the exact lookups handlers/ts.py and handlers/energy.py perform."""
    import os

    from ase.io import read

    payload = yaml.safe_load(open(manifest["_yaml_path"], encoding="utf-8"))
    base_dir = os.path.dirname(manifest["_yaml_path"])
    templates = {}
    for sp_id, record in payload["species"].items():
        path = os.path.join(base_dir, record["template_xyz"])
        templates[sp_id] = {
            "atoms": read(path),  # handlers/context.py reads the template this way
            "ad_idx": record["ad_idx"],
            "name": record["name"],
        }

    for rxn in payload["rxns"]:
        # handlers/ts.py:57-58
        sp_id = rxn["reactant_species"][0]
        rec_bond = rxn["broken_bond"]
        template = templates[sp_id]
        detected = {
            tuple(sorted((int(i), int(j))))
            for i, j in pipeline_geometry.get_bond_connections(
                template["atoms"], shift=0, cutoff=1.2, bond_type="nosurf"
            )
        }
        assert tuple(sorted(rec_bond)) in detected
        assert max(rec_bond) < len(template["atoms"])
        # handlers/energy.py:215-222
        assert len(rxn["reactant_species"]) >= 1
        assert len(rxn["product_species"]) >= 2


def test_species_keys_are_unique_like_handlers_context(manifest):
    """Replicate handlers/context.py::_species_key payload and require uniqueness."""
    import hashlib
    import json

    payload = yaml.safe_load(open(manifest["_yaml_path"], encoding="utf-8"))
    digests = {}
    for entry in manifest["species_entries"]:
        symbols = entry["symbols"]
        counts = {}
        for symbol in symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
        bonds = sorted([int(b["i"]), int(b["j"])] for b in entry["bonds"])
        bonds = [pair if pair[0] <= pair[1] else [pair[1], pair[0]] for pair in bonds]
        hill = _hill(counts)
        record = payload["species"][entry["sp_id"]]
        assert record["name"] == entry["name"]
        assert record["ad_idx"] == entry["ad_idx"]
        payload_for_key = {
            "name": str(entry["name"]).strip().lower(),
            "formula": hill,
            "atom_count": len(symbols),
            "ad_idx": sorted(int(i) for i in entry["ad_idx"]),
            "symbol_counts": [[k, counts[k]] for k in sorted(counts)],
            "bonds": bonds,
        }
        raw = json.dumps(payload_for_key, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True)
        digest = hashlib.md5(raw.encode()).hexdigest()[:12]
        digests.setdefault(digest, []).append(entry["sp_id"])
    collisions = {k: v for k, v in digests.items() if len(v) > 1}
    assert not collisions, f"species key collision: {collisions}"


def _hill(counts):
    order = [s for s in ("C", "H") if s in counts]
    order += sorted(s for s in counts if s not in order)
    return "".join(f"{s}{counts[s]}" if counts[s] != 1 else s for s in order)
