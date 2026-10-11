"""Species parsing, template geometry and determinism."""

from __future__ import annotations

import re

import pytest
from ase.io import read

from network_builder.species import (
    SEED,
    SPECS,
    build_species_table,
    identity_key,
    template_text,
)

#: Species the parent manifest requires (name -> expected formula).
REQUIRED_C1 = ["C", "CH", "CH2", "CH3", "CO", "COH", "HCO", "H2CO", "H3CO", "O", "OH"]
REQUIRED_C2 = ["CC", "CCH", "CCH2", "CCH3", "CHCH", "CHCH2", "CH2CH2", "CH2CH3",
               "CCO", "CHCO", "CH2CO"]
REQUIRED_GAS = ["H2", "H2O", "CH4", "C2H4", "C2H6", "CO"]


def test_every_spec_parses_with_declared_formula(species_list):
    """rdkit parse + declared formula + atom counts must agree (catches
    implicit-hydrogen traps such as ``[CH3]O`` -> methanol)."""
    assert len(species_list) == len(SPECS)
    for species in species_list:
        assert species.formula == next(
            spec.formula for spec in SPECS if spec.key == species.key
        )


@pytest.mark.parametrize("key", REQUIRED_C1 + REQUIRED_C2 + REQUIRED_GAS)
def test_manifest_species_present(key, species_by_key):
    assert key in species_by_key, f"required species {key} missing from the table"


def test_gas_entries_have_empty_ad_idx(species_by_key):
    # The manifest lists CO among the gases; the species table keeps the
    # adsorbed CO* (ad_idx [0], matching the gold standard demo) and the gas
    # reference under a distinct key, so map names to entry keys explicitly.
    gas_entry_keys = {name: name for name in REQUIRED_GAS if name != "CO"}
    gas_entry_keys["CO"] = "CO_GAS"
    for name, key in gas_entry_keys.items():
        species = species_by_key[key]
        assert species.ad_idx == (), f"{name} gas entry must have ad_idx == []"
        assert species.is_gas


def test_adsorbates_declare_adsorption_anchor(species_by_key, species_list):
    """Every non-gas entry points at heavy atoms (atomic H is the sole exception)."""
    for species in species_list:
        if species.is_gas:
            continue
        assert species.ad_idx, f"{species.key} has no adsorption anchor"
        for idx in species.ad_idx:
            if any(symbol != "H" for symbol in species.symbols):
                assert species.symbols[idx] != "H", f"{species.key}: anchor on hydrogen"
        assert sorted(set(species.ad_idx)) == sorted(species.ad_idx)


def test_identities_are_unique_per_molecule(species_list):
    identities = {}
    for species in species_list:
        identities.setdefault(species.identity, []).append(species.key)
    # Only the two required gas duplicates may share an identity.
    duplicated = {k: v for k, v in identities.items() if len(v) > 1}
    assert set(tuple(sorted(v)) for v in duplicated.values()) == {
        ("C2H4", "CH2CH2"),
        ("CO", "CO_GAS"),
    }


def test_isomer_identities_differ(species_by_key):
    """Connectivity + hydrogen distribution must separate the tricky isomers."""
    pairs = [("HCO", "COH"), ("CCH2", "CHCH"), ("CCH3", "CHCH2"),
             ("CHCO", "CCH3"), ("H2CO", "COH")]
    for left, right in pairs:
        assert species_by_key[left].identity != species_by_key[right].identity


def test_atom_order_puts_hydrogens_last(species_by_key):
    """rdkit appends hydrogens, so heavy-atom indices are stable and readable."""
    for key in ("CH3", "H2CO", "CH2CH3", "CH2CO"):
        species = species_by_key[key]
        first_h = next(i for i, s in enumerate(species.symbols) if s == "H")
        assert all(s == "H" for s in species.symbols[first_h:]), key


def test_every_declared_bond_is_visible_to_the_pipeline_rule(species_by_key):
    """Declared bonds must satisfy the distance rule used by handlers/ts.py."""
    import numpy as np
    from ase.data import covalent_radii

    for species in species_by_key.values():
        positions = np.array(species.positions, dtype=float)
        for i, j, _order in species.bonds:
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            cutoff = float(
                (covalent_radii[atomic_number(species.symbols[i])]
                 + covalent_radii[atomic_number(species.symbols[j])]) * 1.2
            )
            assert distance < cutoff, (
                f"{species.key}: bond {i}-{j} at {distance:.3f} A is not detected "
                f"under the pipeline cutoff {cutoff:.3f} A"
            )


def atomic_number(symbol: str) -> int:
    from ase.data import atomic_numbers

    return atomic_numbers[symbol]


def test_template_text_matches_gold_standard_format(species_by_key):
    """Byte-level format check against the cluster demo templates."""
    text = template_text(species_by_key["CO"])
    lines = text.splitlines()
    assert lines[0] == "2"
    assert lines[1] == 'Properties=species:S:1:pos:R:3 pbc="F F F"'
    assert re.fullmatch(r"C\s+0\.00000000\s+0\.00000000\s+0\.00000000", lines[2])
    assert len(lines) == 4  # count + header + 2 atoms
    assert "Lattice" not in text and "cell" not in text


def test_template_roundtrip_preserves_symbols_and_anchors(species_by_key):
    for species in species_by_key.values():
        path_out = None
        import ase.io as ase_io
        import io

        buffer = io.StringIO()
        buffer.write(template_text(species))
        buffer.seek(0)
        atoms = ase_io.read(buffer, format="extxyz")
        assert list(atoms.get_chemical_symbols()) == list(species.symbols)
        positions = atoms.get_positions()
        # the anchor (or centre for gas) sits at the origin
        if species.ad_idx:
            centre = positions[list(species.ad_idx)].mean(axis=0)
        else:
            centre = positions.mean(axis=0)
        assert max(abs(centre)) < 1e-6


def test_generation_is_deterministic(species_list):
    again = build_species_table(seed=SEED)
    assert [s.identity for s in again] == [s.identity for s in species_list]
    assert [template_text(s) for s in again] == [template_text(s) for s in species_list]
    assert [s.positions for s in again] == [s.positions for s in species_list]


def test_identity_key_is_independent_of_atom_order():
    """Order independence is what makes fragment matching robust."""
    forward = identity_key(("C", "O"), [(0, 1)])
    reversed_order = identity_key(("O", "C"), [(0, 1)])
    assert forward == reversed_order
    assert identity_key(("C", "H", "H", "H"), [(0, 1), (0, 2), (0, 3)]) == \
        identity_key(("H", "C", "H", "H"), [(1, 0), (1, 2), (1, 3)])
    # ... and different graphs must stay different.
    assert identity_key(("C", "H", "H", "H"), [(0, 1), (0, 2), (0, 3)]) != \
        identity_key(("C", "H", "H"), [(0, 1), (0, 2)])
    assert identity_key(("C", "H", "O"), [(0, 1), (0, 2)]) != \
        identity_key(("C", "H", "O"), [(0, 1), (1, 2)])
