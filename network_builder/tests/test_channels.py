"""Channel enumeration invariants: mass balance, single bond, closure, order."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from network_builder.channels import (
    deduplicated_entries,
    element_balance,
    enumerate_channels,
    preferred_entries,
)
from network_builder.species import SEED, build_species_table

REQUIRED_BOND_TYPES = {"C-H", "C-C", "C-O", "O-H", "H-H"}
GAS_ONLY_REACTANTS = {"H2", "H2O", "CH4", "C2H6"}
ATOMIC_SPECIES = {"C", "O", "H"}
DUPLICATE_GAS_ENTRIES = {"C2H4", "CO_GAS"}


def atomic_number(symbol: str) -> int:
    return atomic_numbers[symbol]


def test_all_required_bond_types_are_covered(channels):
    enumerated, _ = channels
    present = {channel.bond_type for channel in enumerated}
    assert present == REQUIRED_BOND_TYPES, f"missing {REQUIRED_BOND_TYPES - present}"


def test_mass_conservation_and_single_bond_break(channels, species_by_key):
    enumerated, _ = channels
    assert enumerated
    for channel in enumerated:
        record = element_balance(channel, species_by_key)
        assert record["elements_conserved"], f"{channel.reactant}: elements not conserved"
        assert record["one_bond_broken"], f"{channel.reactant}: not exactly one bond"
        assert record["at_least_two_fragments"]


def test_products_exist_in_species_table_and_are_two(channels, species_by_key):
    enumerated, _ = channels
    for channel in enumerated:
        assert len(channel.products) >= 2
        for key in channel.products:
            assert key in species_by_key


def test_broken_bond_is_a_declared_bond_of_the_reactant_template(channels, species_by_key):
    enumerated, _ = channels
    for channel in enumerated:
        reactant = species_by_key[channel.reactant]
        assert channel.broken_bond in reactant.edges, (
            f"{channel.reactant}: broken_bond {channel.broken_bond} is not a bond "
            "of the reactant template"
        )
        assert max(channel.broken_bond) < reactant.natoms


def test_broken_bond_is_visible_to_pipeline_distance_rule(
    channels, species_by_key, pipeline_geometry
):
    """Uses the real ``utils/geometry.get_bond_connections`` from Recnet."""
    enumerated, _ = channels
    for channel in enumerated:
        reactant = species_by_key[channel.reactant]
        atoms = Atoms(symbols="".join(reactant.symbols),
                      positions=np.array(reactant.positions, dtype=float), pbc=False)
        detected = {
            tuple(sorted((int(i), int(j))))
            for i, j in pipeline_geometry.get_bond_connections(
                atoms, shift=0, cutoff=1.2, bond_type="nosurf"
            )
        }
        pair = tuple(sorted(channel.broken_bond))
        assert pair in detected, (
            f"{channel.reactant}: broken_bond {pair} invisible to handlers/ts.py"
        )


def test_h2_is_the_tightest_geometric_margin(channels, species_by_key):
    enumerated, _ = channels
    ratios = {}
    for channel in enumerated:
        reactant = species_by_key[channel.reactant]
        positions = np.array(reactant.positions, dtype=float)
        i, j = channel.broken_bond
        distance = float(np.linalg.norm(positions[i] - positions[j]))
        cutoff = float(
            (covalent_radii[atomic_number(reactant.symbols[i])]
             + covalent_radii[atomic_number(reactant.symbols[j])]) * 1.2
        )
        ratios[channel.reactant] = distance / cutoff
    worst = max(ratios, key=ratios.get)
    assert ratios[worst] < 1.0
    assert worst == "H2", (
        "H2 is expected to remain the tightest case; re-check the reported "
        f"margins: {sorted(ratios.items(), key=lambda kv: -kv[1])[:3]}"
    )


def test_channels_are_unique(channels):
    enumerated, _ = channels
    keys = [(c.reactant, c.products, c.bond_type) for c in enumerated]
    assert len(keys) == len(set(keys)), "duplicated channel emitted"
    pairs = [(c.reactant, c.broken_bond) for c in enumerated]
    assert len(pairs) == len(set(pairs)), "the same bond emitted twice for one species"


def test_atomic_species_have_no_channels_and_molecules_have_some(channels, species_list):
    enumerated, _ = channels
    coverage = {}
    for channel in enumerated:
        coverage[channel.reactant] = coverage.get(channel.reactant, 0) + 1
    skip = ATOMIC_SPECIES | DUPLICATE_GAS_ENTRIES
    for species in species_list:
        if species.key in skip:
            continue
        assert coverage.get(species.key, 0) >= 1, f"{species.key} has no channel"
    for key in ATOMIC_SPECIES:
        assert coverage.get(key, 0) == 0


def test_gas_reference_duplicates_do_not_own_channels(channels, species_list):
    enumerated, _ = channels
    reactants = {channel.reactant for channel in enumerated}
    assert not (DUPLICATE_GAS_ENTRIES & reactants)
    assert {"CO", "CH2CH2"} <= reactants
    duplicates = deduplicated_entries(species_list)
    assert {entry["entry"] for entry in duplicates} == {"C2H4", "CO_GAS"}
    assert {entry["same_molecule_as"] for entry in duplicates} == {"CO", "CH2CH2"}


def test_gas_only_species_are_enumerated_through_their_single_entry(channels):
    enumerated, _ = channels
    reactants = {channel.reactant for channel in enumerated}
    assert GAS_ONLY_REACTANTS <= reactants
    gas_channels = [c for c in enumerated if c.reactant_is_gas]
    assert len(gas_channels) == 5  # H2, H2O, CH4, C2H6 (C-H and C-C)


def test_fragments_resolve_to_the_surface_bound_entry(channels):
    """A scission fragment stays adsorbed: CO* / CH2CH2*, never the gas entry."""
    enumerated, _ = channels
    products = {key for channel in enumerated for key in channel.products}
    assert "CO_GAS" not in products and "C2H4" not in products
    assert "CO" in products and "CH2CH2" in products
    ch2ch3 = [c for c in enumerated if c.reactant == "CH2CH3"]
    assert ("CH2CH2", "H") in [c.products for c in ch2ch3]
    hco = [c for c in enumerated if c.reactant == "HCO"]
    assert ("CO", "H") in [c.products for c in hco]


def test_fragment_closure_reports_every_drop(channels, species_by_key):
    """Nothing is silently discarded: drops carry a reason."""
    _enumerated, dropped = channels
    for record in dropped:
        assert record["reactant"] in species_by_key
        assert record["reason"]
    # The only expected drops: removing an alpha-H of ethyl yields the CH3-CH
    # (ethylidene) skeleton, which is not part of the declared species set.
    assert {record["reactant"] for record in dropped} == {"CH2CH3"}
    assert all(record["reason"] == "fragment identity absent from species table"
               for record in dropped)


def test_bond_order_policy_single_is_supported(species_list, channels):
    enumerated, _ = channels
    single, _ = enumerate_channels(species_list, bond_order_policy="single")
    assert len(single) < len(enumerated)
    assert all(channel.bond_order == 1.0 for channel in single)
    assert not [c for c in single if c.reactant == "CO"]


def test_enumeration_is_deterministic(species_list):
    first, dropped_first = enumerate_channels(build_species_table(seed=SEED))
    second, dropped_second = enumerate_channels(build_species_table(seed=SEED))
    assert [(c.reactant, c.products, c.broken_bond, c.bond_type) for c in first] == \
        [(c.reactant, c.products, c.broken_bond, c.bond_type) for c in second]
    assert dropped_first == dropped_second


def test_preferred_entry_is_the_surface_bound_form(species_list):
    preferred = preferred_entries(species_list)
    by_key = {species.key: species for species in species_list}
    assert preferred[by_key["CO"].identity].key == "CO"
    assert preferred[by_key["CH2CH2"].identity].key == "CH2CH2"
    assert preferred[by_key["CH4"].identity].key == "CH4"  # gas-only molecule


@pytest.mark.parametrize(
    "reactant,products,bond_type",
    [
        ("CO", ("C", "O"), "C-O"),
        ("CCH3", ("CCH2", "H"), "C-H"),
        ("CH2CH3", ("CH3", "CH2"), "C-C"),
        ("H2", ("H", "H"), "H-H"),
        ("CH4", ("CH3", "H"), "C-H"),
    ],
)
def test_named_channel_expectations(reactant, products, bond_type, channels):
    enumerated, _ = channels
    match = [c for c in enumerated
             if c.reactant == reactant and c.products == products
             and c.bond_type == bond_type]
    assert len(match) == 1, f"expected exactly one {reactant} -> {products} ({bond_type})"
