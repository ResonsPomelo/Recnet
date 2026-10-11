"""Single-bond scission channel enumeration.

Contract (verified, see ``Recnet/network_inputs/README.md``):

* one channel == exactly one bond (one graph edge) broken;
* the products are the two resulting fragments, and every fragment must already
  exist in the species table (fragment closure);
* reactions are emitted in the *breaking* direction (precursor -> two
  fragments) because ``handlers/ts.py`` anchors the TS search on
  ``reactant_species[0]`` and ``handlers/energy.py`` requires at least two
  product species. A forming step such as ``CH2 + H -> CH3`` therefore appears
  as ``CH3 -> CH2 + H``; the barrier is direction-independent.

Bond order is recorded as metadata but is *not* part of the channel identity:
the pipeline detects bonds from distances only, and the cluster gold standard
demo writes ``CO* -> C* + O*`` with a single ``broken_bond`` atom pair for the
C=O connection. ``--bond-order-policy single`` restricts enumeration to
order-1 edges if a strict single-bond reading is ever required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .species import BOND_TYPES, Species, fragment_identity, split_fragments


@dataclass(frozen=True)
class Channel:
    """One breaking-direction elementary step."""

    reactant: str  # species key
    products: Tuple[str, ...]  # species keys, atom-0-first fragment ordering
    broken_bond: Tuple[int, int]  # indices inside the REACTANT template
    bond_type: str  # C-H | C-C | C-O | O-H | H-H
    bond_order: float
    reactant_label: str
    product_label: str
    reactant_is_gas: bool

    @property
    def key(self) -> Tuple[str, Tuple[str, ...], str]:
        return (self.reactant, self.products, self.bond_type)

    def as_yaml_record(self, sp_ids: Dict[str, str]) -> Dict[str, object]:
        """Field-for-field match with the gold standard reaction record.

        ``reactant_species`` / ``product_species`` must carry ``sp_xxx`` ids:
        ``handlers/ts.py`` looks the reactant template up as
        ``ctx.ads_templates[rxn["reactant_species"][0]]``, and that mapping is
        keyed by the species dict keys of the yaml.
        """
        return {
            "reactant": self.reactant_label,
            "product": self.product_label,
            "broken_bond": [int(self.broken_bond[0]), int(self.broken_bond[1])],
            "reactant_species": [sp_ids[self.reactant]],
            "product_species": [sp_ids[key] for key in self.products],
        }


def bond_type_of(symbol_a: str, symbol_b: str) -> str:
    """Normalised element-pair label, or ``""`` if the pair is out of scope."""
    pair = frozenset((symbol_a, symbol_b))
    for first, second in BOND_TYPES:
        if pair == frozenset((first, second)):
            return f"{first}-{second}"
    return ""


def label_of(species: Species, adsorbed_suffix: str = "*", gas_suffix: str = "(g)") -> str:
    """Display-only label (the pipeline never reads ``reactant``/``product``)."""
    return f"{species.name}{gas_suffix if species.is_gas else adsorbed_suffix}"


def preferred_entries(species_list: Sequence[Species]) -> Dict[str, Species]:
    """Map molecule identity -> the entry that owns its channels.

    The manifest requires gas reference entries for CO and C2H4 *and* adsorbed
    entries for the same molecules (CO*, CH2CH2*). Enumerating both would emit
    duplicated channels, so each molecule is enumerated exactly once: through
    its surface-bound entry when one exists, otherwise through its gas entry.
    """
    preferred: Dict[str, Species] = {}
    for species in species_list:
        current = preferred.get(species.identity)
        if current is None or (current.is_gas and not species.is_gas):
            preferred[species.identity] = species
    return preferred


def deduplicated_entries(species_list: Sequence[Species]) -> List[Dict[str, str]]:
    """Entries that do not own channels, with the owner they defer to."""
    preferred = preferred_entries(species_list)
    report = []
    for species in species_list:
        owner = preferred[species.identity]
        if owner.key != species.key:
            report.append(
                {
                    "entry": species.key,
                    "name": species.name,
                    "role": "gas-reference",
                    "same_molecule_as": owner.key,
                }
            )
    return report


def _resolve_fragment(
    component: Sequence[int],
    species: Species,
    identity_to_key: Dict[str, str],
) -> str:
    identity = fragment_identity(species.symbols, species.edges, component)
    return identity_to_key.get(identity, "")


def enumerate_channels(
    species_list: Sequence[Species],
    bond_order_policy: str = "any",
) -> Tuple[List[Channel], List[Dict[str, object]]]:
    """Enumerate one-bond scission channels.

    Returns ``(channels, dropped)`` where ``dropped`` records candidate cuts
    whose fragments are not in the species table, so nothing disappears
    silently.
    """
    if bond_order_policy not in {"any", "single"}:
        raise ValueError(f"unknown bond_order_policy {bond_order_policy!r}")

    preferred = preferred_entries(species_list)
    # Fragments stay on the surface after a scission, so they must resolve to
    # the surface-bound entry when a molecule has both an adsorbed and a gas
    # entry (CO -> CO*, not CO(g); CH2CH2* rather than C2H4(g)).
    identity_to_key = {identity: entry.key for identity, entry in preferred.items()}
    by_key = {species.key: species for species in species_list}

    channels: List[Channel] = []
    dropped: List[Dict[str, object]] = []

    for species in species_list:
        if preferred[species.identity].key != species.key:
            continue  # gas reference duplicate: channels belong to its owner

        seen: Dict[Tuple[Tuple[str, ...], str, float], bool] = {}
        for i, j, order in species.bonds:
            pair = tuple(sorted((int(i), int(j))))
            element_type = bond_type_of(species.symbols[pair[0]], species.symbols[pair[1]])
            if not element_type:
                dropped.append({
                    "reactant": species.key, "broken_bond": list(pair),
                    "reason": "element pair outside C-H/C-C/C-O/O-H/H-H",
                })
                continue
            if bond_order_policy == "single" and order != 1.0:
                dropped.append({
                    "reactant": species.key, "broken_bond": list(pair),
                    "reason": f"bond order {order:g} excluded by policy 'single'",
                })
                continue

            components = split_fragments(species.natoms, species.edges, pair)
            if len(components) < 2:
                dropped.append({
                    "reactant": species.key, "broken_bond": list(pair),
                    "reason": "cut does not separate the molecule",
                })
                continue

            fragment_keys = [
                _resolve_fragment(component, species, identity_to_key)
                for component in components
            ]
            if not all(fragment_keys):
                missing = [
                    fragment_identity(species.symbols, species.edges, component)
                    for component, key in zip(components, fragment_keys)
                    if not key
                ]
                dropped.append({
                    "reactant": species.key, "broken_bond": list(pair),
                    "reason": "fragment identity absent from species table",
                    "missing_fragment_identity": missing,
                })
                continue

            # Components are returned sorted by their smallest index, so the
            # fragment holding atom 0 (or the lowest index) comes first -- this
            # reproduces the gold standard ordering "C* O*" for CO* -> C* + O*.
            signature = (tuple(fragment_keys), element_type, order)
            if signature in seen:
                # Symmetry-equivalent cut (e.g. the four C-H bonds of ethylene).
                continue
            seen[signature] = True

            channels.append(
                Channel(
                    reactant=species.key,
                    products=tuple(fragment_keys),
                    broken_bond=pair,
                    bond_type=element_type,
                    bond_order=float(order),
                    reactant_label=label_of(species),
                    product_label=" ".join(label_of(by_key[key]) for key in fragment_keys),
                    reactant_is_gas=species.is_gas,
                )
            )

    return channels, dropped


def element_balance(
    channel: Channel, by_key: Dict[str, Species]
) -> Dict[str, object]:
    """Mass-conservation evidence for one channel."""
    reactant = by_key[channel.reactant]
    product_counts: Dict[str, int] = {}
    for key in channel.products:
        for symbol, count in by_key[key].counts.items():
            product_counts[symbol] = product_counts.get(symbol, 0) + count

    reactant_edges = len(reactant.edges)
    product_edges = sum(len(by_key[key].edges) for key in channel.products)

    return {
        "reactant": channel.reactant,
        "reactant_elements": dict(sorted(reactant.counts.items())),
        "product_elements": dict(sorted(product_counts.items())),
        "products": list(channel.products),
        "elements_conserved": dict(sorted(reactant.counts.items()))
        == dict(sorted(product_counts.items())),
        "bonds_broken": reactant_edges - product_edges,
        "one_bond_broken": (reactant_edges - product_edges) == 1,
        "at_least_two_fragments": len(channel.products) >= 2,
    }


def coverage_by_species(channels: Iterable[Channel]) -> Dict[str, Dict[str, int]]:
    """Per species: total channels and counts per bond type."""
    table: Dict[str, Dict[str, int]] = {}
    for channel in channels:
        row = table.setdefault(channel.reactant, {})
        row["total"] = row.get("total", 0) + 1
        row[channel.bond_type] = row.get(channel.bond_type, 0) + 1
    return table
