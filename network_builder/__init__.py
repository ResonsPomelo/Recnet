"""Deterministic builder for RecNet "prepared" datasets (RMG-free path).

RecNet's stock data preparation (``Recnet/prepare_rmg_data.py``) needs both
``rdkit`` and RMG's ``molecule`` package; the latter is unavailable locally and
on SAI. This package produces the *same prepared schema* directly:

* :mod:`network_builder.species`  - species enumeration + rdkit 3D templates
* :mod:`network_builder.channels` - single-bond scission channel enumeration
* :mod:`network_builder.build`    - prepared yaml + ``MANIFEST.json`` writer
* :mod:`network_builder.cli`      - command line entry point

Verified consumer contract (see ``Recnet/network_inputs/README.md`` for the
file/line evidence): every reaction must be written in the *breaking*
direction (precursor -> two fragments) because ``handlers/ts.py`` drives the TS
search from ``rxn["reactant_species"][0]`` plus ``rxn["broken_bond"]`` (atom
indices *inside that template*), and ``handlers/energy.py`` skips any reaction
with fewer than two product species.
"""

from .species import SEED, Species, SpeciesSpec, build_species_table, template_text
from .channels import Channel, enumerate_channels
from .build import build_dataset, PAPER_STEPS

__all__ = [
    "SEED",
    "Species",
    "SpeciesSpec",
    "build_species_table",
    "template_text",
    "Channel",
    "enumerate_channels",
    "build_dataset",
    "PAPER_STEPS",
]
