"""Shared fixtures.

The tests import :mod:`network_builder` from the repository tree and load
``Recnet/utils/geometry.py`` directly (package init pulls in deepmd-backed
helpers that the data-preparation environment does not have).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

RECNET_ROOT = Path(__file__).resolve().parents[2]
if str(RECNET_ROOT) not in sys.path:
    sys.path.insert(0, str(RECNET_ROOT))

from network_builder.build import build_dataset  # noqa: E402
from network_builder.channels import enumerate_channels  # noqa: E402
from network_builder.species import SEED, build_species_table  # noqa: E402

NETWORK_INPUTS = RECNET_ROOT / "network_inputs"
YAML_NAME = "fe5c2_510_c2.prepared_rmg_data.yaml"


@pytest.fixture(scope="session")
def species_list():
    return build_species_table(seed=SEED)


@pytest.fixture(scope="session")
def species_by_key(species_list):
    return {species.key: species for species in species_list}


@pytest.fixture(scope="session")
def channels(species_list):
    enumerated, dropped = enumerate_channels(species_list, bond_order_policy="any")
    return enumerated, dropped


@pytest.fixture(scope="session")
def manifest(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("prepared_dataset")
    return build_dataset(out_dir=out_dir, yaml_name=YAML_NAME, seed=SEED)


@pytest.fixture(scope="session")
def pipeline_geometry():
    """The real ``handlers/ts.py`` bond-detection helper."""
    path = RECNET_ROOT / "utils" / "geometry.py"
    spec = importlib.util.spec_from_file_location("_test_recnet_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
