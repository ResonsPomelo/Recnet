"""Keep model-selection tests headless without project-specific fixtures."""
import os

import ase.visualize
import slabsite


def _no_view(*args, **kwargs):
    return None


ase.visualize.view = _no_view
slabsite.view = _no_view
os.environ["MPLBACKEND"] = "Agg"
os.environ.pop("DISPLAY", None)
