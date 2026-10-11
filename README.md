# Recnet core engineering candidate

This branch packages the existing upstream scientific core after the handlers
import fix. It contains `handlers`, `ccqn`, `utils`, `run_dp_ts` and `slabsite`.
It does not contain the fork's adapter, neutral backend, network builder,
campaign tools, private checkpoints, SAI scripts or run evidence.

The distribution name `recnet` and version `0.0.1` are provisional review
metadata, not a published release. This candidate is based on local upstream
commit `97b9772` plus import fix `f7b102a`; it does not claim to track current
remote upstream.

## Install and inspect

Use a separate environment with the appropriate scientific dependencies:

```bash
python -m pip install -e '.[dev]'
python -m run_dp_ts --help
make check PYTHON=python
```

The base installation declares ASE, NumPy, SciPy, PyYAML, Sella and DeepMD's
PyTorch support because importing the core workflow loads those modules.
The `core` and `pipeline` extras remain accepted as empty compatibility aliases.
The `dev` extra supplies local quality tools.
Build/install probes additionally need pip, setuptools and wheel.

`make check` runs correctness lint, negative tests for the wheel allowlist,
and wheel/editable installation probes.
The probes reuse the selected interpreter's dependency directories in private
venvs, verify actual dependency and package import locations, and check the
installed CLI help from outside the source directory. They do not execute
parent editable hooks or install into the original environment. To use another
pipeline interpreter, set `PIPELINE_PYTHON=/absolute/path/to/python`; the first
`PYTHON` must also have pipeline dependencies for its wheel/editable probes.

## Scope and compatibility

This is a packaging candidate: scientific source files are unchanged relative
to PR-1. The inherited core still has its historical hard-coded model setting
in `handlers/context.py`. This branch does not implement the fork's explicit
model/head contract and must not replace its production deployment on its own.
The separate backend/adapter change must carry that contract when integrating
the fork workflow.

Legacy `prepare_rmg_data.py` and `graph_generator.py` require additional RMG
preparation dependencies and remain source-only utilities. They are not part
of the installed core surface. No new console-script name is introduced.

Installed imports and help are engineering smoke tests, not GPU inference,
transition-state convergence or scientific-result validation. Do not co-install
this distribution with the monolithic `recnet-ft2dp-adapter` fork distribution:
they contain overlapping Python modules. Test each candidate in isolation.
