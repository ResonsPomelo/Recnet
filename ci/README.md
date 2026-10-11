# CPU quality gate

`.github/workflows/cpu-quality.yml` runs the same `make check` and
installation probe included in `make check` (also available as
`make package-smoke`) used locally. Its scope is Linux x86_64, CPython 3.12,
released DeepMD 3.2.0 and CPU PyTorch 2.11.0. It does not load private models,
submit scientific jobs or publish packages. A future authorized push/PR can
trigger it; adding the file locally is not a hosted CI run.

`constraints-cpu.txt` records the tested CI dependency set. This is separate
from the supported dependency ranges in `pyproject.toml`: existing research
conda environments can retain their tested development DeepMD version.

## Reproduce in a clean environment

```bash
python3.12 -m venv /path/to/recnet-ci
source /path/to/recnet-ci/bin/activate
python -m pip install --upgrade -c ci/constraints-cpu.txt pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu -c ci/constraints-cpu.txt torch
python -m pip install -c ci/constraints-cpu.txt -e '.[builder,pipeline,dev]'
python -m pip check
export CUDA_VISIBLE_DEVICES= MPLBACKEND=Agg PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 DP_INTRA_OP_PARALLELISM_THREADS=1 DP_INTER_OP_PARALLELISM_THREADS=1
make check PYTHON=python
```

The existing two-interpreter Makefile overrides remain available; CI installs
both extras in its disposable environment so RDKit compatibility tests also run.
The installation probe creates private venvs and adds the selected interpreter’s
dependency directories as plain paths, checking their import origins without
executing parent editable-install hooks. Wheel imports must resolve inside the
probe environment. Missing historical data snapshots may skip with a reason. CPU model tests build
tiny checkpoints locally, exercising native model/head semantics and energy/force
parity; this is not production model or GPU validation.

## CLI coverage

`scripts/check_help.py` explicitly lists 40 migrated tools. Each is checked as
an old script, an old module and a canonical module, plus nine package/root
entries (129 invocations). It clears user model settings and DISPLAY, uses
temporary cwd/HOME, checks exit status and usage text, and fails on created files.
The adapter/builder split follows existing dependency environments. The list is
reviewed rather than executing every discovered Python file.

`slab_utils` is a library. `slabsite.py` is a legacy demo, and the historical
`prepare_rmg_data.py` / `graph_generator.py` dependency contract remains outside
this gate. Additional campaign input checkers, when present, are tested with their consumers.
Direct execution of package `__main__.py` is not a supported invocation; use `-m`.
Thus the gate does not claim to fix every historical audit entry.

## Maintenance

Actions are pinned to upstream commit SHAs; permissions are `contents: read`,
checkout credentials are not persisted, and jobs have a 30-minute limit.
Update CI constraints deliberately in a fresh venv and rerun both Make targets.
Keep package metadata independent of this CI stack. Dependency/runner updates
must pass the gate; do not turn a failure into a skip to keep CI green.

Sources used for workflow setup:
[GitHub Python workflows](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
and [PyTorch installation selector](https://pytorch.org/get-started/locally/).
