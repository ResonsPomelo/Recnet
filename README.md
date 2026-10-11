# Recnet

Recnet builds and analyzes reaction networks and runs atomistic adsorption/TS
workflows through DeepMD. This engineering branch combines the scientific core,
explicit model inputs, network builder and compatible research tools. Distribution
name `recnet` and version `0.0.1` are provisional; this is not a published release.

## Install and verify

```bash
python -m pip install -e '.[builder,dev]'
make check PYTHON=python
python -m recnet_adapter --help
python -m network_builder.cli --help
```

Base runtime dependencies are ASE, NumPy, SciPy, PyYAML, Sella and DeepMD's PyTorch
backend. `builder` adds RDKit; `surface` adds optional pymatgen; `dev` adds pytest
and ruff. `core` and `pipeline` remain compatibility extras. Existing separate
environments work with `PIPELINE_PYTHON` (or `ADAPTER_PYTHON`) and `BUILDER_PYTHON`.
The selected interpreters need their relevant scientific dependencies installed.
Build/install checks also need pip, setuptools and wheel.

`make check` runs correctness lint, scientific-interface/tool tests, wheel payload
controls, explicit CLI help inventory, and source-external wheel/editable probes.
It verifies selected dependency origins, aliases, missing-model refusal and actual
builder generation through installed core geometry. `make package-smoke` aliases
the installation gate. See [CPU CI](ci/README.md) for the tested environment.
Local execution does not establish a hosted GitHub Actions result.

## Model and project input

```bash
python -m recnet_adapter run --model /path/to/model.pt --head science \
  --path /path/to/case --slab /path/to/slab --prepared /path/to/prepared.yaml
```

Select a model using CLI, `RECNET_DP_MODEL`, or an explicitly chosen project JSON
(`--project-config` / `RECNET_PROJECT_CONFIG`). `RECNET_MODEL_CONFIG` selects a
model-only JSON. Packaged configuration contains no checkpoint or head.

```json
{"root": "/path/to/project", "model": "models/model.pt", "head": "science"}
```

Relative project model paths resolve from `root`, or from the project file's
directory when root is omitted. Model-only JSON paths resolve from its directory.
Multi-head models require an explicit valid branch or alias, including `Default`;
`none` is not a multi-head selection. Single-head models use native DeepMD calls
without injecting a head. S/M labels, when explicitly configured for campaign
tools, are user aliases and never automatic model/head choices.

## Architecture and compatibility

- `handlers`, `ccqn`, `utils`: scientific workflow, depending on `recnet_backend`.
- `recnet_backend`: model metadata, validation and native DeepMD calculator.
- `recnet_adapter`: CLI/configuration and orchestration; old calculator imports
  share module identity with the backend.
- `network_builder`: network generation; output defaults to invocation-directory
  `network_inputs`, never the installation directory.
- `recnet_adapter/tools`: campaign, network, report, DFT, surface and wave tools.
  Original script/module names remain wrappers around the grouped implementations.
- `recnet_adapter/sai`: source-only deployment wrappers, examples and explicit
  project inputs. See its README before submitting anything.

Fe5C2 seed tables, expansion rules, C2 closure judgments and literature mappings
remain project-specific scientific inputs/compatibility tools. Their presence
is not a claim of arbitrary-chemistry support. See [architecture](docs/architecture.md).
Do not co-install this distribution with `recnet-ft2dp-adapter`: modules overlap.
Historical `prepare_rmg_data.py` and `graph_generator.py` remain source-only RMG
utilities with additional dependencies. The original untracked check_cases_tsv
script is not a supported CLI; campaign execution uses its validated input parser.

## TS inputs and behavior

Matching `<case>/rxn/seeds/` files are used by default; `--no-ts-seed` restores
ordinary template guesses. Vacancy-group filenames take precedence. Invalid
atom count, element order or unreadable seeds log a reason and fall back. Valid
seeds supply complete geometry, skip orientation enumeration and use the actual
optimizer cell/PBC for the reactive-bond restraint. `seed override: <path>` is
logged. Reproduce old seed-free runs with `--no-ts-seed` or a fixed seed-free case.
A seed does not force a site to be visited by the search order.

The primary-mode bond-projection threshold defaults to 0.16. Explicit
`RECNET_TS_MIN_BOND_PROJ` must be finite and in `(0, 1]`; invalid input fails.
Every new attempt records the selected value with round-trip precision. Equal
projection passes this gate, subject to the other independent checks. Project
choices such as 0.30 are not universal defaults. Dissociated adsorption sites
are retained as ordered TS backups; existing all-failed fallback semantics remain.

Voronoi viewing requires both `show=True` and `SLABSITE_SHOW_VORONOI=1`. Tests use
viewer substitutes. Scientific verification uses synthetic CPU models and fake
optimizers where indicated; it does not establish production GPU convergence or
change archived scientific conclusions. The process-global install operation
supports sequential switching, not concurrent model-context isolation.
