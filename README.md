# Recnet model-input and builder candidate

This candidate packages the upstream reaction-network core with a neutral model
boundary (`recnet_backend`) and a small CLI adapter (`recnet_adapter`). Its
scientific workflow remains the core from the PR-1 import fix. It also includes
the validated `network_builder` tree. It contains no campaign tools, SAI workflow, private checkpoints, or run
evidence. The distribution name `recnet` and version `0.0.1` are provisional
review metadata and are not a published release.

## Install and check

Use an environment with the scientific dependencies selected for your work:

```bash
python -m pip install -e '.[builder,dev]'
make check PYTHON=python PIPELINE_PYTHON=python
python -m run_dp_ts --help
python -m recnet_adapter --help
python -m network_builder.cli --help
```

Base installation declares ASE, NumPy, SciPy, PyYAML, Sella, and DeepMD's
PyTorch backend because the workflow imports them. `core` and `pipeline` are
empty compatibility extras. The `dev` extra provides pytest
and ruff. Build/install probes additionally need pip, setuptools, and wheel.
The `builder` extra adds RDKit; ASE, NumPy and PyYAML are base dependencies.

`make check` runs correctness lint, model-boundary, builder and package-boundary tests, and actual wheel and
editable installation probes. The model tests use tiny synthetic CPU DeepMD
checkpoints to compare energy and forces against the native calculator; they
do not run a reaction search. The install probes run outside the source tree,
verify package origins and compatibility import aliases, and check CLI help
and missing-model rejection without creating outputs. They use isolated
venvs with third-party packages from the selected interpreter and clear model
selection variables only in probe subprocesses.

## Model and head inputs

The workflow requires a model path from `--model`, `RECNET_DP_MODEL`, or an
explicitly selected JSON file. Select a project JSON with `--project-config`
or `RECNET_PROJECT_CONFIG`; `RECNET_MODEL_CONFIG` selects a model-only JSON.
An example project file is:

```json
{
  "root": "/path/to/workspace",
  "model": "models/example.pt",
  "head": "science"
}
```

Relative model paths in a project file resolve from its `root`; if `root` is
omitted, the project file's directory is used. A model-only JSON resolves
relative paths from its own directory. The packaged `model_config.json` has
no model or head selection. The repository contains no default project
checkpoint, head, or FT2DP/S/M assignment.

```bash
python -m run_dp_ts --model /path/to/model.pt --head science \
  --path /path/to/case --slab /path/to/slab --prepared /path/to/prepared.yaml
python -m recnet_adapter run --model /path/to/model.pt --head science \
  --path /path/to/case --slab /path/to/slab --prepared /path/to/prepared.yaml
```

For a multi-head model, `--head` must name a branch or alias. Even the
`Default` branch must be selected explicitly. `none` clears the selection and
is rejected for a multi-head model. A single-head model follows DeepMD's
native invocation without injecting a head. The same contract applies to
`HarmonicallyForcedDP` and to direct calls through `recnet_backend.calculator.DP`.
The legacy `recnet_adapter/run_dp_ts_ft2dp.py` path remains an entrypoint name;
it does not select an FT2DP model or head.

## Scope

Only the five core DP imports, `handlers/context.py` model constant, and
`run_dp_ts.py` entrypoint change in the scientific core. TS seed, gate,
surface, and campaign behavior from the research fork are outside this
candidate. Legacy `prepare_rmg_data.py` and `graph_generator.py` require
additional RMG dependencies and remain source-only utilities. No new console
script is introduced.

Do not co-install this distribution with the monolithic
`recnet-ft2dp-adapter` fork distribution: they contain overlapping modules.
Installed imports and CPU model parity do not establish GPU inference,
transition-state convergence, or scientific-result validity.

## Builder layer

The `network_builder` source matches the validated local fork snapshot. It
generates the Fe₅C₂(510) C2-and-below prepared dataset; its species seed and
expansion tables are project-specific. Default output is relative to the
invocation directory:

```bash
python -m network_builder.cli --out-dir ./network_inputs
```

The installed builder probe runs the generator outside the source tree and
checks its use of the installed core geometry. Two builder tests skip when
historical `network_inputs` audit snapshots are absent. With separate
scientific and RDKit environments, pass `BUILDER_PYTHON=/path/to/recnet-prep/python`
to `make check`.
