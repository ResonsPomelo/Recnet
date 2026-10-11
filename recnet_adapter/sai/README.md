# SAI launchers for Recnet

These are local deployment templates. They do not select a project, checkpoint,
model head, Slurm account, partition, QOS, GPU count, wall time, or output path.
The former FT2DP/Fe5C2 campaign values are historical evidence, not defaults.
The `04`–`06` entrypoints are project-specific scientific workflows and require
review of their case manifests, gate, thresholds and reference data before use.
Local syntax and fake-command tests do not establish SAI or scientific acceptance.

## Project and runtime inputs

Set `RECNET_DIR` to the deployed checkout when submitting with `sbatch`.
Slurm may execute a spooled copy of the script, so the script-location fallback
is only reliable for direct local runs. Choose one of `RECNET_PROJECT_CONFIG`
(explicit JSON file supported by `runtime_config.py`) or
`RECNET_WORKSPACE_ROOT`. The project file can select the workspace root, Python,
model paths and shared head; environment values override it as documented by
`python recnet_adapter/runtime_config.py --help`. `RECNET_ENV_SCRIPT` may point
to a site environment setup script; omit it if the selected Python already has
the scientific dependencies. `RECNET_CONFIG_PYTHON` chooses the stdlib Python
used to read the project file; the resolved `RECNET_PYTHON` runs the science.
The legacy `RECNET_MODEL_CONFIG` can select an external model file. Both file
selectors are resolved before a wrapper changes its working directory.

Example project file (replace every path and model with verified deployment
inputs; relative model paths resolve under `root`):

```json
{
  "root": "/project/work",
  "python": "/project/env/bin/python",
  "models": {
    "S": {"path": "models/single.pt", "head": "none"},
    "M": {"path": "models/multi.pt", "head": "selected-head"}
  }
}
```

The single-model `00`, `01`, `02`, and `07` wrappers require a selected model:
`RECNET_DP_MODEL`, project-file `model`/`models.default`, or the positional
checkpoint for `02`. A single-head model uses its native no-head invocation;
a multi-head checkpoint needs a valid head from `RECNET_DP_HEAD` or the
selected project configuration. Freeze validates checkpoint metadata before
calling DeepMD. Other wrappers let the backend validate model/head pairing.
`RECNET_RUN_ROOT` selects
run output for `00`, `03`, `04`, `06_round2`, and `07`; legacy `RUN_ROOT` is
accepted. `01` writes to its explicit `RECNET_CASE`; `02` uses `OUT_DIR`.

The wrappers leave all Slurm resource directives to the submitter. Pass
`--account`, `--partition`, `--qos`, `--nodes`, `--ntasks`, GPU request,
`--time`, `--output` and `--error` explicitly based on current SAI policy and
the project's allocation. Choose an output directory that exists before
submission. Keep the submit shell clean; this site has previously rejected
`sbatch --export` for these jobs. Example shape only:

```bash
export RECNET_DIR=/project/Recnet
export RECNET_PROJECT_CONFIG=/project/config/recnet.json
export RECNET_RUN_ROOT=/project/runs
export RECNET_DP_MODEL=/project/models/chosen.pt
export RECNET_DP_HEAD=chosen-head
# Optional: export RECNET_ENV_SCRIPT=/project/env/setup.sh

sbatch --account="$PROJECT_ACCOUNT" --partition="$PROJECT_PARTITION" \
  --qos="$PROJECT_QOS" --nodes=1 --ntasks=1 --gpus-per-node=1 \
  --time="$PROJECT_TIME" --output="$PROJECT_LOG_DIR/backend-%j.out" \
  --error="$PROJECT_LOG_DIR/backend-%j.err" \
  "$RECNET_DIR/recnet_adapter/sai/00_backend_check.sbatch"
```

`--help` is safe for every wrapper and does not source the environment or
create output. The first positional argument to `02`, `03`, `04`, and
`06_co_d2s` is the script's own input; put all Slurm flags before the script.

## Entry points

| Script | Required project input | Behavior |
|---|---|---|
| `00_backend_check.sbatch` | model, run root; head for multi-head checkpoints | Runs adapter backend check in a new run directory. |
| `01_run_pipeline.sbatch` | `RECNET_CASE`, prepared YAML, model; head for multi-head checkpoints; optional `RECNET_SLAB` | Runs the retained mapped `run_dp_ts_ft2dp.py` entrypoint. Extra pipeline arguments follow the script path. |
| `02_freeze_export.sbatch` | `OUT_DIR`, checkpoint; head for multi-head checkpoints | Validates model metadata, calls the selected Python's `-m deepmd --pt freeze`, then checks finite CO energy and forces on the `.pt2`. This DPA4 export smoke is not a model reference validation. |
| `03_case_farm.sbatch` | `cases.tsv`, run root, `RECNET_CONCURRENCY`, `RECNET_THREADS` | Calls the retained `tools/farm_runner.py`; model/head may be in each manifest row or explicitly supplied as fallback. |
| `04_seed_scan.sbatch` | scan plan, S/M models, case root, species, reaction key, freeze threshold, concurrency, threads, timeout; arm heads for multi-head checkpoints | Historical per-site scan on selected cases. A failed scan makes the job fail while retaining its summary. |
| `05_finalize_c2.sbatch` | root(s), expected root(s), report, anchors, coverage threshold | Historical Fe5C2 C2 gate and paper comparison. |
| `06_co_d2s.sbatch` | D2S case TSV, reaction runner directory, output root, concurrency | Historical CO dissociation recovery using an external `d2s_run.py`. |
| `06_round2_if_needed.sbatch` | selected gate and cases TSV, run ID, run root, concurrency, threads | Skips on gate pass or already attempted/rejected audit; otherwise runs farm. Regenerate the gate afterward. |
| `07_gpu_concurrency_probe.sbatch` | model, two structures, workers, repeats, run root; head for multi-head checkpoints | Project-selected TS/MD frame throughput probe. |

For a prepared case, choose a slab and
`<case>/prepared_data/prepared_rmg_data.yaml`, then submit `01` with
`RECNET_CASE=/project/runs/case-a`. The prepared RMG conversion is a separate
step and may need RDKit/RMG packages. `RECNET_EXTRA_ARGS` is a legacy,
whitespace-split convenience; use arguments after the script for quoted
values. The retained mapped entrypoint and current `run_dp_ts.py` both install
the adapter.

Farm manifest is tab separated with six columns:

```text
case_name<TAB>case_dir<TAB>model<TAB>head<TAB>extra_args<TAB>time_budget_s
```

Model cells may be empty only if a default model is selected through
`RECNET_DP_MODEL` or a selected project file with `model` or
`models.default`. A blank head means native no-head invocation; the backend
rejects it for multi-head checkpoints. A global or per-model head can be
selected explicitly. `none` is the single-head spelling. `case_name` must be
unique.
Relative manifest paths are interpreted by `farm_runner.py`, so inspect its
`--help` and dry run with the same deployed checkout before submission:

```bash
"$RECNET_PYTHON" "$RECNET_DIR/recnet_adapter/tools/farm_runner.py" \
  --cases /project/inputs/cases.tsv --recnet-dir "$RECNET_DIR" \
  --concurrency 2 --threads 4 --dry-run
RECNET_CONCURRENCY=2 RECNET_THREADS=4 \
  sbatch --account="$PROJECT_ACCOUNT" --partition="$PROJECT_PARTITION" \
  --qos="$PROJECT_QOS" --nodes=1 --ntasks="$PROJECT_TASKS" \
  --gpus-per-node="$PROJECT_GPUS" --time="$PROJECT_TIME" \
  --output="$PROJECT_LOG_DIR/farm-%j.out" --error="$PROJECT_LOG_DIR/farm-%j.err" \
  "$RECNET_DIR/recnet_adapter/sai/03_case_farm.sbatch" \
  /project/inputs/cases.tsv campaign-a
```

The wrapper checks `CUDA_VISIBLE_DEVICES` before farm execution. A missing
allocation mask is an error; the wrapper never assumes host GPU numbers.
Inspect `farm_summary.tsv`, each case's `RUN_MANIFEST.txt`, and case logs for
runtime evidence. These classify execution failures, not scientific validity.

## Historical Fe5C2 workflows

`04_seed_scan.sbatch` consumes a plan of `arm<TAB>site<TAB>vg`, where arm is
`S` or `M`. Set `RECNET_SCAN_CASE_ROOT`, `RECNET_SCAN_SPECIES`, `RECNET_RXN`,
`RECNET_SCAN_CONCURRENCY`, `RECNET_SCAN_THREADS`,
`RECNET_SCAN_TIMEOUT_S`, and `RECNET_SCAN_FREEZE_THRESHOLD`. Select
`RECNET_MODEL_S` and `RECNET_MODEL_M` directly or via project config
`models.S/M`. Per-arm heads resolve through the same runtime configuration,
including `models.S/M.head` and `RECNET_MODEL_HEAD_S/M`; the legacy
`RECNET_SCAN_HEAD_S/M` overrides remain accepted. Without a head, single-head
models use their native path and multi-head models fail backend validation. The
threshold, species and reaction key are scientific project decisions. The
optional `RECNET_SCAN_DRY_RUN=1` delegates planning to the scan tool. Outputs
are under `RECNET_RUN_ROOT/<run_id>` and the selected case directories.

`05_finalize_c2.sbatch` forwards explicit repeated `--root`, repeated
`--expected-root`, `--report`, `--anchors`, and `--coverage-threshold` to
`tools/finalize_network.sh`. The retained finalizer has Fe5C2 channel
mapping; a different title does not generalize it. Attach any `afterany`
dependency through `sbatch --dependency` after choosing the predecessor job.

`06_round2_if_needed.sbatch` reads `--gate`, `--cases`, and `--run-id` (legacy
environment names `RECNET_ROUND2_GATE`, `RECNET_ROUND2_CASES`, `RECNET_RUN_ID`
remain accepted). Attach a chosen `afterok` dependency through `sbatch` if
appropriate. `06_co_d2s.sbatch` consumes rows of
`name<TAB>case<TAB>model<TAB>head<TAB>outdir` and needs
`RECNET_REACTION_DIR` containing `d2s_run.py`,
`RECNET_D2S_OUT_ROOT`, and `RECNET_D2S_CONCURRENCY`.

For every scientific wrapper, review model provenance, head selection, inputs,
Slurm allocation, and output location in the target project before submitting.
