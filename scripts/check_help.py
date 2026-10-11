"""Bounded help checks for supported entry points (not an all-tools claim)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
# Explicitly reviewed CLI inventory. Do not auto-execute newly discovered scripts.
# slab_utils is a library; legacy demos and untracked scripts are not supported CLI.
TOOLS = {
    "adapter": (
        "campaign.build_campaign", "campaign.campaign_progress", "campaign.farm_runner",
        "campaign.gpu_concurrency_probe", "campaign.subset_prepared",
        "dft.build_abacus_cases_from_campaign", "dft.collect_ts_qc", "dft.eval_geometry_both",
        "network.aggregate_network", "network.analyze_network_table",
        "network.audit_channel_completeness", "network.audit_missing_attempts",
        "network.c2_gate", "network.missing_channels_to_indices",
        "report.channel_thermo", "report.co_thermo", "report.export_adsorbate_library",
        "report.export_module_table", "report.export_network_graph", "report.export_network_table",
        "report.export_profile_barriers", "report.export_thermo_table", "report.make_network_report",
        "report.summarize_campaign_xval", "surface.build_slab", "surface.make_seeds_scan",
        "surface.preflight_case",
    ),
    "builder": (
        "network.closure_audit", "network.validate_closure_table",
        "waves.audit_dropped_cuts", "waves.audit_wave_snapshot", "waves.make_expansion_c3",
        "waves.make_expansion_frontier", "waves.prepare_c3wave1", "waves.prepare_wave3",
        "waves.prepare_wave5", "waves.prepare_wave6", "waves.profile_c3_tiers",
        "waves.rehearse_scission_closure", "waves.triage_dropped_fragments",
    ),
}
COMMANDS = {
    "adapter": [
        ["-m", "recnet_adapter", "--help"],
        ["-m", "recnet_adapter", "show", "--help"],
        ["-m", "recnet_adapter", "check", "--help"],
        ["-m", "recnet_adapter", "run", "--help"],
        [str(ROOT / "run_dp_ts.py"), "--help"],
        [str(ROOT / "recnet_adapter/run_dp_ts_ft2dp.py"), "--help"],
        ["-m", "recnet_adapter.runtime_config", "--help"],
    ],
    "builder": [
        ["-m", "network_builder.cli", "--help"],
        [str(ROOT / "network_builder/cli.py"), "--help"],
    ],
}
for suite, modules in TOOLS.items():
    for module in modules:
        name = module.rsplit(".", 1)[1]
        COMMANDS[suite].extend([
            [str(ROOT / f"recnet_adapter/tools/{name}.py"), "--help"],
            ["-m", f"recnet_adapter.tools.{name}", "--help"],
            ["-m", f"recnet_adapter.tools.{module}", "--help"],
        ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=COMMANDS)
    parser.add_argument("--root", type=Path, default=ROOT, help="Source root or installed module directory")
    args = parser.parse_args()
    root = args.root.resolve()
    for template in COMMANDS[args.suite]:
        command = [arg.replace(str(ROOT), str(root)) for arg in template]
        with tempfile.TemporaryDirectory(prefix="recnet-help-") as directory:
            env = dict(os.environ)
            env.pop("DISPLAY", None)
            for key in list(env):
                if key.startswith("RECNET_"):
                    env.pop(key)
            env.update(PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1",
                       MPLBACKEND="Agg", HOME=directory, XDG_CACHE_HOME=directory)
            proc = subprocess.run([sys.executable, *command], cwd=directory,
                                  env=env, capture_output=True, text=True, timeout=30)
            output = proc.stdout + proc.stderr
            artifacts = list(Path(directory).rglob("*"))
            if proc.returncode or "usage:" not in output.lower() or artifacts:
                print(f"FAIL {command}: exit={proc.returncode}, artifacts={artifacts}")
                print(output)
                return 1
            print(f"PASS {' '.join(command)}")
    print(f"PASS {args.suite}: {len(COMMANDS[args.suite])} help invocations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
