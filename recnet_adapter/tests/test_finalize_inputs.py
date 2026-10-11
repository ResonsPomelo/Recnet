"""The historical C2 finalizer must receive project inputs explicitly."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/finalize_network.sh'


def environment(tmp_path):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('RECNET_') and k != 'DPEVA_PYTHON'}
    env.update(RECNET_CONFIG_PYTHON=sys.executable, RECNET_WORKSPACE_ROOT=str(tmp_path),
               PYTHONDONTWRITEBYTECODE='1')
    return env


def test_help_precedes_configuration_and_has_no_side_effects(tmp_path):
    env = environment(tmp_path)
    env['RECNET_PROJECT_CONFIG'] = str(tmp_path / 'missing.json')
    proc = subprocess.run(['bash', str(SCRIPT), '--help'], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert 'Usage:' in proc.stdout
    assert not list(tmp_path.iterdir())


def test_missing_anchors_does_not_use_implicit_project_file(tmp_path):
    old = tmp_path / 'validation-pipeline/reaction/paper_network_table.json'
    old.parent.mkdir(parents=True)
    old.write_text('{}')
    env = environment(tmp_path)
    env['RECNET_PYTHON'] = '/nonexistent/python'
    proc = subprocess.run(['bash', str(SCRIPT), '--root', str(tmp_path),
                           '--report', str(tmp_path / 'report')], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 2
    assert '--anchors is required' in proc.stderr
    assert not (tmp_path / 'report').exists()


@pytest.mark.parametrize('title', [None, "Custom $(touch UNEXPECTED) ' title"])
def test_explicit_inputs_reach_report_without_shell_evaluation(tmp_path, title):
    env = environment(tmp_path)
    probe = tmp_path / 'probe python'
    probe.write_text('#!' + sys.executable + '\n' + '''import json, pathlib, sys
args=sys.argv[1:]
with open('calls.jsonl', 'a') as f: f.write(json.dumps(args)+'\\n')
out=pathlib.Path(args[args.index('--out')+1])
if args[0].endswith('aggregate_network.py'):
    out.mkdir(exist_ok=True)
    for name in ('network_summary.json','network_summary.md'): (out/name).write_text('{}')
else:
    out.write_text('{}')
    if args[0].endswith('c2_gate.py'): out.with_suffix('.md').write_text('gate')
''')
    probe.chmod(0o755)
    env['RECNET_PYTHON'] = str(probe)
    anchors = tmp_path / 'anchors with spaces.json'
    anchors.write_text('{}')
    args = ['bash', str(SCRIPT), '--root', str(tmp_path), '--anchors', str(anchors),
            '--report', str(tmp_path / 'report')]
    if title is not None:
        args += ['--title', title]
    proc = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, proc.stderr
    calls = [json.loads(line) for line in (tmp_path / 'calls.jsonl').read_text().splitlines()]
    assert len(calls) == 5
    report = calls[-1]
    assert report[report.index('--anchors')+1] == str(anchors)
    actual_title = report[report.index('--title')+1]
    assert actual_title == (title if title is not None else 'Reaction network report')
    assert not (tmp_path / 'UNEXPECTED').exists()
    assert (tmp_path / 'report/SHA256SUMS').is_file()
