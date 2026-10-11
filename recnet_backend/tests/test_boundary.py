"""The scientific core can select a model without importing the CLI adapter."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_core_and_calculator_do_not_import_adapter(tmp_path):
    code = r'''
import importlib
import json
import os
from pathlib import Path
import sys

class RejectAdapter:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'recnet_adapter' or fullname.startswith('recnet_adapter.'):
            raise AssertionError('core imported recnet_adapter: ' + fullname)

sys.meta_path.insert(0, RejectAdapter())
for name in ('handlers.adsorption', 'handlers.ts', 'handlers.irc',
             'handlers.energy', 'utils.constraints'):
    importlib.import_module(name)
from recnet_backend import AdapterError
from recnet_backend import calculator, model_selection
from utils.constraints import HarmonicallyForcedDP
model = Path(sys.argv[1]) / 'multi.pt'
model.touch()
single = model.with_name('single.pt')
single.touch()
model_selection.inspect_model = lambda path: (
    {'model_dict': {'Default': {}, 'science': {}}} if path == model else {'type_map': ['H']}
)
calls = []
calculator.NativeDP.__init__ = lambda self, model, *args, **kwargs: calls.append((model, kwargs))
for value in (None, 'none'):
    try:
        calculator.DP(model=model, **({} if value is None else {'head': value}))
    except AdapterError as exc:
        assert 'head' in str(exc)
    else:
        raise AssertionError('multi-head model accepted without explicit head')
calculator.DP(model=model, head='Default')
assert calls[-1] == (str(model), {'head': 'Default'})
HarmonicallyForcedDP(model=model, head='science')
assert calls[-1] == (str(model), {'head': 'science'})
os.environ['RECNET_DP_HEAD'] = 'science'
calculator.DP(model=model)
assert calls[-1] == (str(model), {'head': 'science'})
os.environ['RECNET_DP_MODEL'] = str(model)
HarmonicallyForcedDP()
assert calls[-1] == (str(model), {'head': 'science'})
del os.environ['RECNET_DP_MODEL']
del os.environ['RECNET_DP_HEAD']
config = model.with_name('selection.json')
config.write_text(json.dumps({'model': 'multi.pt', 'head': 'Default'}))
os.environ['RECNET_MODEL_CONFIG'] = str(config)
HarmonicallyForcedDP()
assert calls[-1] == (str(model), {'head': 'Default'})
del os.environ['RECNET_MODEL_CONFIG']
calculator.DP(model=single, head='science')
assert calls[-1] == (str(single), {})
assert 'recnet_adapter' not in sys.modules
'''
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "DP_INTRA_OP_PARALLELISM_THREADS": "1", "DP_INTER_OP_PARALLELISM_THREADS": "1"}
    for name in ("RECNET_DP_MODEL", "RECNET_DP_HEAD", "RECNET_MODEL_CONFIG", "RECNET_PROJECT_CONFIG"):
        env.pop(name, None)
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_and_backend_modules_are_identical():
    import recnet_adapter as adapter
    import recnet_adapter.calculator as old_calc
    import recnet_adapter.model_selection as old_selection
    from recnet_backend import AdapterError, Resolution, calculator, model_selection

    assert old_calc is calculator
    assert old_selection is model_selection
    assert old_calc.DP is calculator.DP
    assert adapter.AdapterError is AdapterError
    assert adapter.Resolution is Resolution


def test_neutral_calculator_with_no_adapter_directory(tmp_path):
    shutil.copytree(ROOT / "recnet_backend", tmp_path / "recnet_backend",
                    ignore=shutil.ignore_patterns("__pycache__", "tests"))
    code = r'''
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
class RejectAdapter:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'recnet_adapter' or fullname.startswith('recnet_adapter.'):
            raise AssertionError('neutral calculator imported adapter')
sys.meta_path.insert(0, RejectAdapter())
from recnet_backend import calculator, model_selection
assert Path(calculator.__file__).is_relative_to(Path(sys.argv[1]))
model = Path(sys.argv[1]) / 'model.pt'
model.touch()
model_selection.inspect_model = lambda path: {'model_dict': {'Default': {}}}
calls = []
calculator.NativeDP.__init__ = lambda self, model, **kwargs: calls.append((model, kwargs))
calculator.DP(model=model, head='Default')
assert calls == [(str(model), {'head': 'Default'})]
'''
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
