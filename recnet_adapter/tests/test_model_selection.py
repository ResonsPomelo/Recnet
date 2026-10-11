"""Model selection is user input, including an explicit Default branch."""
import json
import pytest
import recnet_adapter as adapter


def test_packaged_config_has_no_project_defaults(monkeypatch):
    for key in ('RECNET_DP_MODEL', 'RECNET_DP_HEAD', 'RECNET_MODEL_CONFIG', 'RECNET_PROJECT_CONFIG'):
        monkeypatch.delenv(key, raising=False)
    result = adapter.resolve()
    assert result.model is None
    assert result.head is None


def test_no_model_never_uses_pipeline_builtin(monkeypatch):
    monkeypatch.setenv('RECNET_DP_MODEL', 'none')
    with pytest.raises(adapter.AdapterError, match='model|模型'):
        adapter.install(require_model=False, verbose=False)


def test_explicit_project_config_is_opt_in(monkeypatch, tmp_path):
    for key in ('RECNET_DP_MODEL', 'RECNET_DP_HEAD', 'RECNET_MODEL_CONFIG', 'RECNET_PROJECT_CONFIG'):
        monkeypatch.delenv(key, raising=False)
    cfg = tmp_path / 'project.json'
    cfg.write_text(json.dumps({'root':'data', 'model':'candidate.pt', 'head':'branch'}))
    assert adapter.resolve().model is None
    monkeypatch.setenv('RECNET_PROJECT_CONFIG', str(cfg))
    assert adapter.resolve().model == tmp_path / 'data/candidate.pt'
    assert adapter.resolve().head == 'branch'
    assert adapter.resolve(head='none').head is None


def test_explicit_bad_config_fails(monkeypatch, tmp_path):
    monkeypatch.setenv('RECNET_PROJECT_CONFIG', str(tmp_path / 'missing.json'))
    with pytest.raises(adapter.AdapterError):
        adapter.resolve()


@pytest.fixture
def backend(monkeypatch, tmp_path):
    from recnet_adapter import model_selection as ms
    import deepmd.calculator
    model = tmp_path / 'multi.pt'
    model.touch()
    metadata = {'model_dict': {'Default': {}, 'science': {'model_branch_alias': ['alias']}}}
    monkeypatch.setattr(ms, 'inspect_model', lambda path: metadata)
    calls = []
    monkeypatch.setattr(deepmd.calculator, 'DP', lambda **kwargs: calls.append(kwargs) or kwargs)
    return ms, model, metadata, calls


@pytest.mark.parametrize('head', [None, '', 'none', 'null', 'wrong', 0])
def test_multi_head_rejected_before_calculator(backend, head):
    ms, model, _, calls = backend
    with pytest.raises(adapter.AdapterError, match='head'):
        ms.create_calculator(model, head)
    assert calls == []


@pytest.mark.parametrize('head,expected', [('Default','Default'), ('science','science'), ('ALIAS','alias')])
def test_explicit_valid_head_and_alias_use_native_rules(backend, head, expected):
    ms, model, _, calls = backend
    ms.create_calculator(model, head)
    assert calls == [{'model':str(model), 'head':expected}]


def test_single_head_has_no_injected_head(backend):
    ms, model, metadata, calls = backend
    metadata.clear()
    metadata['type_map'] = ['H']
    ms.create_calculator(model, 'unrelated-project-head')
    assert calls == [{'model':str(model)}]


def test_metadata_uses_backend_show_protocol_and_cache(monkeypatch, tmp_path):
    import deepmd.infer
    from recnet_adapter import model_selection as ms
    model = tmp_path / 'inspect.pt'
    model.write_bytes(b'fake')
    calls = []
    class Evaluator:
        def __init__(self, path, **kwargs):
            calls.append((path, kwargs))
        def get_model_def_script(self):
            return {'type_map':['H']}
    monkeypatch.setattr(deepmd.infer, 'DeepEval', Evaluator)
    assert ms.validate_model_head(model) == (model, None)
    ms.validate_model_head(model)
    assert calls == [(str(model), {'head':0})]
    model.write_bytes(b'changed')
    ms.validate_model_head(model)
    assert len(calls) == 2


def test_install_and_existing_harmonic_follow_each_selection(monkeypatch, tmp_path):
    from recnet_adapter import calculator, model_selection as ms
    import deepmd.calculator
    from utils.constraints import HarmonicallyForcedDP
    original = deepmd.calculator.DP
    calls = []
    model = tmp_path / 'multi.pt'
    single = tmp_path / 'single.pt'
    model.touch(); single.touch()
    monkeypatch.setattr(ms, 'inspect_model', lambda path: {} if path == single else {'model_dict':{'Default':{},'science':{}}})
    def record(self, model, *args, **kwargs):
        calls.append((model, kwargs))
    monkeypatch.setattr(calculator.NativeDP, '__init__', record)
    monkeypatch.setattr(calculator, '_configured', None)
    # bind_model modifies pipeline globals; restore using monkeypatch teardown.
    import sys
    for module in list(sys.modules.values()):
        if module and getattr(module, '__name__', '').startswith(('handlers', 'utils')) and hasattr(module, 'MODEL'):
            monkeypatch.setattr(module, 'MODEL', module.MODEL)
    for selected in ('science', 'Default'):
        adapter.install(model, selected, verbose=False)
        HarmonicallyForcedDP(model=str(model))
        assert calls[-1] == (str(model), {'head': selected})
    adapter.install(single, 'none', verbose=False)
    HarmonicallyForcedDP(model=str(single))
    assert calls[-1] == (str(single), {})
    with pytest.raises(adapter.AdapterError, match='head'):
        calculator.DP(model=str(model))
    assert deepmd.calculator.DP is original


def test_cpu_native_checkpoint_and_pipeline_calculator(tmp_path):
    """Exercise installed DeepMD using tiny synthetic checkpoints, without CUDA."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    code = r'''
from pathlib import Path
import sys
import numpy as np
import torch
from deepmd.pt.model.model import get_model
from deepmd.pt.train.wrapper import ModelWrapper
from deepmd.calculator import DP as NativeDP
from ase import Atoms
from recnet_adapter import AdapterError, install
from recnet_adapter.model_selection import create_calculator
from utils.constraints import HarmonicallyForcedDP
params = {'type_map':['H'], 'descriptor':{'type':'se_e2_a','sel':[2], 'rcut':3.0,'rcut_smth':2.5,'neuron':[4,8],'axis_neuron':2}, 'fitting_net':{'neuron':[4]}}
root = Path(sys.argv[1])
for kind in ('single', 'multi'):
    path = root / (kind + '.pt')
    if kind == 'single':
        wrapper = ModelWrapper(get_model(params), model_params=params)
    else:
        wrapper = ModelWrapper({k:get_model(params) for k in ('Default','science')}, model_params={'model_dict':{k:params for k in ('Default','science')}})
    torch.save(wrapper.state_dict(), path)
    if kind == 'multi':
        for invalid in (None, 'none', 'typo'):
            try:
                create_calculator(path, invalid)
            except AdapterError:
                pass
            else:
                raise AssertionError('invalid multi-head input accepted')
    for head in ([None] if kind == 'single' else ['science', 'Default']):
        selection = {} if head is None else {'head': head}
        reference = Atoms('H2', positions=[[0,0,0],[0,0,0.8]], cell=[10,10,10], pbc=True)
        reference.calc = NativeDP(model=str(path), **selection)
        energy, forces = reference.get_potential_energy(), reference.get_forces()
        install(path, head if head is not None else 'none', verbose=False)
        for calc in (create_calculator(path, head), HarmonicallyForcedDP(model=str(path))):
            atoms = reference.copy(); atoms.calc = calc
            np.testing.assert_allclose(atoms.get_potential_energy(), energy, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(atoms.get_forces(), forces, rtol=1e-10, atol=1e-10)
        print(kind, head, 'native energy/forces parity PASS')
# Existing harmonic class must reset after returning from multi to single.
install(root / 'single.pt', 'none', verbose=False)
HarmonicallyForcedDP(model=str(root / 'single.pt'))
'''
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1', 'DP_INTRA_OP_PARALLELISM_THREADS':'1', 'DP_INTER_OP_PARALLELISM_THREADS':'1'}
    for key in ('RECNET_DP_MODEL', 'RECNET_DP_HEAD', 'RECNET_MODEL_CONFIG', 'RECNET_PROJECT_CONFIG'):
        env.pop(key, None)
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('parity PASS') == 3


def test_exact_branch_name_precedes_case_insensitive_alias(backend):
    ms, model, metadata, calls = backend
    metadata['model_dict'] = {'science':{}, 'Science':{}}
    ms.create_calculator(model, 'Science')
    assert calls[-1]['head'] == 'Science'


def test_same_model_symlink_retains_explicit_selection(monkeypatch, tmp_path):
    from recnet_adapter import calculator, model_selection as ms
    model = tmp_path / 'model.pt'
    model.touch()
    alias = tmp_path / 'selected.pt'
    alias.symlink_to(model)
    monkeypatch.setattr(ms, 'inspect_model', lambda path: {'model_dict':{'Default':{},'science':{}}})
    monkeypatch.setattr(calculator, '_configured', adapter.Resolution(model, 'science', {}))
    calls = []
    monkeypatch.setattr(calculator.NativeDP, '__init__', lambda self, model, **kwargs: calls.append((model, kwargs)))
    calculator.DP(model=str(alias))
    assert calls == [(str(model), {'head':'science'})]
