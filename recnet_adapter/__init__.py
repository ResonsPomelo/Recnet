"""Recnet model inputs: explicit arguments > environment > selected configuration.

Model selection belongs to the caller. Multi-head checkpoints require an explicit
valid head; no project checkpoint or Default branch is selected implicitly.
"""
from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from typing import Any, Optional

from recnet_backend import AdapterError, Resolution
from recnet_backend import _clean_head as _clean_head
from recnet_backend import _load_config as _backend_load_config
from recnet_backend import _resolve_model_path as _backend_resolve_model_path
from recnet_backend import resolve as _backend_resolve

__version__ = "0.0.1"

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent
CONFIG_PATH = ADAPTER_DIR / "model_config.json"

ENV_MODEL = "RECNET_DP_MODEL"
ENV_HEAD = "RECNET_DP_HEAD"

#: 管线自有的模块前缀（inject/rebind 只作用于这些命名空间）
_PIPELINE_PREFIXES = ("handlers", "utils", "ccqn", "recnet_adapter", "run_dp_ts")

#: 原生 DP 类，仅用于显式所选模型的对照检查。
_ORIGINAL_DP: Optional[type] = None

__all__ = [
    "AdapterError",
    "Resolution",
    "resolve",
    "install",
    "bind_model",
    "check_backend",
    "describe",
    "run_pipeline",
    "REPO_ROOT",
    "ADAPTER_DIR",
    "ENV_MODEL",
    "ENV_HEAD",
]


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------

def _load_config(config_path: Any = None) -> dict:
    return _backend_load_config(config_path, default_config_path=CONFIG_PATH)


def _resolve_model_path(value: Any, repo_root: Any = None) -> Optional[Path]:
    return _backend_resolve_model_path(value, repo_root if repo_root is not None else REPO_ROOT)


def resolve(model: Any = None, head: Any = None, *, config_path: Any = None,
            repo_root: Any = None, project_config: Any = None) -> Resolution:
    """Resolve model inputs with adapter defaults that callers may override."""
    return _backend_resolve(
        model=model, head=head, config_path=config_path, repo_root=repo_root,
        project_config=project_config, default_config_path=CONFIG_PATH,
        default_repo_root=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# 注入实现
# ---------------------------------------------------------------------------

def bind_model(model: Path) -> list:
    """把管线各模块中的 ``MODEL`` 常量重绑到给定检查点（会先导入 handlers 包）。"""
    _ensure_repo_on_path()
    try:
        importlib.import_module("handlers.context")
    except Exception as exc:  # noqa: BLE001 - 需要把缺依赖信息透传给用户
        raise AdapterError(
            "无法导入 Recnet 管线包 handlers（请确认依赖 ase/deepmd/sella/ccqn 已安装，"
            f"且从 Recnet 仓库根运行）：{exc.__class__.__name__}: {exc}"
        ) from exc

    value = str(model)
    bound = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith(_PIPELINE_PREFIXES):
            continue
        if hasattr(module, "MODEL"):
            setattr(module, "MODEL", value)
            bound.append(f"{name}.MODEL")
    return sorted(bound)


def _ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def install(
    model: Any = None,
    head: Any = None,
    *,
    require_model: bool = True,
    verbose: bool = True,
    project_config: Any = None,
) -> dict:
    """Configure Recnet's checked calculator; existing harmonic subclasses follow it.

    ``require_model`` remains accepted for API compatibility, but no call may
    fall back to a built-in scientific model.
    """
    global _ORIGINAL_DP
    from .model_selection import validate_model_head

    res = resolve(model=model, head=head, project_config=project_config)
    res.model, res.head = validate_model_head(res.model, res.head)
    from .calculator import NativeDP, configure

    _ORIGINAL_DP = NativeDP
    configure(res)
    report = {"resolution": res, "patched": [f"Recnet DP: model={res.model}, head={res.head!r}"],
              "rebound": bind_model(res.model), "warnings": []}

    if verbose:
        for line in report["patched"]:
            print(f"[recnet_adapter] {line}")
        if report["rebound"]:
            print(f"[recnet_adapter] 重绑定: {', '.join(report['rebound'])}")
        for warning in report["warnings"]:
            print(f"[recnet_adapter] WARNING: {warning}", file=sys.stderr)
    return report


# ---------------------------------------------------------------------------
# 运行时入口
# ---------------------------------------------------------------------------

def run_pipeline(extra_args: list, model: Any = None, head: Any = None, *, project_config=None) -> None:
    """Run the common CLI, with model selection validated after argument parsing."""
    _ensure_repo_on_path()
    script = REPO_ROOT / "run_dp_ts.py"
    argv = list(extra_args or [])
    if argv and argv[0] == "--":
        argv.pop(0)
    for key, value in (("--model", model), ("--head", head), ("--project-config", project_config)):
        if value is not None:
            argv.extend([key, str(value)])
    previous = sys.argv
    try:
        sys.argv = [str(script), *argv]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous


# ---------------------------------------------------------------------------
# 环境与校验
# ---------------------------------------------------------------------------

def _env_info() -> dict:
    info: dict = {}
    try:
        import deepmd  # noqa: F401

        info["deepmd-kit"] = getattr(deepmd, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        info["deepmd-kit"] = f"MISSING ({exc.__class__.__name__})"
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"MISSING ({exc.__class__.__name__})"
        info["cuda"] = False
    return info


def describe(stream=None, *, model=None, head=None, project_config=None) -> None:
    """打印当前解析结果（不导入管线、不加载模型）。"""
    out = stream or sys.stdout
    res = resolve(model=model, head=head, project_config=project_config)
    env = _env_info()

    print("== recnet_adapter show ==", file=out)
    print(f"adapter    : {ADAPTER_DIR}", file=out)
    print(f"env        : deepmd-kit {env['deepmd-kit']} | torch {env['torch']} | cuda {env['cuda']}", file=out)
    print(f"config     : {CONFIG_PATH}", file=out)

    if res.model is None:
        print(f"model      : <unconfigured>   (source: {res.source['model']})", file=out)
    else:
        exists = "OK" if res.model.exists() else "MISSING"
        size = f"{res.model.stat().st_size / 1e6:.1f} MB" if res.model.exists() else "-"
        print(f"model      : {res.model}  [{exists}, {size}]   (source: {res.source['model']})", file=out)
    print(f"head       : {res.head!r}   (source: {res.source['head']})", file=out)



def _test_structures() -> dict:
    """构造小的 C/H/O/Fe 测试体系（含周期性表面）。"""
    from ase import Atoms
    from ase.build import bcc110, bulk

    structures = {}

    co = Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.13)])
    co.set_cell([20.0, 20.0, 20.0])
    co.set_pbc([False, False, False])
    co.center()
    structures["CO(gas)"] = co

    h2o = Atoms("OH2", positions=[(0.0, 0.0, 0.0), (0.9572, 0.0, 0.0), (-0.24, 0.927, 0.0)])
    h2o.set_cell([20.0, 20.0, 20.0])
    h2o.set_pbc([False, False, False])
    h2o.center()
    structures["H2O(gas)"] = h2o

    fe_bulk = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    structures["Fe-bcc(2x2x2)"] = fe_bulk

    slab = bcc110("Fe", size=(2, 2, 4), vacuum=8.0)
    structures["Fe(110)-slab"] = slab
    return structures


def _check_sella_runtime() -> tuple[bool, str]:
    """Sella 初始化检查；用于提前暴露 user-site/JAX/NumPy 版本冲突。"""
    try:
        import importlib.metadata as metadata

        import sella
        from ase.build import molecule

        atoms = molecule("H2O")
        atoms.center(vacuum=5.0)
        opt = sella.Sella(atoms, order=1, delta0=0.1)
        version = getattr(sella, "__version__", None)
        if version is None:
            try:
                version = metadata.version("Sella")
            except Exception:  # noqa: BLE001
                version = "?"
        detail = f"Sella {version} | {sella.__file__}"
        del opt
        return True, detail
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc.__class__.__name__}: {exc}"


def check_backend(model: Any = None, head: Any = None, *, repeats: int = 3, project_config=None) -> bool:
    """端到端校验映射：管线调用式 vs 显式 head 参考、包装器、时延。

    返回 True 表示全部通过。
    """
    import numpy as np


    print("== recnet_adapter check ==")
    env = _env_info()
    print(f"env        : deepmd-kit {env['deepmd-kit']} | torch {env['torch']} | cuda {env['cuda']}")
    report = install(model=model, head=head, project_config=project_config)
    res = report["resolution"]

    from .calculator import DP as patched_dp
    ctx = importlib.import_module("handlers.context")
    constraints = importlib.import_module("utils.constraints")
    model_used = ctx.MODEL
    print(f"model      : {model_used}")
    print(f"head       : {res.head!r}  (patched: {patched_dp is not _ORIGINAL_DP})")
    print("")

    def evaluate(atoms, calc):
        frame = atoms.copy()
        frame.calc = calc
        return float(frame.get_potential_energy())

    def tolerance(e_ref: float) -> float:
        """float32 推理在大总量能量上有 ~1e-9 相对舍入；给 1e-5 eV 下限即可。"""
        return max(1e-5, 1e-9 * abs(e_ref))

    structures = _test_structures()
    rows = []
    all_ok = True

    header = f"{'structure':<16}{'pipeline-E':>15}{'ref(head)-E':>15}{'|diff|/meV':>12}"
    print(header)
    print("-" * len(header))

    for label, atoms in structures.items():
        try:
            e_pipe = evaluate(atoms, patched_dp(model=model_used))
            if res.head:
                e_ref = evaluate(atoms, _ORIGINAL_DP(model=model_used, head=res.head))
            else:
                e_ref = evaluate(atoms, _ORIGINAL_DP(model=model_used))
        except Exception as exc:  # noqa: BLE001 - 报清楚而不是抛栈
            all_ok = False
            print(f"{label:<16}  ERROR: {exc.__class__.__name__}: {exc}")
            continue

        diff = abs(e_pipe - e_ref)
        ok = diff <= tolerance(e_ref)
        all_ok &= ok
        rows.append((label, e_pipe, e_ref, diff, ok))
        flag = "OK" if ok else "FAIL"
        print(f"{label:<16}{e_pipe:>15.4f}{e_ref:>15.4f}{diff * 1e3:>12.3f}  {flag}")

    print("")
    # --- Sella 运行时（Recnet TS/CCQN 必经路径；backend check 不能只测 E/F）---
    sella_ok, sella_detail = _check_sella_runtime()
    print(f"Sella init          : {sella_detail}  {'OK' if sella_ok else 'FAIL'}")
    all_ok &= sella_ok

    # --- Recnet 自带包装器 HarmonicallyForcedDP（表面积点位点约束路径）---
    slab = structures["Fe(110)-slab"]
    site_pos = slab.positions[0].copy()
    wrapper_ok = False
    try:
        wrapper_calc = constraints.HarmonicallyForcedDP(
            model=model_used,
            atom_bond_potentials=[],
            center_bond_potentials=[],
            site_bond_potentials=[{"site_pos": site_pos, "ind": 5, "k": 0.5, "deq": 0.0}],
        )
        e_wrap = evaluate(slab, wrapper_calc)
        ref_kwargs = {"head": res.head} if res.head else {}
        ref_calc = constraints.HarmonicallyForcedDP(
            model=model_used,
            atom_bond_potentials=[],
            center_bond_potentials=[],
            site_bond_potentials=[{"site_pos": site_pos, "ind": 5, "k": 0.5, "deq": 0.0}],
            **ref_kwargs,
        )
        e_wrap_ref = evaluate(slab, ref_calc)
        wrap_diff = abs(e_wrap - e_wrap_ref)
        wrapper_ok = wrap_diff <= tolerance(e_wrap_ref)
        print(f"HarmonicallyForcedDP : E={e_wrap:.4f}  diff vs explicit head={wrap_diff * 1e3:.3f} meV  "
              f"{'OK' if wrapper_ok else 'FAIL'}")
    except Exception as exc:  # noqa: BLE001 - 报清楚而不是抛栈
        print(f"HarmonicallyForcedDP : ERROR: {exc.__class__.__name__}: {exc}")
    all_ok &= wrapper_ok

    # --- 时延（扰动坐标以绕开 ASE 结果缓存）---
    try:
        rng = np.random.default_rng(0)
        frame = slab.copy()
        frame.calc = patched_dp(model=model_used)
        frame.get_potential_energy()  # warmup（含首次加载）
        timings = []
        for _ in range(max(1, repeats)):
            pos = frame.get_positions()
            pos[:, 0] += rng.normal(0.0, 0.005, len(pos))
            frame.set_positions(pos)
            t0 = __import__("time").perf_counter()
            frame.get_potential_energy(), frame.get_forces()
            timings.append(__import__("time").perf_counter() - t0)
        print(f"timing: {len(slab)}-atom Fe(110) slab, median {np.median(timings) * 1e3:.1f} ms / E+F eval")
    except Exception as exc:  # noqa: BLE001 - 报清楚而不是抛栈
        all_ok = False
        print(f"timing: ERROR: {exc.__class__.__name__}: {exc}")

    print("")
    print("result:", "PASS" if all_ok else "FAIL")
    return all_ok
