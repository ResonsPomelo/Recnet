"""测试用最小合成 case + 管线替身（不加载 DP、不需要 GPU）。

合成 case 复刻真实 case 的目录口径：

* ``<case>/Fe5C2_510.vasp``                                       slab
* ``<case>/prepared_data/prepared_rmg_data.yaml``                 反应/物种表
* ``<case>/prepared_data/ads_templates/*.xyz``                    吸附模板
* ``<case>/rxn/Adsorbates/<species_key>/<site>[_vg<k>]_opt.xyz``  已弛豫吸附结构（组装结构）
* ``<case>/rxn/seeds/<rxn_key>_site_<site>[_vg<k>].xyz``          TS seed（本功能输入）

替身只替换 DP/优化器/CCQN/振动；几何逻辑、判据、归档全部走真实实现。
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write

#: Recnet 仓库根（测试文件的 ../../）
REPO_ROOT = Path(__file__).resolve().parents[2]

#: 合成 slab 的单元格（z 方向留足真空）
SLAB_CELL = np.diag([5.74, 5.74, 24.0])
#: 只放开最上层 Fe 与吸附质（合成 slab 的表层在 z=15）
BOTTOM_FREEZE_THRESHOLD = 14.0
#: 模板 CO 的 C-O 距离（决定默认路径的 1.1*d 弹簧与断键上限）
D_TEMPLATE = 1.25
#: 期望的位点索引（真实 pilot 例子里是 10；合成 slab 位点少时自动回退）
PREFERRED_SITE = 10


@dataclass
class CaseInfo:
    """合成 case 的路径与 key 信息。"""

    dir: Path
    site: int
    rxn_key: str
    species_key: str
    slab: Atoms
    ads_template: Atoms
    seed_dir: Path
    ads_dir: Path

    @property
    def n_slab(self) -> int:
        return len(self.slab)

    def assembly(self, d_co: float, *, order: tuple[str, str] = ("C", "O"),
                 z: float = 16.4) -> Atoms:
        """组装结构（slab + CO）：C 在 (x0,y0,z)，O 沿 +x 拉到 d_co。"""
        x0, y0 = 2.87, 2.87
        first, second = order
        ads = Atoms(
            symbols=f"{first}{second}",
            positions=[(x0, y0, z), (x0 + d_co, y0, z)],
            cell=SLAB_CELL,
            pbc=True,
        )
        return self.slab.copy() + ads

    def seeds_file(self, d_co: float, *, vg: int | None = 0,
                   order: tuple[str, str] = ("C", "O")) -> Path:
        """写 ``rxn/seeds/<rxn_key>_site_<site>[_vg<k>].xyz``，返回路径。"""
        name = f"{self.rxn_key}_site_{self.site}"
        if vg is not None:
            name += f"_vg{vg}"
        path = self.seed_dir / f"{name}.xyz"
        path.parent.mkdir(parents=True, exist_ok=True)
        write(path, self.assembly(d_co, order=order))
        return path

    def adsorbate_opt(self, d_co: float, *, vg: int | None = 0) -> Path:
        """写 ``rxn/Adsorbates/<species_key>/<site>[_vg<k>]_opt.xyz``（B 工具输入）。"""
        name = f"{self.site}"
        if vg is not None:
            name += f"_vg{vg}"
        self.ads_dir.mkdir(parents=True, exist_ok=True)
        path = self.ads_dir / f"{name}_opt.xyz"
        write(path, self.assembly(d_co))
        return path


def build_slab() -> Atoms:
    """4 层 Fe 极小板（z 法向；表层 Fe 会被 SlabSite 认作位点层）。"""
    return Atoms(
        "Fe4",
        positions=[(2.87, 2.87, 15.0), (2.87, 2.87, 12.0),
                   (2.87, 2.87, 9.0), (2.87, 2.87, 6.0)],
        cell=SLAB_CELL,
        pbc=True,
    )


def _template_atoms(symbols: str) -> Atoms:
    positions = [(0.0, 0.0, 0.0)]
    if len(symbols) == 2:
        positions.append((D_TEMPLATE, 0.0, 0.0))
    return Atoms(symbols, positions=positions, cell=[40.0, 40.0, 40.0], pbc=False)


#: 物种表：CO（reactant）+ C/O（products）
SPECIES = {
    "sp_000": {"name": "CO", "symbols": "CO", "ad_idx": [0]},
    "sp_001": {"name": "C", "symbols": "C", "ad_idx": [0]},
    "sp_002": {"name": "O", "symbols": "O", "ad_idx": [0]},
}
#: 反应表：CO* -> C* + O*（C-O 断键）
RXN = {
    "reactant": "CO*",
    "product": "C* O*",
    "broken_bond": [0, 1],
    "reactant_species": ["sp_000"],
    "product_species": ["sp_001", "sp_002"],
}


def create_case(case_dir: Path, *, site: int = 10) -> CaseInfo:
    """写 slab / prepared_data / 模板，返回 CaseInfo（含 rxn_key、species_key）。"""
    from handlers.context import compute_reaction_key, compute_species_key

    case_dir = Path(case_dir)
    prepared_dir = case_dir / "prepared_data"
    (prepared_dir / "ads_templates").mkdir(parents=True, exist_ok=True)

    slab = build_slab()
    write(case_dir / "Fe5C2_510.vasp", slab)

    templates = {}
    species_payload = {}
    for sp_id, rec in SPECIES.items():
        atoms = _template_atoms(rec["symbols"])
        rel = f"ads_templates/{sp_id}.xyz"
        write(prepared_dir / rel, atoms)
        templates[sp_id] = {"atoms": atoms, "ad_idx": rec["ad_idx"], "name": rec["name"]}
        species_payload[sp_id] = {
            "name": rec["name"],
            "adjlist": "",
            "template_xyz": rel,
            "ad_idx": list(rec["ad_idx"]),
        }

    with open(prepared_dir / "prepared_rmg_data.yaml", "w") as fh:
        yaml.safe_dump({"rxns": [dict(RXN)], "species": species_payload}, fh, sort_keys=False)

    species_key_by_id = {sp: compute_species_key(tpl) for sp, tpl in templates.items()}
    rxn_key = compute_reaction_key(RXN, species_key_by_id)

    return CaseInfo(
        dir=case_dir,
        site=site,
        rxn_key=rxn_key,
        species_key=species_key_by_id["sp_000"],
        slab=slab,
        ads_template=templates["sp_000"]["atoms"],
        seed_dir=case_dir / "rxn" / "seeds",
        ads_dir=case_dir / "rxn" / "Adsorbates" / species_key_by_id["sp_000"],
    )


def make_context(case: CaseInfo, *, sites=None, output_suffix: str = "vg0", **overrides):
    """在合成 case 上构造真实 WorkflowContext，并注入待搜索位点。"""
    from handlers.context import WorkflowContext

    ctx = WorkflowContext(
        path=str(case.dir),
        prepared_data_file=str(case.dir / "prepared_data" / "prepared_rmg_data.yaml"),
        slab_path=str(case.dir / "Fe5C2_510.vasp"),
        surface_normal=(0.0, 0.0, 1.0),
        normal_axis="z",
        top_x=1,
        bottom_freeze_threshold=BOTTOM_FREEZE_THRESHOLD,
        output_suffix=output_suffix,
        **overrides,
    )
    site_list = list(sites) if sites is not None else [case.site]
    for rxn in ctx.rxns_dict:
        rxn["valid_reactant_sites"] = site_list
        rxn["valid_reactant_sites_all"] = site_list
    return ctx


def copy_case(case: CaseInfo, dest: Path) -> CaseInfo:
    """复制合成 case（含 rxn/seeds 等既有文件）到新目录，用于隔离对比运行。"""
    shutil.copytree(case.dir, dest)
    return CaseInfo(
        dir=dest,
        site=case.site,
        rxn_key=case.rxn_key,
        species_key=case.species_key,
        slab=case.slab,
        ads_template=case.ads_template,
        seed_dir=dest / "rxn" / "seeds",
        ads_dir=dest / "rxn" / "Adsorbates" / case.species_key,
    )


def resolve_site_index(case: CaseInfo, preferred: int = PREFERRED_SITE) -> int:
    """在合成 slab 上解析一个真实存在的位点索引（优先 preferred）。"""
    ctx = make_context(case)
    n_sites = len(ctx.slab.sites)
    if n_sites <= 0:
        raise RuntimeError("synthetic slab exposes no surface sites")
    return min(int(preferred), n_sites - 1)


def load_baseline_module(relative_path: str, module_name: str):
    """Load immutable PR-4 reference, independent of Git HEAD or availability."""
    import hashlib
    import json
    import handlers

    baseline = Path(__file__).parent / "baselines" / "f1c240f"
    manifest = json.loads((baseline / "manifest.json").read_text())
    filename = Path(relative_path).name
    source = baseline / filename
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["sha256"][filename]
    spec = importlib.util.spec_from_file_location(f"handlers.{module_name}", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------
#  管线替身
# ----------------------------------------------------------------------

class FakeDP(Calculator):
    """DP 计算器替身：常数能量 + 零力（只需满足 ASE Calculator 协议）。"""

    implemented_properties = ["energy", "forces"]

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = dict(kwargs)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(self.atoms), 3))


class Records:
    """单次运行的记录：优化器起点、CCQN 调用、弹簧参数、方位角枚举调用。"""

    def __init__(self):
        self.optimizer_starts: list[np.ndarray] = []
        self.ccqn_calls: list[np.ndarray] = []
        self.spring_params: list[list[dict]] = []
        self.azimuth_calls: list[dict] = []
        self.bond_global: tuple[int, int] | None = None

    def reset(self):
        self.optimizer_starts.clear()
        self.ccqn_calls.clear()
        self.spring_params.clear()
        self.azimuth_calls.clear()


class _FakeOptimizer:
    """MDMin/QuasiNewton/Sella 通用替身：记录起点几何、不移动原子。"""

    def __init__(self, records: Records):
        self._records = records

    def __call__(self, atoms, *args, **kwargs):
        records = self._records

        class _Run:
            def __init__(self, inner_atoms):
                self.atoms = inner_atoms

            def run(self, fmax=None, steps=None):
                records.optimizer_starts.append(
                    np.array(self.atoms.get_positions(), dtype=float)
                )
                return False

        return _Run(atoms)


def make_fake_ccqn(records: Records, target_d: float | None):
    """CCQN 替身：把反应键拉伸到 ``target_d``（None = 保持几何不动）。"""

    class FakeCCQN:
        def __init__(self, atoms, logfile=None, e_vector_method=None,
                     reactive_bonds=None, saddle_optimizer=None, sella_kwargs=None):
            self.atoms = atoms
            self.logfile = logfile
            self.reactive_bonds = list(reactive_bonds or [])

        def run(self, fmax=None, steps=None):
            records.ccqn_calls.append(np.array(self.atoms.get_positions(), dtype=float))
            if target_d is not None and self.reactive_bonds:
                i, j = self.reactive_bonds[0]
                pos = self.atoms.get_positions()
                vec = pos[j] - pos[i]
                norm = float(np.linalg.norm(vec))
                if norm > 1e-9:
                    direction = vec / norm
                    delta = (float(target_d) - norm) / 2.0
                    pos[i] -= direction * delta
                    pos[j] += direction * delta
                    self.atoms.set_positions(pos)
            if self.logfile:
                with open(self.logfile, "w") as fh:
                    fh.write("fake ccqn run\n")
            return False

    return FakeCCQN


def make_fake_vibrations(bond_global, imag_attempts):
    """Vibrations 替身：按 ``_vib_try<k>`` 的 k 决定虚频个数。

    ``imag_attempts`` 内的 attempt 返回 1 个沿反应键方向的显著虚频（无漂移分量），
    其余返回纯实频（触发"显著虚频数不为 1"的拒绝分支）。
    """
    attempt_re = re.compile(r"_vib_try(\d+)$")

    class FakeVibrations:
        def __init__(self, atoms, name=None, indices=None, delta=0.01, nfree=2):
            self.atoms = atoms
            self.name = str(name)
            self.indices = list(indices or [])
            match = attempt_re.search(self.name)
            self.attempt = int(match.group(1)) if match else 1
            self.has_imag = self.attempt in imag_attempts

        def run(self):
            return None

        def summary(self):
            print(f"(fake vibrations {os.path.basename(self.name)} imag={self.has_imag})")

        def get_frequencies(self):
            if self.has_imag:
                return np.array([-450.0, 120.0, 300.0])
            return np.array([120.0, 300.0, 480.0])

        def get_mode(self, mode_idx):
            mode = np.zeros((len(self.atoms), 3))
            i, j = bond_global
            vec = self.atoms.get_positions()[j] - self.atoms.get_positions()[i]
            norm = float(np.linalg.norm(vec))
            if norm < 1e-9:
                return mode
            unit = vec / norm
            mode[i] = -unit
            mode[j] = unit
            return mode

    return FakeVibrations


class _LazyVibrations:
    """延迟决策的 Vibrations 替身：反应键全局索引在运行期才取到。"""

    def __init__(self, records: Records, imag_attempts):
        self._records = records
        self._imag_attempts = imag_attempts

    def __call__(self, atoms, name=None, **kwargs):
        bond = self._records.bond_global or (len(atoms) - 2, len(atoms) - 1)
        cls = make_fake_vibrations(bond, self._imag_attempts)
        return cls(atoms, name=name, **kwargs)


def patch_pipeline(ts_module, records: Records, *, ccqn_target_d: float | None,
                   imag_attempts, count_azimuth: bool = True):
    """把 ``ts_module`` 的 DP/优化器/CCQN/振动换成替身，返回还原函数。"""
    import utils.constraints as constraint_utils

    originals = {
        "DP": ts_module.DP,
        "MDMin": ts_module.MDMin,
        "QuasiNewton": ts_module.QuasiNewton,
        "Sella": ts_module.Sella,
        "CCQN": ts_module.CCQN,
        "Vibrations": ts_module.Vibrations,
        "harmonic": constraint_utils.HarmonicallyForcedDP,
        "enum": ts_module._enumerate_azimuth_orientation,
    }

    def fake_harmonic(*args, **kwargs):
        atom_bonds = list(kwargs.get("atom_bond_potentials") or [])
        records.spring_params.append(atom_bonds)
        for params in atom_bonds:
            if "ind1" in params and "ind2" in params:
                records.bond_global = (int(params["ind1"]), int(params["ind2"]))
        return FakeDP(*args, **kwargs)

    def counting_enum(*args, **kwargs):
        records.azimuth_calls.append({"n_candidates": len(kwargs.get("atom_bond_params_list") or [])})
        return originals["enum"](*args, **kwargs)

    ts_module.DP = FakeDP
    ts_module.MDMin = _FakeOptimizer(records)
    ts_module.QuasiNewton = _FakeOptimizer(records)
    ts_module.Sella = _FakeOptimizer(records)
    ts_module.CCQN = make_fake_ccqn(records, ccqn_target_d)
    ts_module.Vibrations = _LazyVibrations(records, imag_attempts)
    if count_azimuth:
        ts_module._enumerate_azimuth_orientation = counting_enum
    constraint_utils.HarmonicallyForcedDP = fake_harmonic

    def restore():
        ts_module.DP = originals["DP"]
        ts_module.MDMin = originals["MDMin"]
        ts_module.QuasiNewton = originals["QuasiNewton"]
        ts_module.Sella = originals["Sella"]
        ts_module.CCQN = originals["CCQN"]
        ts_module.Vibrations = originals["Vibrations"]
        ts_module._enumerate_azimuth_orientation = originals["enum"]
        constraint_utils.HarmonicallyForcedDP = originals["harmonic"]

    return restore


def distance_between(atoms: Atoms, i: int, j: int) -> float:
    pos = atoms.get_positions()
    return float(np.linalg.norm(pos[j] - pos[i]))


def read_stage_dir(case_dir: Path) -> dict:
    """读取某个 case 目录下的 TS 产物（供断言与逐位比较）。"""
    guess_dir = case_dir / "rxn" / "TS_guesses"
    archive_dir = case_dir / "rxn" / "TS_archive"
    out = {
        "guess_xyzs": {p.name: read(p) for p in sorted(guess_dir.glob("*.xyz"))},
        "summary_logs": {
            p.name: p.read_text()
            for p in sorted(guess_dir.glob("optimization_summary_*.log"))
        },
        "archives": {
            p.parent.name: read(p.parent / "initial.xyz")
            for p in sorted(archive_dir.glob("*/meta.yaml"))
        },
        "records_yaml": {
            p.name: p.read_text()
            for p in sorted(archive_dir.glob("ts_records*.yaml"))
        },
    }
    return out
