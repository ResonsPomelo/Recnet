"""共享几何/校验助手：层分析、冻结阈值建议、最近距离、元素覆盖。

被 ``build_slab.py`` 与 ``preflight_case.py`` 复用；只依赖 numpy + ase。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from ase import Atoms

#: FT2DP（DPA4-Air-ZBL-MatPES 基座）微调域内的元素；域外元素可以跑但精度无担保
MODEL_DOMAIN_ELEMENTS = ("C", "Fe", "H", "O")


def group_layers(atoms: Atoms, axis: int = 2, tol: float = 0.4) -> list[dict]:
    """按沿 ``axis`` 的坐标把原子聚成层（间隙 > tol 即新层），返回按 z 升序的层列表。"""
    positions = atoms.get_positions()
    order = np.argsort(positions[:, axis])
    layers: list[dict] = []
    for idx in order:
        z = float(positions[idx, axis])
        if layers and (z - layers[-1]["z_max"]) <= tol:
            layers[-1]["indices"].append(int(idx))
            layers[-1]["z_max"] = z
        else:
            layers.append({"indices": [int(idx)], "z_min": z, "z_max": z})
    for layer in layers:
        layer["center"] = 0.5 * (layer["z_min"] + layer["z_max"])
        layer["elements"] = sorted({atoms[i].symbol for i in layer["indices"]})
        layer["n_atoms"] = len(layer["indices"])
    return layers


def suggest_bottom_freeze_threshold(
    atoms: Atoms, n_freeze_layers: int = 2, margin: float = 0.2, axis: int = 2
) -> tuple[float | None, list[dict]]:
    """建议 ``--bottom-freeze-threshold``：冻结底部 ``n_freeze_layers`` 层时取该层顶部 + margin。"""
    layers = group_layers(atoms, axis=axis)
    if len(layers) <= n_freeze_layers:
        return None, layers
    return float(layers[n_freeze_layers - 1]["z_max"] + margin), layers


def min_distance(atoms: Atoms, mic: bool = True) -> tuple[float, tuple[int, int]]:
    """最小原子间距离（默认考虑周期性）。"""
    dist = atoms.get_all_distances(mic=mic)
    np.fill_diagonal(dist, np.inf)
    ij = np.unravel_index(int(np.argmin(dist)), dist.shape)
    i, j = int(ij[0]), int(ij[1])
    return float(dist[i, j]), (i, j)


def top_layer_elements(atoms: Atoms, axis: int = 2, window: float = 1.5) -> list[str]:
    """表层窗口（最高原子向下 ``window`` Å）内的元素集合。"""
    positions = atoms.get_positions()
    z_max = float(positions[:, axis].max())
    mask = positions[:, axis] >= (z_max - window)
    return sorted({atoms[i].symbol for i in np.nonzero(mask)[0]})


def element_coverage(symbols: Iterable[str]) -> tuple[set[str], set[str]]:
    """返回 (全部元素, 域外元素)。"""
    all_elements = {str(s) for s in symbols}
    return all_elements, all_elements - set(MODEL_DOMAIN_ELEMENTS)


def normal_axis_alignment(atoms: Atoms) -> tuple[int, float]:
    """返回（最接近 +z 的晶格向量索引, 该向量与 z 轴夹角余弦）。"""
    cell = np.array(atoms.get_cell(), dtype=float)
    z = np.array([0.0, 0.0, 1.0])
    cosines = []
    for vec in cell:
        norm = np.linalg.norm(vec)
        cosines.append(abs(float(np.dot(vec / norm, z))) if norm > 1e-9 else 0.0)
    idx = int(np.argmax(cosines))
    return idx, float(cosines[idx])


def rotation_to_z(vector) -> np.ndarray:
    """构造把 ``vector`` 旋到 +z 的正交旋转矩阵。"""
    v = np.asarray(vector, dtype=float)
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    cos = float(np.clip(np.dot(v, z), -1.0, 1.0))
    if cos > 1.0 - 1e-12:
        return np.eye(3)
    if cos < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(v, z)
    axis = axis / np.linalg.norm(axis)
    ang = float(np.arccos(cos))
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(ang) * k + (1.0 - np.cos(ang)) * (k @ k)


def orient_normal_to_z(atoms: Atoms, normal_index: int = 2) -> Atoms:
    """旋转原子/晶胞，使第 ``normal_index`` 个晶格向量（slab 法向）对齐 +z。"""
    cell = np.array(atoms.get_cell(), dtype=float)
    rot = rotation_to_z(cell[normal_index])
    atoms = atoms.copy()
    atoms.set_cell(cell @ rot.T, scale_atoms=False)
    atoms.set_positions(atoms.get_positions() @ rot.T)
    return atoms
