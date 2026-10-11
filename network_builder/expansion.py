"""闭包扩展物种表（wave-1）：把「当前表一歩成键就能形成、但表里没有」的物种补进来。

**这份清单不是拍脑袋来的**，由 `recnet_adapter/tools/closure_audit.py` 机械枚举得到：

1. 取当前 29 条目 / 27 唯一身份物种表；
2. 对每个物种取连接点 = {带悬挂价（未成对电子）的原子} ∪ {表面吸附锚点 `ad_idx`}；
3. 枚举所有 (A, B) 组合与连接点对，成键阶 1/2/3，要求 rdkit 可收敛、净电荷 0、
   元素预算 C ≤ 2 / O ≤ 2、**禁止 O–O 键**（贫氧碳化物表面无过氧化学）、
   残留悬挂价 ≤ 2（吸附物靠悬挂价表面成键，这一条是必须的：否则会把
   formate / acetyl / ethoxy 这类关键中间体全部漏掉）；
4. 产物不在表内 ⇒ 缺失物种（gap）。

审计结果：当前表 27 物种 vs 机械闭包 **94 物种**（本轮 4 轮迭代），
第一轮就有 31 个「一步可形成」的缺失物种、合计 52 条「两碎片都已在表内」的
可枚举通道。本模块收录其中 **ready ≥ 2** 的 16 个（其余 15 个 ready ≤ 1
留待 wave-2 与人工裁决）。

每条 `SpeciesSpec` 的 `note` 保留了它的**形成路线**（哪两个表内物种、以什么键阶
偶联而成），因此每个新物种可追溯到它为什么出现在这里；`source="closure-round1"`
与 `manifest-required` 区分开，便于报告中分口径统计。

`ad_idx` 的取法（与审计同源，保证索引一致性）：解析 SMILES 后取**全部悬挂价原子**；
若为闭壳层物种（无悬挂价）则取 O（醇/醚/酸类以氧为吸附锚点），否则取第一个 C。
"""

from __future__ import annotations

from typing import Tuple

from rdkit import Chem, RDLogger

from .species import SpeciesSpec

RDLogger.DisableLog("rdApp.*")


def anchor_of(smiles: str) -> Tuple[int, ...]:
    """表面吸附锚点：悬挂价原子优先；闭壳层取 O，其次 C。"""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError(f"SMILES 无法解析: {smiles!r}")
    radicals = tuple(atom.GetIdx() for atom in mol.GetAtoms()
                     if atom.GetNumRadicalElectrons() > 0)
    if radicals:
        return radicals
    for symbol in ("O", "C"):
        indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == symbol]
        if indices:
            return (indices[0],)
    return (0,)


#: (key, smiles, formula, route)
_RAW: Tuple[Tuple[str, str, str, str], ...] = (
    ("CH4O_1", "[H]OC([H])([H])[H]", "CH4O", "CH3 + OH"),
    ("CH3O_1", "[H]O[C]([H])[H]", "CH3O", "CH2 + OH"),
    ("CH2O_1", "[H][C]O[H]", "CH2O", "CH + OH"),
    ("CH2O2_1", "[H]O[C]O[H]", "CH2O2", "COH + OH"),
    ("C2H6O_1", "[H]C([H])([H])OC([H])([H])[H]", "C2H6O", "CH3 + H3CO"),
    ("C2H5O_1", "[H]C([H])([H])C([H])([H])[O]", "C2H5O", "O + CH2CH3"),
    ("C2H5O_2", "[H][C]([H])OC([H])([H])[H]", "C2H5O", "CH2 + H3CO"),
    ("C2H4O_1", "[H]O[C]C([H])([H])[H]", "C2H4O", "CH3 + COH"),
    ("C2H4O_2", "[H][C]OC([H])([H])[H]", "C2H4O", "CH + H3CO"),
    ("C2H3O_1", "[H]C([H])([H])[C]=O", "C2H3O", "O + CCH3"),
    ("C2H3O_2", "[H]O[C]=C([H])[H]", "C2H3O", "CH2 + COH"),
    ("C2H3O_3", "[H][C]([H])C([H])=O", "C2H3O", "CH2 + HCO"),
    ("C2H2O_1", "[H][C]=[C]O[H]", "C2H2O", "CH + COH"),
    ("C2H2O_2", "[H][C]C([H])=O", "C2H2O", "CH + HCO"),
    ("C2H4_1", "[H][C]C([H])([H])[H]", "C2H4", "CH + CH3"),
    ("C2HO_1", "[H]OC#[C]", "C2HO", "C + COH"),
)

EXPANSION_SPECS: Tuple[SpeciesSpec, ...] = tuple(
    SpeciesSpec(key, key, "expansion", smiles, anchor_of(smiles), "closure-round1",
                formula, f"closure-audit route: {route}")
    for key, smiles, formula, route in _RAW
)
