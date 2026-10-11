"""闭包扩展物种表（wave-2）：round-0 闭包里 ready == 1 的剩余缺口。

wave-1（:mod:`network_builder.expansion`）只收了 ready ≥ 2 的 16 个物种，理由是"新增通道更多"；
但那个筛选**漏掉了若干经典 FT 含氧物**——乙醛（`[H]C(=O)C([H])([H])[H]`）、乙醇（`[H]OC([H])([H])C([H])([H])[H]`）、
甲酸（`[H]OC([H])=O`）、甲酸甲酯（`[H]C(=O)OC([H])([H])[H]`）在审计里 ready 都等于 1，因此落在本轮。

清单仍由 `recnet_adapter/tools/closure_audit.py` 机械枚举给出（同一套判据：连接点 = 悬挂价原子 ∪ 吸附锚点、
成键阶 1/2/3、rdkit 收敛、净电荷 0、C ≤ 2 / O ≤ 2、禁 O–O、残留悬挂价 ≤ 2），
只是把"未入库且 ready ≥ 1"的全部 14 个纳入——**规则客观，不是人工挑选**。

几何/锚点规则与 wave-1 完全一致（`anchor_of` 复用 wave-1 的实现，避免两套语义）。
"""

from __future__ import annotations

from typing import Tuple

from .expansion import anchor_of
from .species import SpeciesSpec


#: (key, smiles, formula, closure route)
_RAW: Tuple[Tuple[str, str, str, str], ...] = (
    ("C2H2O2_w2_1", "[H]C(=O)C([H])=O", "C2H2O2", "HCO + HCO"),
    ("C2H2O2_w2_2", "[H]OC([H])=C=O", "C2H2O2", "OH + CHCO"),
    ("C2H2O2_w2_3", "[H]O[C]C([H])=O", "C2H2O2", "COH + HCO"),
    ("C2H4O_w2_1", "[H]C(=O)C([H])([H])[H]", "C2H4O", "CH3 + HCO"),
    ("C2H4O_w2_2", "[H]OC([H])=C([H])[H]", "C2H4O", "OH + CHCH2"),
    ("C2H4O2_w2_1", "[H]C(=O)OC([H])([H])[H]", "C2H4O2", "HCO + H3CO"),
    ("C2H4O2_w2_2", "[H]O[C]OC([H])([H])[H]", "C2H4O2", "COH + H3CO"),
    ("C2H6O_w2_1", "[H]OC([H])([H])C([H])([H])[H]", "C2H6O", "OH + CH2CH3"),
    ("C2HO2_w2_1", "[H]C([O])=C=O", "C2HO2", "O + CHCO"),
    ("C2HO2_w2_2", "[H]O[C]=C=O", "C2HO2", "OH + CCO"),
    ("C2O2_w2_1", "[O][C]=C=O", "C2O2", "O + CCO"),
    ("CH2O2_w2_1", "[H]OC([H])=O", "CH2O2", "HCO + OH"),
    ("CHO2_w2_1", "[H]C([O])=O", "CHO2", "HCO + O"),
    ("CHO2_w2_2", "[H]O[C]=O", "CHO2", "COH + O"),
)

EXPANSION2_SPECS: Tuple[SpeciesSpec, ...] = tuple(
    SpeciesSpec(key, key, "expansion2", smiles, anchor_of(smiles), "closure-round1-ready1",
                formula, f"closure-audit route: {route}")
    for key, smiles, formula, route in _RAW
)
