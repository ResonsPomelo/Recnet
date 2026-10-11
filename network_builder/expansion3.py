"""闭包扩展物种表（wave-3）：当前表的下一轮闭包缺口（30 物种）。

由 `recnet_adapter/tools/make_expansion_frontier.py` 同源的审计输出生成：
`closure_audit-expanded/closure_audit.json`（`closure_audit.py --expansion2 --iterate`）。
判据与 wave-1/wave-2 完全一致（连接点 = 悬挂价原子 ∪ 吸附锚点、成键阶 1/2/3、rdkit 收敛、
净电荷 0、C ≤ 2 / O ≤ 2、禁 O–O、残留悬挂价 ≤ 2），**规则客观，不是人工挑选**。

状态：**已备好、未启动**——是否跑由维护者裁决（见 `network_inputs/expansion_frontier.md` 选项 a/b1/b2/d/c）。
"""

from __future__ import annotations

from typing import Tuple

from .expansion import anchor_of
from .species import SpeciesSpec


#: (key, smiles, formula, closure route)
_RAW: Tuple[Tuple[str, str, str, str], ...] = (
    ("C2H2O_w3_1", "[H]OC([H])=[C]", "C2H2O", "C + CH2O_1 (o=2)"),
    ("C2H2O2_w3_1", "[H]OC(=[C])O[H]", "C2H2O2", "C + CH2O2_1 (o=2)"),
    ("C2H2O2_w3_2", "[H]O[C]=[C]O[H]", "C2H2O2", "COH + COH (o=2)"),
    ("C2H2O2_w3_3", "[H][C]C(=O)O[H]", "C2H2O2", "CH + CHO2_w2_2 (o=1)"),
    ("C2H2O2_w3_4", "[H][C]OC([H])=O", "C2H2O2", "CH + CHO2_w2_1 (o=1)"),
    ("C2H3O_w3_1", "[H][C]=C([H])O[H]", "C2H3O", "CH + CH2O_1 (o=2)"),
    ("C2H3O2_w3_1", "[H]C(=O)C([H])([H])[O]", "C2H3O2", "O + C2H3O_3 (o=1)"),
    ("C2H3O2_w3_2", "[H]C([H])([H])C([O])=O", "C2H3O2", "O + C2H3O_1 (o=1)"),
    ("C2H3O2_w3_3", "[H]OC(=O)[C]([H])[H]", "C2H3O2", "CH2 + CHO2_w2_2 (o=1)"),
    ("C2H3O2_w3_4", "[H]O[C]([H])C([H])=O", "C2H3O2", "HCO + CH2O_1 (o=1)"),
    ("C2H3O2_w3_5", "[H]O[C]=C([H])O[H]", "C2H3O2", "COH + CH2O_1 (o=2)"),
    ("C2H3O2_w3_6", "[H][C]([H])OC([H])=O", "C2H3O2", "CH2 + CHO2_w2_1 (o=1)"),
    ("C2H3O2_w3_7", "[H][C]=C(O[H])O[H]", "C2H3O2", "CH + CH2O2_1 (o=2)"),
    ("C2H4O_w3_1", "[H][C]C([H])([H])O[H]", "C2H4O", "CH + CH3O_1 (o=1)"),
    ("C2H4O2_w3_1", "[H]OC(=O)C([H])([H])[H]", "C2H4O2", "CH3 + CHO2_w2_2 (o=1)"),
    ("C2H4O2_w3_2", "[H]OC([H])([H])C([H])=O", "C2H4O2", "HCO + CH3O_1 (o=1)"),
    ("C2H4O2_w3_3", "[H]O[C](O[H])[C]([H])[H]", "C2H4O2", "CH2 + CH2O2_1 (o=1)"),
    ("C2H4O2_w3_4", "[H]O[C]([H])[C]([H])O[H]", "C2H4O2", "CH2O_1 + CH2O_1 (o=1)"),
    ("C2H4O2_w3_5", "[H]O[C]C([H])([H])O[H]", "C2H4O2", "COH + CH3O_1 (o=1)"),
    ("C2H5O_w3_1", "[H]OC([H])([H])[C]([H])[H]", "C2H5O", "CH2 + CH3O_1 (o=1)"),
    ("C2H5O_w3_2", "[H]O[C]([H])C([H])([H])[H]", "C2H5O", "CH3 + CH2O_1 (o=1)"),
    ("C2H5O2_w3_1", "[H]C([H])([H])OC([H])([H])[O]", "C2H5O2", "O + C2H5O_2 (o=1)"),
    ("C2H5O2_w3_2", "[H]O[C](O[H])C([H])([H])[H]", "C2H5O2", "CH3 + CH2O2_1 (o=1)"),
    ("C2H5O2_w3_3", "[H]O[C]([H])C([H])([H])O[H]", "C2H5O2", "CH3O_1 + CH2O_1 (o=1)"),
    ("C2H5O2_w3_4", "[H]O[C]([H])OC([H])([H])[H]", "C2H5O2", "H3CO + CH2O_1 (o=1)"),
    ("C2H6O2_w3_1", "[H]OC([H])([H])C([H])([H])O[H]", "C2H6O2", "CH3O_1 + CH3O_1 (o=1)"),
    ("C2H6O2_w3_2", "[H]OC([H])([H])OC([H])([H])[H]", "C2H6O2", "H3CO + CH3O_1 (o=1)"),
    ("CH3O2_w3_1", "[H]OC([H])([H])[O]", "CH3O2", "O + CH3O_1 (o=1)"),
    ("CH3O2_w3_2", "[H]O[C]([H])O[H]", "CH3O2", "OH + CH2O_1 (o=1)"),
    ("CH4O2_w3_1", "[H]OC([H])([H])O[H]", "CH4O2", "OH + CH3O_1 (o=1)"),
)

EXPANSION3_SPECS: Tuple[SpeciesSpec, ...] = tuple(
    SpeciesSpec(key, key, "expansion3", smiles, anchor_of(smiles), "closure-round2-gap",
                formula, f"closure-audit route: {route}")
    for key, smiles, formula, route in _RAW
)
