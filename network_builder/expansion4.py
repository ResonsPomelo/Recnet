"""闭包收口物种表（wave-5）：wave-4（档 c = 全 30 前沿物种）审计后残余的 **7 个产物**。

生成方式与 wave-1/2/3 同源：`recnet_adapter/tools/closure_audit.py --expansion2 --expansion3
--wave3-option c --iterate` 的输出 `Recnet/network_inputs/closure_audit-wave4c/closure_audit.json`
的 ``gaps`` 段（每个物种含 SMILES、formula、routes、new_channels_ready）。

判据与前几轮完全一致（连接点 = 悬挂价原子 ∪ 吸附锚点、成键阶 1/2/3、rdkit 收敛、净电荷 0、
C ≤ 2 / O ≤ 2、禁 O–O、残留悬挂价 ≤ 2）——**规则客观，不是人工挑选**。

状态：**已备好、未启动**——是否跑由维护者裁决（见台账 R175）。
"""

from __future__ import annotations

from typing import Tuple

from .expansion import anchor_of
from .species import SpeciesSpec


#: (key, smiles, formula, closure route) —— 顺序 = closure_audit-wave4c 的 gaps 字典序
_RAW: Tuple[Tuple[str, str, str, str], ...] = (
    ("C2H4O2_w4_1", "[H][C]OC([H])([H])O[H]", "C2H4O2", "CH + CH3O2_w3_1 (o=1)"),
    ("C2H4O2_w4_2", "[H][C]C([H])(O[H])O[H]", "C2H4O2", "CH + CH3O2_w3_2 (o=1)"),
    ("C2H5O2_w4_1", "[H]OC([H])([H])O[C]([H])[H]", "C2H5O2", "CH2 + CH3O2_w3_1 (o=1)"),
    ("C2H5O2_w4_2", "[H]OC([H])(O[H])[C]([H])[H]", "C2H5O2", "CH2 + CH3O2_w3_2 (o=1)"),
    ("C2H5O2_w4_3", "[H]OC([H])([H])C([H])([H])[O]", "C2H5O2", "O + C2H5O_w3_1 (o=1)"),
    ("C2H5O2_w4_4", "[H]OC([H])([O])C([H])([H])[H]", "C2H5O2", "O + C2H5O_w3_2 (o=1)"),
    ("C2H6O2_w4_1", "[H]OC([H])(O[H])C([H])([H])[H]", "C2H6O2", "CH3 + CH3O2_w3_2 (o=1)"),
)

EXPANSION4_SPECS: Tuple[SpeciesSpec, ...] = tuple(
    SpeciesSpec(key, key, "expansion4", smiles, anchor_of(smiles), "closure-round3-gap",
                formula, f"closure-audit route: {route}")
    for key, smiles, formula, route in _RAW
)
