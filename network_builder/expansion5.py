"""断键闭包补全（wave-6 最小集）：**CO2（气相）**。

来源：R224（`audit_dropped_cuts.py` 量化 76 条被丢 cut）+ R225（`rehearse_scission_closure.py`
完整断键闭包上界 +46 物种 / +244 通道；rdkit 判价显示 32 个缺失碎片里**唯一干净闭壳分子 = CO2**，
其余 8 个只能写成两性离子共振式、23 个为自由基/悬挂态，需化学评审后再定）。

纳入 CO2 解锁 **4 条通道**（本地实测 294 → 298）：
  * `CHO2_w2_1* -> CO2(g) H*`、`CHO2_w2_2* -> CO2(g) H*`（甲酸盐脱羧 ×2）
  * `C2H3O2_w3_2* -> CH3* CO2(g)`（乙酸盐脱羧）
  * `CO2(g) -> CO* O*`（CO2 解离；与 `CO* + O* -> CO2(g)` 同通道反向）

CO2 为**气相条目**（`ad_idx == ()`，与 H2O / CH4 同款）：`handlers/energy.py` 按 `ad_idx == []`
自动把该物种列为 gas-capable（无需额外白名单）。

冻结路径约束：本模块只在显式 `expansion5=True` 时被引入；无 flag 的构建必须保持字节一致
（`network_builder/tests/test_schema.py::test_rebuild_is_byte_identical` 等既有测试兜底）。
"""

from __future__ import annotations

from typing import Tuple

from .species import SpeciesSpec


EXPANSION5_SPECS: Tuple[SpeciesSpec, ...] = (
    SpeciesSpec(
        "CO2_GAS",
        "CO2",
        "gas",
        "O=C=O",
        (),
        "closure-scission-residue",
        "CO2",
        "gas-phase CO2 — the only clean closed-shell fragment of the scission residue "
        "(R224/R225); unlocks formate/acetate decarboxylation and CO2 dissociation",
    ),
)
