# tools/ — case 准备与预检

> **Phase 3 布局（2026-10-07）**：实现已按职责迁入 `campaign/ network/ report/ dft/ surface/ waves/` 子包；
> 本目录下同名 `.py` 为 thin compat wrapper（直接执行、`-m recnet_adapter.tools.<name>` 与旧 import 继续可用）。
> 新调用推荐 canonical 形式，如 `python -m recnet_adapter.tools.surface.build_slab --help`；`finalize_network.sh` 仍在原位。

| 脚本 | 用途 |
|---|---|
| `build_slab.py` | 从 bulk 结构切任意 Miller 指数表面板（pymatgen），输出 Recnet 就绪的 slab CIF + 几何报告 + `--bottom-freeze-threshold` 建议 |
| `preflight_case.py` | 提交前预检 case 目录（slab + prepared_data）：字段/索引/元素/几何/（可选）DP 单点 |
| `make_seeds_scan.py` | 断键距离约束扫描生成 per-site TS seed（`<case>/rxn/seeds/`），用于 C–O 等易滑走通道 |
| `slab_utils.py` | 两者复用的几何助手（分层、最近距离、元素覆盖、法向对齐） |

## 典型流程

```bash
# 0) 依赖：pymatgen 仅在 build_slab 需要（SAI: `ase`/`atst` env 已含；本机: pip install pymatgen）

# 1) 列出 (510) 的所有终止，挑一个 Fe 终止面
python -m recnet_adapter.tools.surface.build_slab Fe5C2_bulk.vasp --miller 5 1 0 --list

# 2) 生成 slab（3x2 扩胞、~4 层、12 Å 真空）
python -m recnet_adapter.tools.surface.build_slab Fe5C2_bulk.vasp --miller 5 1 0 --termination 0 \
    --size 3 2 --layers 8 --vacuum 12 --out $R/recnet-runs/fe5c2-510-ft/Fe5C2_510.cif

# 3) 把 Fe2C(110) case 的 prepared_data/ 拷进来（反应网络与表面解耦，可复用）
cp -r $R/recnet-runs/fe2c110-case/prepared_data $R/recnet-runs/fe5c2-510-ft/

# 4) 预检
python -m recnet_adapter.tools.surface.preflight_case $R/recnet-runs/fe5c2-510-ft --dp-check

# 5) 通过后提交
RECNET_CASE=$R/recnet-runs/fe5c2-510-ft \
RECNET_SLAB=$R/recnet-runs/fe5c2-510-ft/Fe5C2_510.cif \
RECNET_EXTRA_ARGS="--use-c-vacancy-io --bottom-freeze-threshold <报告建议值>" \
  sbatch ../sai/01_run_pipeline.sbatch
```

## 约定与注意

- **z 轴朝上 + Fe 表层**：`handlers/context.py` 用「最高 Fe 层 z_max − 0.1」定义表层，并在 Fe 原子中
  找 Voronoi 位点。C 终止面会拿不到位点——若确需 C 终止，先联系维护者加 `RECNET_TOP_ELEMENT` 开关。
- `prepared` 数据里 `broken_bond` 是**对 reactant_species[0] 模板**的索引（`handlers/ts.py` 直接如此使用）；
  预检会检查越界并在报错信息里提示 prepare 阶段的常见错因。
- 冻结阈值：默认冻结底部 2 层；预检/建板脚本都会给出建议值，冷启动别用管线内置默认。
- `--dp-check` 建议在 GPU 节点跑（登录节点慢）；它只做单点，不做弛豫。

## TS seed 扫描（`make_seeds_scan.py`）

背景：C–O 断裂族（CO\* -> C\* + O\*）的 CCQN 常沿解离坐标滑走（末步 d(reactive)
远超拉伸上限）→ 所有取向被拒 → 拿不到 TS。该工具从**已弛豫吸附结构**出发做固定键长
扫描（`ase.constraints.FixBondLength`，每步弛豫其余原子），取能量最高帧写成
`<case>/rxn/seeds/<rxn_key>_site_<site>[_vg<k>].xyz`；管线（`handlers/ts.py`）默认
发现即用（`--no-ts-seed` 关闭），并跳过方位角枚举（azimuth 只留 0°）。

```bash
# 0) 先看反应表（确认通道与 rxn_key）
python -m recnet_adapter.tools.surface.make_seeds_scan <case> --list-rxns
# 1) 只规划、不下场算（不需要模型/GPU，不写文件）
python -m recnet_adapter.tools.surface.make_seeds_scan <case> --rxn 'CO*' --site 10 --vg 0 --dry-run
# 2) 真跑（GPU 节点；模型来自 --model 或 RECNET_DP_MODEL）
python -m recnet_adapter.tools.surface.make_seeds_scan <case> --rxn 'CO*' --site 10 --vg 0 \
    --bottom-freeze-threshold <与管线同值> --d-max 2.2 --fmax 0.05 --steps 200
# 3) 再跑一遍 TS 阶段（会发现 seed）或显式关闭覆写作对照
python -m recnet_adapter run --path <case> --prepared <case>/prepared_data/prepared_rmg_data.yaml ...
python -m recnet_adapter run --path <case> --no-ts-seed ...
```

输出：seed extxyz（原子顺序 = 组装结构）+ 台账
`rxn/seeds/SCAN_MANIFEST.json`（规范账本，按 `<rxn_key>_site_<site>[_vg<k>]` 合并）
与 `rxn/seeds/SCAN_MANIFEST_<rxn_key>_site_<site>[_vg<k>].json`（per-target 副本，
并发扫描互不争用）；每步 d/E/fmax/是否收敛 + 所取帧索引。

## wave-3（闭包前沿）预备：`prepare_wave3.py`

按 `network_inputs/expansion_frontier.md` 的档位（`b1/b2/d/c`）**建 wave-3 数据集 + 只挑"新增通道"的 campaign 子集**
（零 GPU、零集群写）：

```bash
PYTHONPATH=Recnet python -m recnet_adapter.tools.waves.prepare_wave3 --option d \
    --table validation-pipeline/summary/c2r2-network-20260922/network_channels_final.json \
    --out ~/scratch/wave3-d
# 随后（集群）：build_campaign.py --out $R/recnet-runs/c2r3-d --prepared <out>/prepared_rmg_data.yaml --chunk-size 8 --top-x 2
```

**实测（2026-09-23）**：选项 d ⇒ 73 物种条目 / 197 行 → 与已交付 138 通道比对得 **58 行 / 57 条新通道**；
`build_campaign.py --dry-run` ⇒ 16 个 case（8 chunk × 2 臂）。生成器 `network_builder/expansion3.py` 已就位
（30 个前沿物种，默认关闭，不影响冻结数据集：重生成 sha `3910ea17…` 不变）。
