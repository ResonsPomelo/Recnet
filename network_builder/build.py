"""Assemble the prepared dataset (yaml + templates) and the audit manifest.

Outputs (``--out-dir`` defaults to ``Recnet/network_inputs``):

* ``<name>.prepared_rmg_data.yaml`` - gold-standard aligned schema
* ``ads_templates/sp_xxx.xyz``      - one extxyz template per species
* ``MANIFEST.json``                 - sha256, counts, per-channel element
  checks, pipeline bond-alignment check, and the paper-step coverage verdict

Everything is deterministic: no timestamps, stable ordering, pinned seed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from ase import Atoms
from ase.data import covalent_radii

from .channels import (
    Channel,
    coverage_by_species,
    deduplicated_entries,
    element_balance,
    enumerate_channels,
)
from .species import SEED, Species, build_species_table, template_text

#: The paper network this dataset is measured against. ``network-Fe5C2-0813``
#: directory names are the chemistry; step ids follow
#: ``validation-pipeline/reaction/index.json`` and ``frames/manifest.json``.
PAPER_STEPS: Tuple[Dict[str, object], ...] = (
    {
        "id": "COdis",
        "paper": "CO* -> C* + O*",
        "paper_sites": "top / 4fold / 3fold",
        "status": "mapped",
        "expect": {"reactant": "CO", "bond_type": "C-O", "products": ["C", "O"]},
        "equivalence": "同一通道；论文三个位点由管线位点枚举覆盖"
                       "（handlers/ts.py 遍历 valid_reactant_sites），不需要三个通道条目",
    },
    {
        "id": "TS1_2H_to_CHCv_H",
        "paper": "2H -> CHCv + H",
        "paper_sites": "-",
        "status": "boundary_cv",
        "reason": "IS/FS 组成恒为 Fe80C36H2：H 与晶格碳结合/脱出晶格，涉及 Fe–C 晶格键与碳空位生成，"
                  "既不是吸附物内部的单键断裂，也无法用『一个前体 -> 两个吸附碎片』表示",
        "suggestion": "作为 slab/Cv 状态相关步骤单独处理：需要含 Cv 缺陷的专属底物 + 晶格碳参与的约束设置，"
                      "建议在 campaign 层以独立 case 处理，而非塞入本 prepared yaml",
    },
    {
        "id": "TS2_CHCv_H_to_CH_Cv_H",
        "paper": "CHCv -> CH + Cv + H",
        "paper_sites": "-",
        "status": "boundary_cv",
        "reason": "晶格 CH 出溶生成 Cv（Fe–C 晶格键断裂 + 空位生成），非吸附物断键",
        "suggestion": "同 TS1：归入 Cv 专项工作流；本数据集不强行表达",
    },
    {
        "id": "TS3_CH_CO_H_to_CH_CCv_O_H",
        "paper": "CH + CO + H -> CH + CCv + O + H",
        "paper_sites": "top",
        "status": "partial",
        "expect": {"reactant": "CO", "bond_type": "C-O", "products": ["C", "O"]},
        "equivalence": "吸附物层面等价于 CO* -> C* + O*（C–O 断裂）",
        "reason": "论文步的另一半是把 C 填入碳空位（CCv）并保留 O，属 Cv 位点化学，"
                  "当前 schema 只能表达 C–O 断裂",
        "suggestion": "用 CO* -> C* + O* 通道 + 位点选择（Cv 处位点）近似：空位在位点层面可由 "
                      "valid_reactant_sites 控制；若要精确复现 CCv，需要 Cv-aware 专项支持",
    },
    {
        "id": "C-CH",
        "paper": "C-CH（C1-coupling/C-CH，论文 TS4：C_Cv_CH_to_CCH_Cv）",
        "paper_sites": "-",
        "status": "mapped_with_cv_caveat",
        "expect": {"reactant": "CCH", "bond_type": "C-C", "products": ["CH", "C"]},
        "equivalence": "论文写成成键方向 C* + CH* -> CCH*，按断裂方向改写为 CCH* -> C* + CH*，势垒与方向无关",
        "reason": "论文该步同时涉及 Cv（组成 Fe80C37H1，C 位于空位处），Cv 部分本数据集不表达",
    },
    {
        "id": "C-CH4/CH3-H",
        "paper": "CH3-H（C-CH4 族第一步）",
        "paper_sites": "-",
        "status": "mapped",
        "expect": {"reactant": "CH4", "bond_type": "C-H", "products": ["CH3", "H"]},
        "equivalence": "论文给的是 C–H 断裂方向的甲烷脱氢阶梯；CH4 在物种表中只有气体条目"
                       "（ad_idx=[]，管线允许以几何中心锚定方式放在表面）",
    },
    {
        "id": "C-CH4/CH2-H",
        "paper": "CH2-H（C-CH4 族第二步）",
        "paper_sites": "-",
        "status": "mapped",
        "expect": {"reactant": "CH3", "bond_type": "C-H", "products": ["CH2", "H"]},
        "equivalence": "CH3* -> CH2* + H*",
    },
    {
        "id": "C-CH4/CH-H",
        "paper": "CH-H（C-CH4 族第三步）",
        "paper_sites": "-",
        "status": "mapped",
        "expect": {"reactant": "CH2", "bond_type": "C-H", "products": ["CH", "H"]},
        "equivalence": "CH2* -> CH* + H*",
    },
    {
        "id": "C-CH4/sur-H",
        "paper": "sur-H（C-CH4 族第四步）",
        "paper_sites": "-",
        "status": "mapped",
        "expect": {"reactant": "CH", "bond_type": "C-H", "products": ["C", "H"]},
        "equivalence": "CH* -> C* + H*",
    },
    {
        "id": "CH-CH",
        "paper": "CH-CH（CHCH 路径）",
        "paper_sites": "-",
        "status": "mapped",
        "expect": {"reactant": "CHCH", "bond_type": "C-C", "products": ["CH", "CH"]},
        "equivalence": "论文的 C–C 偶联 CH* + CH* -> CHCH* 按断裂方向写为 CHCH* -> CH* + CH*",
    },
    {
        "id": "TS5_TS6_CCH_H_to_CCH2",
        "paper": "CCH--CCH2",
        "paper_sites": "-",
        "status": "mapped",
        "expect": {"reactant": "CCH2", "bond_type": "C-H", "products": ["CCH", "H"]},
        "equivalence": "H 转移步 CCH* + H* -> CCH2* 改写为 CCH2* -> CCH* + H*",
    },
    {
        "id": "TS8_CCH2_H_to_CCH3",
        "paper": "CCH2--CCH3",
        "paper_sites": "-",
        "status": "mapped",
        "expect": {"reactant": "CCH3", "bond_type": "C-H", "products": ["CCH2", "H"]},
        "equivalence": "H 转移步 CCH2* + H* -> CCH3* 改写为 CCH3* -> CCH2* + H*"
                       "（父清单点名的改写范例）",
    },
    {
        "id": "CCH2_CHCH2",
        "paper": "CCH2--CHCH2（C1-coupling/CCH2--CHCH2）",
        "paper_sites": "-",
        "status": "mapped",
        "expect": {"reactant": "CHCH2", "bond_type": "C-H", "products": ["CCH2", "H"]},
        "equivalence": "CCH2* + H* -> CHCH2*（乙烯基 C–H 加氢后异构）改写为 CHCH2* -> CCH2* + H*",
    },
    {
        "id": "CCH3_CCCH3",
        "paper": "CCH3--CCCH3（TS_left / mid / right）",
        "paper_sites": "-",
        "status": "unresolved",
        "candidates": [
            {"reactant": "CCH3", "bond_type": "C-H",
             "note": "若 CCCH3 指 CCH3 的 H 转移异构体（两端口径同为 C2H3），即 CCH3* -> CCH2* + H*"},
            {"reactant": "CHCH2", "bond_type": "C-H",
             "note": "若 CCCH3 指乙烯基型 C2H3，则对应 CHCH2* -> CCH2* + H*"},
        ],
        "reason": "本地证据不足：frames/manifest.json 显示该族 IS/FS 组成同为 Fe80C38H3（单个 C2H3 吸附物），"
                  "两侧取自 TS_left/TS_right，无法从目录名判断 CCCH3 指哪种 C2H3 构型或是否为 C–C 事件",
        "suggestion": "请提供论文该族的结构/图示；确认后把候选之一改写为 mapped（实现只需在 PAPER_STEPS 改状态）",
    },
    {
        "id": "CCH3_Cv",
        "paper": "CCH3-Cv",
        "paper_sites": "-",
        "status": "boundary_cv",
        "reason": "吸附物与碳空位耦合（Cv 参与），组成为 Fe80C37H3；空位是晶格缺陷而非物种",
        "suggestion": "归入 Cv 专项工作流；本数据集用 CCH3* 的 C–H/C–C 通道覆盖其非空位部分",
    },
    {
        "id": "CHCH_CHCH2",
        "paper": "CHCH--CHCH2",
        "paper_sites": "-",
        "status": "mapped",
        "expect": {"reactant": "CHCH2", "bond_type": "C-H", "products": ["CHCH", "H"]},
        "equivalence": "CHCH* + H* -> CHCH2* 改写为 CHCH2* -> CHCH* + H*",
    },
    {
        "id": "CHCH2_CH2CH2",
        "paper": "CHCH2--CH2CH2",
        "paper_sites": "-",
        "status": "mapped",
        "expect": {"reactant": "CH2CH2", "bond_type": "C-H", "products": ["CHCH2", "H"]},
        "equivalence": "CHCH2* + H* -> CH2CH2* 改写为 CH2CH2* -> CHCH2* + H*",
    },
    {
        "id": "CHCH2_Cv",
        "paper": "CHCH2-Cv（本地 frames/manifest.json 观察到、父清单未点名的族）",
        "paper_sites": "-",
        "status": "boundary_cv",
        "reason": "吸附物与碳空位耦合，非单物种断键",
        "suggestion": "同 CCH3-Cv",
    },
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pipeline_geometry():
    """Import ``Recnet/utils/geometry.py`` by path (avoids package side effects).

    ``utils/__init__.py`` pulls in deepmd-backed constraint helpers, which are
    not importable in the data-preparation environment, so the module file is
    loaded directly.
    """
    path = Path(__file__).resolve().parents[1] / "utils" / "geometry.py"
    if not path.exists():
        return None, str(path)
    spec = importlib.util.spec_from_file_location("_recnet_geometry_probe", path)
    if spec is None or spec.loader is None:
        return None, str(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, str(path)


def check_pipeline_bond_pair(
    species: Species, broken_bond: Sequence[int], geometry_module
) -> Dict[str, object]:
    """Does the pipeline's own distance rule see the declared broken bond?"""
    atoms = Atoms(symbols="".join(species.symbols),
                  positions=np.array(species.positions, dtype=float), pbc=False)
    detected = {
        tuple(sorted((int(i), int(j))))
        for i, j in geometry_module.get_bond_connections(
            atoms, shift=0, cutoff=1.2, bond_type="nosurf"
        )
    }
    pair = tuple(sorted((int(broken_bond[0]), int(broken_bond[1]))))

    positions = np.array(species.positions, dtype=float)
    distance = float(np.linalg.norm(positions[pair[0]] - positions[pair[1]]))
    threshold = float(
        (covalent_radii[atoms[pair[0]].number] + covalent_radii[atoms[pair[1]].number]) * 1.2
    )
    return {
        "reactant": species.key,
        "broken_bond": list(pair),
        "detected": pair in detected,
        "distance_angstrom": round(distance, 4),
        "cutoff_angstrom": round(threshold, 4),
        "cutoff_ratio": round(distance / threshold, 4),
    }


def _paper_coverage(channels: Sequence[Channel]) -> Dict[str, object]:
    """Cross-check every mapped paper step against the enumerated channels."""
    index = {(channel.reactant, channel.bond_type, tuple(channel.products)): channel
             for channel in channels}
    by_reactant_type = {(channel.reactant, channel.bond_type) for channel in channels}

    mapped: List[Dict[str, object]] = []
    boundaries: List[Dict[str, object]] = []
    unresolved: List[Dict[str, object]] = []
    failures: List[str] = []

    for step in PAPER_STEPS:
        status = str(step["status"])
        if status in {"mapped", "mapped_with_cv_caveat", "partial"}:
            expect = step["expect"]  # type: ignore[index]
            match = index.get(
                (expect["reactant"], expect["bond_type"], tuple(expect["products"]))
            )
            if match is None and (
                (expect["reactant"], expect["bond_type"]) in by_reactant_type
            ):
                # Product fragment naming may differ (e.g. symmetric splits);
                # fall back to (reactant, bond_type) and record it.
                failures.append(
                    f"{step['id']}: expected {expect} matched only by reactant+bond_type"
                )
            elif match is None:
                failures.append(f"{step['id']}: expected channel {expect} NOT FOUND")
            mapped.append({
                "id": step["id"],
                "paper": step["paper"],
                "status": status,
                "channel": None if match is None else {
                    "reactant": match.reactant,
                    "products": list(match.products),
                    "broken_bond": list(match.broken_bond),
                    "bond_type": match.bond_type,
                    "label": f"{match.reactant_label} -> {match.product_label}",
                },
                "equivalence": step.get("equivalence"),
                "caveat": step.get("reason"),
            })
        elif status == "boundary_cv":
            boundaries.append({
                "id": step["id"], "paper": step["paper"],
                "reason": step["reason"], "suggestion": step["suggestion"],
            })
        elif status == "unresolved":
            candidates_found = [
                {"candidate": candidate,
                 "present": (candidate["reactant"], candidate["bond_type"])
                 in by_reactant_type}
                for candidate in step["candidates"]  # type: ignore[index]
            ]
            unresolved.append({
                "id": step["id"], "paper": step["paper"],
                "reason": step["reason"], "suggestion": step["suggestion"],
                "candidates": candidates_found,
            })
        else:
            failures.append(f"{step['id']}: unknown status {status!r}")

    # A channel counts as paper-covered when a mapped paper step points at the
    # same (reactant, bond type) pair; the increment list is what is left.
    mapped_pairs = {
        (entry["channel"]["reactant"], entry["channel"]["bond_type"])
        for entry in mapped
        if entry["channel"]
    }
    increments: Dict[str, int] = {}
    increment_channels: List[Dict[str, object]] = []
    covered_channels = 0
    for channel in channels:
        if (channel.reactant, channel.bond_type) in mapped_pairs:
            covered_channels += 1
            continue
        key = f"{channel.bond_type} (order {channel.bond_order:g})"
        increments[key] = increments.get(key, 0) + 1
        increment_channels.append({
            "label": f"{channel.reactant_label} -> {channel.product_label}",
            "reactant": channel.reactant,
            "bond_type": channel.bond_type,
            "bond_order": channel.bond_order,
            "reactant_is_gas": channel.reactant_is_gas,
        })

    gas_entry_channels = [c for c in channels if c.reactant_is_gas]

    return {
        "mapped": mapped,
        "boundaries": boundaries,
        "unresolved": unresolved,
        "verification_failures": failures,
        "paper_covered_channel_count": covered_channels,
        "increment_channels_by_bond_type": dict(sorted(increments.items())),
        "increment_channels": increment_channels,
        "channel_classes": {
            "surface_scission": len(channels) - len(gas_entry_channels),
            "gas_entry_scission_desorption_direction": len(gas_entry_channels),
            "pure_desorption_without_bond_break": 0,
            "note": "纯脱附（吸附物 -> 气体，不涉及断键）不能作为 RecNet 通道："
                    "handlers/energy.py 要求 len(product_species) >= 2；"
                    "气体侧由 ad_idx=[] 条目的气相热力学项覆盖",
        },
        "mapped_channel_reactant_bond_types": sorted(
            f"{reactant}:{bond_type}" for reactant, bond_type in mapped_pairs
        ),
    }


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def build_dataset(
    out_dir: os.PathLike | str,
    yaml_name: str = "fe5c2_510_c2.prepared_rmg_data.yaml",
    seed: int = SEED,
    bond_order_policy: str = "any",
    relative_root: Optional[os.PathLike | str] = None,
    expansion: bool = False,
    expansion2: bool = False,
    expansion3: bool = False,
    expansion4: bool = False,
    expansion5: bool = False,
    expansion6: bool = False,
    expansion6_tier: str = "closed-shell",
) -> Dict[str, object]:
    """Write templates, prepared yaml and MANIFEST.json. Returns the manifest."""
    out_dir = Path(out_dir).resolve()
    template_dir = out_dir / "ads_templates"
    template_dir.mkdir(parents=True, exist_ok=True)

    species_list = build_species_table(seed=seed, expansion=expansion, expansion2=expansion2,
                                       expansion3=expansion3, expansion4=expansion4,
                                       expansion5=expansion5, expansion6=expansion6,
                                       expansion6_tier=expansion6_tier)
    by_key = {species.key: species for species in species_list}
    sp_ids = {species.key: f"sp_{index:03d}" for index, species in enumerate(species_list)}

    channels, dropped = enumerate_channels(species_list, bond_order_policy=bond_order_policy)

    # ---- templates -------------------------------------------------------
    template_records = []
    for species in species_list:
        sp_id = sp_ids[species.key]
        text = template_text(species)
        path = template_dir / f"{sp_id}.xyz"
        path.write_text(text, encoding="utf-8")
        template_records.append({
            "sp_id": sp_id,
            "species": species.key,
            "name": species.name,
            "group": species.group,
            "formula": species.formula,
            "path": f"ads_templates/{sp_id}.xyz",
            "sha256": _sha256_text(text),
            "natoms": species.natoms,
            "ad_idx": list(species.ad_idx),
        })

    # ---- yaml ------------------------------------------------------------
    rxns_dump = [channel.as_yaml_record(sp_ids) for channel in channels]
    species_dump = {
        sp_ids[species.key]: {
            "name": species.name,
            # The gold standard clears this field (RMG adjacency lists are not
            # consumed by the handlers); real adjacency is kept in the manifest.
            "adjlist": "",
            "template_xyz": f"ads_templates/{sp_ids[species.key]}.xyz",
            "ad_idx": [int(idx) for idx in species.ad_idx],
        }
        for species in species_list
    }
    payload = {"rxns": rxns_dump, "species": species_dump}
    yaml_text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True,
                               default_flow_style=False)
    yaml_path = out_dir / yaml_name
    yaml_path.write_text(yaml_text, encoding="utf-8")

    # ---- checks ----------------------------------------------------------
    balances = [element_balance(channel, by_key) for channel in channels]
    balance_failures = [
        record for record in balances
        if not (record["elements_conserved"] and record["one_bond_broken"]
                and record["at_least_two_fragments"])
    ]

    geometry_module, geometry_path = _load_pipeline_geometry()
    if geometry_module is not None:
        bond_checks = [
            check_pipeline_bond_pair(by_key[channel.reactant], channel.broken_bond,
                                     geometry_module)
            for channel in channels
        ]
        # "Tightest margin" = the largest distance/cutoff ratio, i.e. the bond
        # the pipeline's distance rule could most easily stop recognising.
        tightest = max(bond_checks, key=lambda item: item["cutoff_ratio"])
        bond_check_summary = {
            "source": geometry_path,
            "cutoff_rule": "distance < 1.2 * (covalent_radius_i + covalent_radius_j)"
                           " — same call as handlers/ts.py",
            "checked": len(bond_checks),
            "failures": [item for item in bond_checks if not item["detected"]],
            "tightest_margin_channel": tightest,
        }
    else:
        bond_check_summary = {
            "source": geometry_path,
            "checked": 0,
            "failures": [],
            "note": "pipeline geometry module unavailable; check skipped",
        }

    coverage = coverage_by_species(channels)
    species_without_channel = sorted(
        species.key for species in species_list
        if coverage.get(species.key, {}).get("total", 0) == 0
    )

    by_bond_type: Dict[str, int] = {}
    by_bond_order: Dict[str, int] = {}
    for channel in channels:
        by_bond_type[channel.bond_type] = by_bond_type.get(channel.bond_type, 0) + 1
        key = f"order_{channel.bond_order:g}"
        by_bond_order[key] = by_bond_order.get(key, 0) + 1

    paper = _paper_coverage(channels)

    manifest: Dict[str, object] = {
        "schema_version": 1,
        "generator": "Recnet/network_builder (RMG-free prepared-schema generator)",
        "gold_standard_reference": {
            "path": "SAI:/org/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/recnet-runs/"
                    "ft2dpv22-demo-S/prepared_data/prepared_rmg_data.yaml",
            "aligned_fields_rxns": ["reactant", "product", "broken_bond",
                                    "reactant_species", "product_species"],
            "aligned_fields_species": ["name", "adjlist", "template_xyz", "ad_idx"],
            "template_format": 'extxyz "Properties=species:S:1:pos:R:3 pbc=\'F F F\'", '
                               "no cell, elements + coordinates only",
        },
        "environment": {
            "python": _python_version(),
            "rdkit": _module_version("rdkit"),
            "ase": _module_version("ase"),
            "numpy": _module_version("numpy"),
            "pyyaml": _module_version("yaml"),
            "etkdg_seed": seed,
        },
        "determinism": {
            "timestamps": "omitted by design (byte-identical rebuilds)",
            "ordering": "species table order -> sp_xxx; channels in table/edge order",
            "bond_order_policy": bond_order_policy,
        },
        "counts": {
            "species_entries": len(species_list),
            "distinct_molecules": len({species.identity for species in species_list}),
            "gas_entries": sum(1 for species in species_list if species.is_gas),
            "rxns": len(channels),
            "by_bond_type": dict(sorted(by_bond_type.items())),
            "by_bond_order": dict(sorted(by_bond_order.items())),
            "gas_entry_reactant_channels": sum(1 for channel in channels
                                               if channel.reactant_is_gas),
            "multi_order_channels": sum(1 for channel in channels
                                        if channel.bond_order != 1.0),
            "species_without_channel": species_without_channel,
            "coverage_by_species": dict(sorted(coverage.items())),
        },
        "checks": {
            "mass_conservation": {
                "checked": len(balances),
                "failures": balance_failures,
                "per_channel": balances,
            },
            "one_bond_broken_all_channels": not balance_failures,
            "products_at_least_two_all_channels": all(
                record["at_least_two_fragments"] for record in balances
            ),
            "broken_bond_visible_to_pipeline": bond_check_summary,
            "fragment_closure": {
                "dropped_candidates": dropped,
                "dropped_count": len(dropped),
            },
        },
        "files": {
            "yaml": {"path": yaml_name, "sha256": _sha256_text(yaml_text),
                     "bytes": len(yaml_text.encode("utf-8"))},
            "templates": template_records,
        },
        "species_entries": [
            {
                "sp_id": sp_ids[species.key],
                "key": species.key,
                "name": species.name,
                "group": species.group,
                "source": species.source,
                "smiles": species.smiles,
                "formula": species.formula,
                "natoms": species.natoms,
                "ad_idx": list(species.ad_idx),
                "identity": species.identity,
                "symbols": list(species.symbols),
                "bonds": [{"i": int(i), "j": int(j), "order": order}
                          for i, j, order in species.bonds],
                "note": species.note,
            }
            for species in species_list
        ],
        "deduplicated_entries": deduplicated_entries(species_list),
        "channels": [
            {
                "reactant": channel.reactant,
                "reactant_sp": sp_ids[channel.reactant],
                "products": list(channel.products),
                "product_sps": [sp_ids[key] for key in channel.products],
                "broken_bond": list(channel.broken_bond),
                "bond_type": channel.bond_type,
                "bond_order": channel.bond_order,
                "label": f"{channel.reactant_label} -> {channel.product_label}",
                "reactant_is_gas": channel.reactant_is_gas,
            }
            for channel in channels
        ],
        "paper_coverage": paper,
    }

    if paper["verification_failures"]:
        raise AssertionError(
            "paper coverage expectations failed: "
            + "; ".join(paper["verification_failures"])  # type: ignore[arg-type]
        )
    if balance_failures:
        raise AssertionError(f"element/bond checks failed for {len(balance_failures)} channels")
    if bond_check_summary.get("failures"):
        raise AssertionError(
            "declared broken_bond not detected by the pipeline rule for: "
            + ", ".join(str(item["reactant"]) for item in bond_check_summary["failures"])  # type: ignore[index]
        )

    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest["_yaml_path"] = str(yaml_path)
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def _python_version() -> str:
    import platform

    return platform.python_version()


def _module_version(name: str) -> str:
    module = __import__(name)
    return str(getattr(module, "__version__", "unknown"))
