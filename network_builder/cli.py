"""Command line entry point for the prepared-dataset generator.

    # from the Recnet/ directory
    python -m network_builder.cli --out-dir network_inputs

    # or directly by path (a sys.path bootstrap keeps both working)
    python Recnet/network_builder/cli.py --out-dir Recnet/network_inputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from network_builder.build import build_dataset  # noqa: E402
from network_builder.species import SEED  # noqa: E402

DEFAULT_OUT_DIR = Path("network_inputs")
DEFAULT_YAML_NAME = "fe5c2_510_c2.prepared_rmg_data.yaml"


def _markdown_report(manifest: dict) -> str:
    """Generate every README table so the prose cannot drift from the data."""
    counts = manifest["counts"]
    coverage = counts["coverage_by_species"]
    entries = {entry["key"]: entry for entry in manifest["species_entries"]}
    lines: list = []

    lines += ["### 物种 × 通道类型覆盖表", "",
              "| 物种 | sp_id | 类别 | 分子式 | ad_idx | C–H | C–C | C–O | O–H | H–H | 合计 | 来源 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sp_ids_order(manifest):
        entry = entries[key]
        row = coverage.get(key, {})
        source = "需求清单" if entry["source"] == "manifest-required" else "本次新增(closure)"
        lines.append(
            "| {name} | {sp} | {group} | {formula} | {ad} | {ch} | {cc} | {co} | {oh} | {hh} | {total} | {src} |".format(
                name=key, sp=entry["sp_id"], group=entry["group"], formula=entry["formula"],
                ad=entry["ad_idx"] if entry["ad_idx"] else "[]",
                ch=row.get("C-H", 0), cc=row.get("C-C", 0), co=row.get("C-O", 0),
                oh=row.get("O-H", 0), hh=row.get("H-H", 0), total=row.get("total", 0),
                src=source,
            )
        )

    lines += ["", "### 论文步 → 本数据集通道映射", "",
              "| 论文步 | 论文写法 | 状态 | 我们的通道（断裂方向） | 等价改写 / 说明 |",
              "|---|---|---|---|---|"]
    for entry in manifest["paper_coverage"]["mapped"]:
        channel = entry["channel"]
        label = "-" if not channel else f"`{channel['label']}` ({channel['bond_type']}, broken_bond={channel['broken_bond']})"
        detail = entry.get("equivalence") or entry.get("caveat") or ""
        if entry.get("caveat"):
            detail = f"{entry.get('equivalence', '')}；注意：{entry['caveat']}"
        lines.append(f"| `{entry['id']}` | {entry['paper']} | {entry['status']} | {label} | {detail} |")
    for entry in manifest["paper_coverage"]["boundaries"]:
        lines.append(f"| `{entry['id']}` | {entry['paper']} | **边界（Cv 参与）** | 无法表达 | {entry['reason']} |")
    for entry in manifest["paper_coverage"]["unresolved"]:
        candidates = "；".join(
            f"`{item['candidate']['reactant']}:{item['candidate']['bond_type']}`" for item in entry["candidates"]
        )
        lines.append(f"| `{entry['id']}` | {entry['paper']} | **待论文确认** | 候选：{candidates} | {entry['reason']} |")

    lines += ["", "### 论文未覆盖、本次新增的通道（增量证据）", "",
              "| 断键类型（含键级） | 新增通道数 |", "|---|---|"]
    for key, value in manifest["paper_coverage"]["increment_channels_by_bond_type"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", f"新增通道合计 **{len(manifest['paper_coverage']['increment_channels'])}** 条；"
                  f"论文覆盖 **{manifest['paper_coverage']['paper_covered_channel_count']}** 条；"
                  f"总计 **{counts['rxns']}** 条。", ""]
    lines += ["| 新增通道 | 断键 | 键级 | 反应物为气体条目 |", "|---|---|---|---|"]
    for item in manifest["paper_coverage"]["increment_channels"]:
        lines.append(f"| `{item['label']}` | {item['bond_type']} | {item['bond_order']:g} | "
                     f"{'是' if item['reactant_is_gas'] else '否'} |")

    lines += ["", "### 论文步无法表达的清单与建议", "", "| 论文步 | 原因 | 建议 |", "|---|---|---|"]
    for entry in manifest["paper_coverage"]["boundaries"]:
        lines.append(f"| `{entry['id']}` | {entry['reason']} | {entry['suggestion']} |")
    for entry in manifest["paper_coverage"]["unresolved"]:
        lines.append(f"| `{entry['id']}` | {entry['reason']} | {entry['suggestion']} |")

    lines += ["", "### 通道类别计数", ""]
    for key, value in manifest["paper_coverage"]["channel_classes"].items():
        if key == "note":
            continue
        lines.append(f"- {key}: {value}")
    lines += ["", f"> {manifest['paper_coverage']['channel_classes']['note']}", ""]
    return "\n".join(lines) + "\n"


def sp_ids_order(manifest: dict):
    """Species keys in sp_id order (drives report row order)."""
    return [entry["key"] for entry in sorted(manifest["species_entries"],
                                             key=lambda item: item["sp_id"])]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="network_builder.cli",
        description="Build the RecNet prepared dataset for Fe5C2(510) C2-and-below chemistry",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="output directory (default: ./network_inputs, relative to the invocation directory)")
    parser.add_argument("--yaml-name", default=DEFAULT_YAML_NAME,
                        help="prepared yaml file name")
    parser.add_argument("--seed", type=int, default=SEED, help="ETKDG random seed")
    parser.add_argument("--bond-order-policy", choices=("any", "single"), default="any",
                        help="'any' (default) breaks one edge of any bond order, "
                             "'single' restricts to order-1 edges only")
    parser.add_argument("--markdown-report", default=None,
                        help="optional path for a generated markdown coverage report")
    parser.add_argument("--expansion", action="store_true",
                        help="append the closure-round-1 species table "
                             "(network_builder/expansion.py) after the frozen 29 entries; "
                             "use with --yaml-name to keep the frozen dataset untouched")
    parser.add_argument("--expansion2", action="store_true",
                        help="additionally append the wave-2 closures "
                             "(network_builder/expansion2.py: remaining ready==1 gaps, "
                             "including acetaldehyde / ethanol / formic acid)")
    parser.add_argument("--expansion3", action="store_true",
                        help="additionally append the wave-3 closure gaps "
                             "(network_builder/expansion3.py: the 30 species still missing from the "
                             "mechanical closure; prepared 2026-09-23, not launched — see "
                             "network_inputs/expansion_frontier.md)")
    parser.add_argument("--expansion4", action="store_true",
                        help="additionally append the wave-4 closure-completion species "
                             "(network_builder/expansion4.py: the 7 residual products found by the "
                             "wave-4 option-c audit; launched 2026-09-25 as wave-5)")
    parser.add_argument("--expansion5", action="store_true",
                        help="additionally append the scission-residue species "
                             "(network_builder/expansion5.py: CO2 gas — the clean closed-shell "
                             "fragment of the 76 dropped cuts; R224/R225)")
    parser.add_argument("--expansion6", action="store_true",
                        help="additionally append the C3 closure gaps "
                             "(network_builder/expansion6.py: the C3 species missing from the "
                             "mechanical closure; generated from closure_audit --max-c 3, R309). "
                             "Default tier = closed-shell; use --expansion6-tier to widen")
    parser.add_argument("--expansion6-tier", default="closed-shell",
                        choices=("closed-shell", "o1", "ready2", "all"),
                        help="expansion6 的物种档：closed-shell(默认,45) / o1(85) / ready2(136) / "
                             "all(241，含双悬挂价物种，需化学评审)")
    args = parser.parse_args(argv)

    manifest = build_dataset(
        out_dir=args.out_dir,
        yaml_name=args.yaml_name,
        seed=args.seed,
        bond_order_policy=args.bond_order_policy,
        expansion=args.expansion,
        expansion2=args.expansion2,
        expansion3=args.expansion3,
        expansion4=args.expansion4,
        expansion5=args.expansion5,
        expansion6=args.expansion6,
        expansion6_tier=args.expansion6_tier,
    )

    counts = manifest["counts"]

    print(f"[network_builder] species entries : {counts['species_entries']} "
          f"(distinct molecules {counts['distinct_molecules']}, "
          f"gas entries {counts['gas_entries']})")
    print(f"[network_builder] channels        : {counts['rxns']}")
    print("[network_builder] by bond type    : "
          + ", ".join(f"{key}={value}" for key, value in counts["by_bond_type"].items()))
    print("[network_builder] by bond order   : "
          + ", ".join(f"{key}={value}" for key, value in counts["by_bond_order"].items()))
    print(f"[network_builder] gas-entry reactants channels: "
          f"{counts['gas_entry_reactant_channels']}")
    print(f"[network_builder] species without channel     : "
          f"{counts['species_without_channel'] or 'none'}")

    coverage = coverage_by_species_from_manifest(manifest)
    print("[network_builder] ---- species x bond type ----")
    for key in sp_ids_order(manifest):
        row = coverage.get(key, {})
        print("  {:<8s} total={:<2d} C-H={:<2d} C-C={:<2d} C-O={:<2d} O-H={:<2d} H-H={:<2d}".format(
            key, row.get("total", 0), row.get("C-H", 0), row.get("C-C", 0),
            row.get("C-O", 0), row.get("O-H", 0), row.get("H-H", 0)))

    paper = manifest["paper_coverage"]
    print(f"[network_builder] paper steps mapped={len(paper['mapped'])} "
          f"boundary(Cv)={len(paper['boundaries'])} unresolved={len(paper['unresolved'])}")
    print("[network_builder] increment channels (paper-uncovered) by bond type: "
          + ", ".join(f"{k}={v}" for k, v in paper["increment_channels_by_bond_type"].items()))

    checks = manifest["checks"]
    print(f"[network_builder] checks: mass_conservation="
          f"{checks['mass_conservation']['checked']} channels / "
          f"{len(checks['mass_conservation']['failures'])} failures; "
          f"pipeline bond visibility="
          f"{checks['broken_bond_visible_to_pipeline']['checked']} channels / "
          f"{len(checks['broken_bond_visible_to_pipeline']['failures'])} failures")
    print(f"[network_builder] yaml     : {manifest['_yaml_path']}")
    print(f"[network_builder] manifest : {manifest['_manifest_path']}")
    print(f"[network_builder] yaml sha256: {manifest['files']['yaml']['sha256']}")

    if args.markdown_report:
        report_path = Path(args.markdown_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_markdown_report(manifest), encoding="utf-8")
        print(f"[network_builder] markdown : {report_path}")

    return 0


def coverage_by_species_from_manifest(manifest: dict):
    return manifest["counts"]["coverage_by_species"]


if __name__ == "__main__":
    raise SystemExit(main())
