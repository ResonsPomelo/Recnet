#!/usr/bin/env python
"""通道级「全面性」审计：交付表 vs 生成器全枚举（逐通道，零 GPU）。

`closure_audit.py` 证明的是**物种**封闭性（94/94、0 gaps）；本工具证明**通道**完备性：
生成器对最终物种表枚举出的**全部单键断裂通道**（全量数据集）必须与交付表的通道集合一致——

* 交付表 ⊆ 全枚举：不允许出现"凭空通道"（生成器产不出的通道）；
* 全枚举 − 交付表 ⊆ 在跑波的预期新增：不允许长期缺口（交付完成后应为空集）。

口径：标签字符串 = ``"<reactant> -> <product ...>"``（与 `export_network_table.py`、
各波 gate 的 ``channel`` 字段同一构造），因此比较是纯集合运算、不依赖 rdkit。

用法::

    python audit_channel_completeness.py \\
        --table <snapshot>/network_channels_final.json \\
        --full  Recnet/network_inputs/c2r5/fe5c2_510_c2r5.prepared_rmg_data.yaml \\
        [--expect-diff-from <prepared yaml of the in-flight wave>] \\
        [--json-out audit_channel_completeness.json]

退出码：0 = 通过；2 = 不通过（明细打印）；1 = 输入错误。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _label(reactant: str, product: str) -> str:
    return f"{reactant} -> {' '.join(product.split())}"


def _labels_from_dataset(path: Path) -> "list[str]":
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    rxns = data.get("rxns") or []
    return [_label(str(r["reactant"]), str(r["product"])) for r in rxns]


def _labels_from_table(path: Path) -> "list[str]":
    data = json.loads(path.read_text(encoding="utf-8"))
    channels = data.get("channels") if isinstance(data, dict) else data
    if not isinstance(channels, list):
        raise ValueError(f"unrecognised table schema in {path}")
    return [str(row["channel"]) for row in channels]


def _duplicates(labels: "list[str]") -> "dict[str, int]":
    counts: "dict[str, int]" = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {label: n for label, n in counts.items() if n > 1}


def build_report(table: Path, full: Path, expect_diff: Path | None) -> "dict[str, object]":
    table_labels = _labels_from_table(table)
    full_rows = _labels_from_dataset(full)
    table_set = set(table_labels)
    full_set = set(full_rows)
    duplicates = _duplicates(full_rows)

    table_extra = sorted(table_set - full_set)
    pending = sorted(full_set - table_set)
    expected_pending = sorted(set(_labels_from_dataset(expect_diff))) if expect_diff else None

    ok_subset = not table_extra
    if expected_pending is not None:
        ok_pending = pending == expected_pending
    else:
        ok_pending = not pending
    passed = ok_subset and ok_pending

    report: "dict[str, object]" = {
        "schema": "ft2dp_channel_completeness_audit/1",
        "table": {"path": str(table), "channels": len(table_labels), "unique": len(table_set)},
        "full": {"path": str(full), "rows": len(full_rows), "unique": len(full_set),
                 "duplicate_labels": duplicates},
        "table_extra": table_extra,
        "pending": pending,
        "expected_pending": expected_pending,
        "checks": {"table_is_subset_of_full": ok_subset,
                   "pending_matches_expectation": ok_pending},
        "pass": passed,
    }
    return report


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="通道级网络完备性审计（交付表 vs 生成器全枚举）")
    parser.add_argument("--table", required=True, help="交付表 network_channels_final.json")
    parser.add_argument("--full", required=True, help="全量数据集 prepared_rmg_data.yaml")
    parser.add_argument("--expect-diff-from", default=None,
                        help="在跑波的数据集（其通道集应恰好等于『全枚举 − 交付表』）；缺省要求差集为空")
    parser.add_argument("--json-out", default=None, help="审计 JSON 输出路径")
    args = parser.parse_args(argv)

    table, full = Path(args.table), Path(args.full)
    if not table.is_file() or not full.is_file():
        print(f"[completeness] FATAL: missing input ({table} / {full})", file=sys.stderr)
        return 1
    expect = Path(args.expect_diff_from) if args.expect_diff_from else None
    if expect is not None and not expect.is_file():
        print(f"[completeness] FATAL: --expect-diff-from not found ({expect})", file=sys.stderr)
        return 1

    report = build_report(table, full, expect)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    t, f = report["table"], report["full"]  # type: ignore[assignment]
    print(f"[completeness] table={t['channels']} unique  full_rows={f['rows']} unique={f['unique']}"
          f"  duplicates={len(f['duplicate_labels'])}")
    print(f"[completeness] table_extra={len(report['table_extra'])}  pending={len(report['pending'])}"
          f"  expected_pending={'-' if report['expected_pending'] is None else len(report['expected_pending'])}")
    for label in report["table_extra"][:10]:
        print(f"  [extra] {label}")
    for label in report["pending"][:10]:
        print(f"  [pending] {label}")
    if report["pass"]:
        scope = ("交付表 == 全枚举（无缺口、无多余）"
                 if report["expected_pending"] is None
                 else "缺口 == 在跑波预期新增（无凭空通道）")
        print(f"[completeness] PASS — {scope}")
        return 0
    print("[completeness] FAIL — see report above", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
