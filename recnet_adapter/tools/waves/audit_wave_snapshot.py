#!/usr/bin/env python3
"""参数化"波次快照审计"（R353，2026-09-27）：新波交付快照 vs 既有基线的**非回归 + 增量 + 完整性**。

为什么需要它：C≤2 各波都有自己的 `audit_c*_snapshot.py`（硬编码该波数字）；C3 扩网（以及未来的换表面波）
需要同一套断言但**参数可配**，否则每上一波就得重写一个脚本。本工具把三类断言参数化：

1. **非回归**：基线快照表里的每条通道都必须出现在新表里，且 `barrier_best_eV` 与基线一致（± `--tol`）；
2. **增量**：`|新表| − |基线|` 必须等于 `--expected-new`（若给）；
3. **完整性**：`|新表|` 必须等于 `--expected-total`（若给）；另可校验 `gate/` 与 `thermo/` 是否存在。

用法::
    python audit_wave_snapshot.py --snapshot <new-snap-dir> --baseline <old-table.json> \\
        [--expected-new N] [--expected-total N] [--tol 1e-6] [--out-json report.json]
退出码：0 = 全部通过；1 = 有断言失败；2 = 输入缺失。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXIT_FAIL, EXIT_INPUT = 1, 2


def load_table(path: Path) -> dict:
    """读 `network_channels_final.json`（交付表）→ {channel: row}。"""
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    rows = data if isinstance(data, list) else (data.get("channels") or data.get("rows") or [])
    out = {}
    for r in rows:
        key = r.get("channel") or r.get("label")
        if key:
            out[key] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", required=True, help="新波快照目录（含 network_channels_final.json）")
    ap.add_argument("--baseline", required=True, help="基线交付表 JSON（上一波快照）")
    ap.add_argument("--expected-new", type=int, default=None, help="相对基线的期望新增通道数")
    ap.add_argument("--expected-total", type=int, default=None, help="期望的新表通道总数")
    ap.add_argument("--tol", type=float, default=1e-6, help="barrier_best_eV 非回归容差（eV）")
    ap.add_argument("--check-side-dirs", action="store_true", help="额外要求 gate/ 与 thermo/ 存在")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    snap = Path(args.snapshot)
    try:
        new = load_table(snap / "network_channels_final.json")
        base = load_table(Path(args.baseline))
    except FileNotFoundError as exc:
        print(f"[audit] FATAL: 缺文件 {exc}", file=sys.stderr)
        return EXIT_INPUT
    if not new or not base:
        print(f"[audit] FATAL: 表为空（new={len(new)} base={len(base)}）", file=sys.stderr)
        return EXIT_INPUT

    checks = []

    # 1) 非回归
    missing = [k for k in base if k not in new]
    drifted = []
    for k, br in base.items():
        nr = new.get(k)
        if not nr:
            continue
        try:
            b0, b1 = float(br.get("barrier_best_eV")), float(nr.get("barrier_best_eV"))
        except (TypeError, ValueError):
            continue
        if abs(b0 - b1) > args.tol:
            drifted.append({"channel": k, "baseline": b0, "new": b1, "delta": round(b1 - b0, 6)})
    checks.append({"name": "baseline_channels_present", "ok": not missing,
                   "detail": f"基线 {len(base)} 条，缺失 {len(missing)}", "missing": missing[:20]})
    checks.append({"name": "baseline_barriers_unchanged", "ok": not drifted,
                   "detail": f"漂移 {len(drifted)} 条（tol={args.tol}）", "drifted": drifted[:20]})

    # 2) 增量
    delta = len(new) - len(base)
    if args.expected_new is not None:
        checks.append({"name": "expected_new_channels", "ok": delta == args.expected_new,
                       "detail": f"实测新增 {delta}，期望 {args.expected_new}"})
    # 3) 完整性
    if args.expected_total is not None:
        checks.append({"name": "expected_total_channels", "ok": len(new) == args.expected_total,
                       "detail": f"实测总数 {len(new)}，期望 {args.expected_total}"})
    if args.check_side_dirs:
        for sub in ("gate", "thermo"):
            checks.append({"name": f"side_dir_present:{sub}", "ok": (snap / sub).exists(),
                           "detail": str(snap / sub)})

    ok = all(c["ok"] for c in checks)
    report = {"schema": "wave_snapshot_audit/1", "snapshot": str(snap), "baseline": str(args.baseline),
              "n_new": len(new), "n_base": len(base), "delta": delta, "tol": args.tol,
              "checks": checks, "pass": ok}
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for c in checks:
        print(f"[audit] {'PASS' if c['ok'] else 'FAIL'}  {c['name']}: {c['detail']}")
    print(f"[audit] 总结：{'全部通过' if ok else '**有失败项**'}（new={len(new)} base={len(base)} delta={delta}）")
    return 0 if ok else EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
