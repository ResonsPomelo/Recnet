#!/usr/bin/env python
"""GPU 并发探针：DPA4 单点吞吐 vs 「一张卡上放几个 case 进程」。

用途（RecNet 启动方案的资源决策）：RecNet 的每个 case 是一串**串行** ASE
优化循环（每次迭代一次 DP 单点 + 微小 Python/ASE 开销）。单进程时 GPU 常常
吃不饱，因此真正的问题是「一张卡上并发几个 case 进程」——不是「把 case 变成
CPU 任务」。

本脚本在**本作业可见的一张卡**上跑 k = 1/2/4 个并发子进程（spawn），每个子进程
加载同一模型后做 ``--repeats`` 次能量+受力单点，报告：

* ``cold_load_s``：模型加载/图构建（每 case 一次，与并发无关）；
* ``per_eval_ms``：稳态单次单点延迟（该 k 下的实测值）；
* ``evals_per_s_aggregate``：k 个进程合计吞吐（判断 GPU 是否还有余量）；
* ``gpu_util_mean/max``：子进程内 ``torch.cuda.utilization()`` 采样（若有）。

用法（父进程）::

    python gpu_concurrency_probe.py --structure S.xyz --model M.pt --head ft2dp \\
        --workers 1,2,4 --repeats 20 --out probe.tsv

注意：子进程用 ``multiprocessing`` spawn，CUDA_VISIBLE_DEVICES 由 sbatch 提供；
脚本自身不设置、不改写设备掩码。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _sample_gpu(stop: threading.Event, samples: list[tuple[float, float]]) -> None:
    """后台采样：本进程可见卡的利用率/显存（nvidia-smi，避免依赖 pynvml）。"""
    target = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    if not target:
        return
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
        f"--id={target}",
    ]
    while not stop.is_set():
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
            if out:
                util, mem = (x.strip() for x in out.splitlines()[0].split(","))
                samples.append((float(util), float(mem)))
        except Exception:  # noqa: BLE001 — 采样失败不影响计量
            pass
        stop.wait(0.25)


def _child(worker: int, args_dict: dict, pipe) -> None:
    """单个 worker：加载模型 → 预热 → 计时 repeats 次单点。"""
    try:
        import numpy as np
        from ase.io import read
        from recnet_adapter.model_selection import create_calculator

        structure = args_dict["structure"]
        model = args_dict["model"]
        head = args_dict["head"]
        repeats = int(args_dict["repeats"])
        warmup = int(args_dict["warmup"])

        atoms = read(structure)
        t0 = time.perf_counter()
        atoms.calc = create_calculator(model, head)
        cold = time.perf_counter() - t0

        rng = np.random.default_rng(1000 + worker)
        n_atoms = len(atoms)

        def one_eval() -> None:
            # 关键：**每次都微扰一位原子**，否则 ASE 会命中结果缓存，
            # 量到的是 Python 开销（曾实测出 0.079 ms 的假值）。
            atoms.positions[rng.integers(n_atoms)] += 1e-3 * rng.standard_normal(3)
            atoms.get_potential_energy()
            atoms.get_forces()

        for _ in range(warmup):
            one_eval()

        samples: list[tuple[float, float]] = []
        stop = threading.Event()
        sampler = threading.Thread(target=_sample_gpu, args=(stop, samples), daemon=True)
        sampler.start()
        try:
            t_start = time.perf_counter()
            for _ in range(repeats):
                one_eval()
            elapsed = time.perf_counter() - t_start
        finally:
            stop.set()
            sampler.join(timeout=5)

        pipe.send(
            {
                "worker": worker,
                "ok": True,
                "natoms": len(atoms),
                "cold_load_s": cold,
                "per_eval_ms": elapsed / repeats * 1e3,
                "repeats": repeats,
                "gpu_util_samples": [s[0] for s in samples],
                "gpu_mem_samples": [s[1] for s in samples],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
        )
    except Exception as exc:  # noqa: BLE001 — 子进程失败要回传而不是静默
        pipe.send({"worker": worker, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        pipe.close()


def _run_level(k: int, args_dict: dict) -> dict:
    """并发 k 个 worker，返回该 k 的聚合统计。"""
    ctx = mp.get_context("spawn")
    pipes = []
    procs = []
    for worker in range(k):
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_child, args=(worker, args_dict, child))
        proc.start()
        child.close()
        procs.append(proc)
        pipes.append(parent)

    results = []
    for pipe in pipes:
        results.append(pipe.recv() if pipe.poll(7200) else {"ok": False, "error": "timeout"})
    for proc in procs:
        proc.join()

    oks = [r for r in results if r.get("ok")]
    utils = [u for r in oks for u in r.get("gpu_util_samples", [])]
    mems = [m for r in oks for m in r.get("gpu_mem_samples", [])]
    per_eval = [r["per_eval_ms"] for r in oks]
    aggregate = (1000.0 / (sum(per_eval) / len(per_eval))) if per_eval else 0.0
    order = sorted(per_eval)
    return {
        "workers": k,
        "ok_workers": len(oks),
        "failed_workers": [r for r in results if not r.get("ok")],
        "natoms": oks[0]["natoms"] if oks else None,
        "cold_load_s": max((r["cold_load_s"] for r in oks), default=None),
        "per_eval_ms_mean": (sum(per_eval) / len(per_eval)) if per_eval else None,
        "per_eval_ms_max": order[-1] if order else None,
        "evals_per_s_aggregate": aggregate,
        "gpu_util_mean": (sum(utils) / len(utils)) if utils else None,
        "gpu_util_max": max(utils) if utils else None,
        "gpu_mem_used_mb": max(mems) if mems else None,
        "cuda_visible_devices": oks[0]["cuda_visible_devices"] if oks else "",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DPA4 GPU 并发单点吞吐探针")
    parser.add_argument("--structure", required=True, help="代表结构（.xyz/.vasp/POSCAR）")
    parser.add_argument("--model", required=True, help="DPA4 checkpoint (.pt)")
    parser.add_argument("--head", default=None, help="多头模型必须显式指定 head")
    parser.add_argument("--workers", default="1,2,4", help="并发行列表，逗号分隔")
    parser.add_argument("--repeats", type=int, default=20, help="每个 worker 的计量次数")
    parser.add_argument("--warmup", type=int, default=2, help="预热次数（不计时）")
    parser.add_argument("--out", default="", help="TSV 输出路径（空=只打印）")
    args = parser.parse_args(argv)

    levels = [int(x) for x in args.workers.split(",") if x.strip()]
    payload = {
        "structure": os.path.abspath(args.structure),
        "model": os.path.abspath(args.model),
        "head": args.head,
        "repeats": args.repeats,
        "warmup": args.warmup,
    }
    print(f"[probe] cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[probe] structure={payload['structure']} model={payload['model']} head={payload['head']!r}")

    rows = [_run_level(k, payload) for k in levels]

    header = [
        "workers",
        "ok_workers",
        "natoms",
        "cold_load_s",
        "per_eval_ms_mean",
        "per_eval_ms_max",
        "evals_per_s_aggregate",
        "gpu_util_mean",
        "gpu_util_max",
        "gpu_mem_used_mb",
        "cuda_visible_devices",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                "" if row.get(col) is None else f"{row[col]:.4f}" if isinstance(row.get(col), float) else str(row[col])
                for col in header
            )
        )
    table = "\n".join(lines)
    print(table)
    for row in rows:
        if row["failed_workers"]:
            print(f"[probe] workers={row['workers']} failures={json.dumps(row['failed_workers'])}", file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(table + "\n")
        print(f"[probe] wrote {args.out}")
    return 0 if all(r["ok_workers"] == r["workers"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
