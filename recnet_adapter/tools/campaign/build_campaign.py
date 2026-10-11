#!/usr/bin/env python3
"""Build a full Fe5C2(510) C2-network campaign for selected models.

Produces, under ``--out``::

    <out>/<label>-c<k>/          one case dir per (model, chunk)
        Fe5C2_510.vasp           copied from --template-case
        c_vacancy_structures/    copied from --template-case
        prepared_data/           subset of the network prepared dataset
    <out>/cases.tsv              farm_runner input (case_name, case_dir, model, head, extra_args, budget)

Channels are distributed round-robin after sorting by "expected hardness"
(C-O scissions first, then C-H/O-H/H-H, then C-C) so heavy channels are spread
over the chunks instead of piling into one worker.  ``--chunk-size`` controls
the nominal number of channels per case.
"""
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recnet_adapter.runtime_config import add_runtime_arguments, runtime_from_args  # noqa: E402

HARDNESS = {"C-O": 0, "C-C": 1, "O-H": 2, "H-H": 3, "C-H": 4}  # lower = slower/harder


def channel_bond_type(reactant_sp, broken_bond, species, templates_dir):
    """Element pair of the broken bond, taken from the reactant template itself.

    (An element-count difference cannot identify a bond: scission conserves the
    formula.  The first version of this helper did that and silently labelled
    every channel "C-H"; the round-robin chunking still spread channels, but the
    "difficulty balance" claim was wrong until this fix.)
    """
    rec = species[reactant_sp]
    tpl = Path(templates_dir) / rec["template_xyz"]
    symbols = []
    if tpl.exists():
        lines = tpl.read_text().splitlines()
        n = int(lines[0].split()[0])
        symbols = [ln.split()[0] for ln in lines[2:2 + n]]
    i, j = (list(broken_bond) + [None, None])[:2]
    if i is None or j is None or i >= len(symbols) or j >= len(symbols):
        return "unknown"
    pair = {symbols[i], symbols[j]}
    for tag in ("C-O", "C-C", "O-H", "H-H", "C-H"):
        if pair == set(tag.split("-")):
            return tag
    return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_runtime_arguments(ap, code=True, interpreter=True)
    ap.add_argument("--out", required=True, help="campaign output dir")
    ap.add_argument("--template-case", required=True, help="case dir providing slab + C-vacancy structures")
    ap.add_argument("--prepared", default=None,
                    help="prepared yaml to chunk (default: Recnet/network_inputs/"
                         "fe5c2_510_c2.prepared_rmg_data.yaml = frozen v1; pass the "
                         "closure-expanded c2r2 yaml to run the expanded network)")
    ap.add_argument("--chunk-size", type=int, default=11)
    ap.add_argument("--only-bond-types", default=None,
                    help="comma-separated bond types to include (e.g. C-O); default all")
    ap.add_argument("--rxn-index-file", default=None,
                    help="file with reaction indices (whitespace/comma separated) to restrict "
                         "the campaign to, e.g. the missing-channel list from a c2_gate run")
    ap.add_argument("--top-x", type=int, default=1,
                    help="value passed to the pipeline's --top-x (number of preferred sites "
                         "searched per channel). Round-2 follow-ups of a single-site campaign "
                         "use a larger value to widen site coverage")
    # 2026-09-24 (reaction-network R141/R146): the old flat default of 14400 s (4 h) silently
    # truncated every wave-3 case — on the 141-atom campaign slab a channel costs ~1.1 GPU·h
    # (measured: S-c0..c3 hit the 4 h wall after finishing only the vg0 sweep).  The default is now
    # chunk_size × 4500 s (1.25 h per channel, ~15 % headroom); pass --budget-s to override.
    ap.add_argument("--budget-s", type=int, default=None,
                    help="per-case wall-clock budget in seconds (default: chunk_size * 4500)")
    # 2026-09-27 (R410): 原实现把 `--slab Fe5C2_510.vasp` 与 `--bottom-freeze-threshold 13.456`
    # 写死在 extra_args 里 —— 给新面（(111)/(021)）建 campaign 时会指向**不存在的 510 slab**、
    # 并用错的冻结阈值（(111) 用 13.456 会把整块板冻住 ⇒ 弛豫全废）。现改为：
    #   --slab-name 默认 "auto"：取模板 case 里的 Fe5C2* 文件（(510) 行为不变）；
    #   --freeze-threshold 默认仍 13.456（(510) 历史值），新面须显式给（(111)=9.542、(021)=8.199）。
    ap.add_argument("--slab-name", default="auto",
                    help="extra_args 里 --slab 用的文件名（默认 auto = 模板 case 内的 Fe5C2* 文件）")
    ap.add_argument("--freeze-threshold", type=float, default=13.456,
                    help="--bottom-freeze-threshold（默认 13.456=(510) 历史值；(111)=9.542、(021)=8.199）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.budget_s is None:
        args.budget_s = int(args.chunk_size) * 4500

    runtime = runtime_from_args(args)
    models = runtime.models
    if not models:
        ap.error("select at least one model with --model, --model-s/m, or --project-config")
    root = runtime.root
    out = Path(args.out).resolve()
    net = Path(args.prepared).resolve() if args.prepared else (
        root / "Recnet/network_inputs/fe5c2_510_c2.prepared_rmg_data.yaml")
    subset = runtime.recnet_dir / "recnet_adapter/tools/subset_prepared.py"
    template = Path(args.template_case).resolve()
    # 模板里的 slab 文件名（默认 auto 检测；显式 --slab-name 覆盖）
    slab_candidates = sorted(p.name for p in template.iterdir() if p.name.startswith("Fe5C2"))
    if args.slab_name == "auto":
        if len(slab_candidates) != 1:
            raise SystemExit(f"模板 case 里的 Fe5C2* 文件不唯一：{slab_candidates}（请显式 --slab-name）")
        slab_name = slab_candidates[0]
    else:
        slab_name = args.slab_name
        if slab_name not in slab_candidates:
            raise SystemExit(f"--slab-name {slab_name} 不在模板 case 中：{slab_candidates}")
    payload = yaml.safe_load(net.read_text())
    rxns, species = payload["rxns"], payload["species"]
    wanted = None
    if args.only_bond_types:
        wanted = {t.strip() for t in args.only_bond_types.split(",") if t.strip()}
    restrict = None
    if args.rxn_index_file:
        text = Path(args.rxn_index_file).read_text().replace(",", " ")
        restrict = {int(t) for t in text.split() if t.strip()}
        out_of_range = sorted(i for i in restrict if not 0 <= i < len(rxns))
        if out_of_range:
            raise SystemExit(f"--rxn-index-file has out-of-range indices: {out_of_range}")
    scored = []
    for i, r in enumerate(rxns):
        if restrict is not None and i not in restrict:
            continue
        sp = r["reactant_species"][0]
        bt = channel_bond_type(sp, r["broken_bond"], species, net.parent)
        if wanted is not None and bt not in wanted:
            continue
        scored.append((HARDNESS.get(bt, 5), i, r["reactant"], bt))
    scored.sort()
    n_chunks = max(1, (len(scored) + args.chunk_size - 1) // args.chunk_size)
    buckets = [[] for _ in range(n_chunks)]
    for k, (_, i, _, _) in enumerate(scored):
        buckets[k % n_chunks].append(i)
    print(f"channels={len(scored)} (of {len(rxns)} total) chunks={n_chunks} "
          f"sizes={[len(b) for b in buckets]}")
    for k, b in enumerate(buckets):
        print(f"  chunk {k}: " + ", ".join(str(rxns[i]["reactant"]) for i in b))

    rows = []
    for arm in models:
        for k, idxs in enumerate(buckets):
            if not idxs:
                continue
            name = f"{arm}-c{k}"
            case = out / name
            if not args.dry_run:
                (case / "prepared_data").mkdir(parents=True, exist_ok=True)
                for f in template.iterdir():
                    if f.name.startswith("Fe5C2"):
                        shutil.copyfile(f, case / f.name)
                if not (case / "c_vacancy_structures").exists():
                    shutil.copytree(template / "c_vacancy_structures", case / "c_vacancy_structures")
                subprocess.run([runtime.python, str(subset), "--src", str(net),
                                "--out", str(case / "prepared_data"),
                                "--rxn-index", ",".join(map(str, idxs))], check=True)
            extra = (f"--slab {shlex.quote(str(case / slab_name))} --top-x {args.top_x} --use-c-vacancy-io "
                     f"--bottom-freeze-threshold {args.freeze_threshold}")
            rows.append((name, str(case), str(models[arm]), runtime.heads[arm] or "none", extra, str(args.budget_s)))

    if not args.dry_run:
        with open(out / "cases.tsv", "w") as fh:
            fh.write(f"# full C2-network campaign: {len(models)} models x chunks; "
                     f"budget={args.budget_s}s/case\n")
            for r in rows:
                fh.write("\t".join(r) + "\n")
        print("wrote", out / "cases.tsv", f"({len(rows)} cases)")
    else:
        print("[dry-run] case dirs not written; would emit", len(rows), "cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
