"""recnet_adapter 命令行入口：``python -m recnet_adapter {show,check,run}``。"""
from __future__ import annotations

import argparse
import sys

from . import AdapterError, check_backend, describe, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recnet_adapter",
        description="使用显式模型输入运行 Recnet 反应网络管线。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show", help="显示模型/head 解析结果与运行环境（不加载模型）")
    for name in ("model", "head", "project-config"):
        show.add_argument("--" + name, default=None)

    check = sub.add_parser("check", help="端到端校验映射（管线调用式 vs 显式 head 参考 + 时延）")
    check.add_argument("--model", default=None, help="模型检查点路径（默认按 环境变量/配置 解析）")
    check.add_argument("--head", default=None, help="多头检查点的显式 head 名；单头模型不需要 head")
    check.add_argument("--project-config", default=None, help="显式选择项目 JSON")
    check.add_argument("--repeats", type=int, default=3, help="时延测试重复次数（默认 3）")

    run = sub.add_parser(
        "run",
        help="安装映射后，以相同 CLI 运行 run_dp_ts.py（其余参数原样透传）",
        add_help=False,
    )
    # Parse run through its own pipeline parser, including --help and model inputs.
    run.add_argument("args", nargs=argparse.REMAINDER)

    return parser


def main(argv: list | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        args = argparse.Namespace(command="run", args=argv[1:])
    else:
        args = parser.parse_args(argv)

    try:
        if args.command == "show":
            describe(model=args.model, head=args.head, project_config=args.project_config)
            return 0
        if args.command == "check":
            ok = check_backend(model=args.model, head=args.head, repeats=args.repeats, project_config=args.project_config)
            return 0 if ok else 1
        if args.command == "run":
            run_pipeline(args.args)
            return 0
    except AdapterError as exc:
        print(f"[recnet_adapter] ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - 干净报错；RECNET_ADAPTER_DEBUG=1 看栈
        import os

        if os.environ.get("RECNET_ADAPTER_DEBUG") == "1":
            raise
        print(
            f"[recnet_adapter] ERROR: {exc.__class__.__name__}: {exc}\n"
            "（设 RECNET_ADAPTER_DEBUG=1 可查看完整堆栈）",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
