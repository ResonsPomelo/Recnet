#!/usr/bin/env python3
"""RecNet case farm 启动器：在单个 Slurm allocation 内并发跑多个 reaction case。

定位
----
``sai/01_run_pipeline.sbatch`` =「一个作业一个 case 一张卡」；本脚本把它扩展成
「一个作业 N 个 case」，用 case 级并发替代排很多小作业（原作业粒度下 4V100 分区
每用户组可排队量是瓶颈）。每个 case 是**独立子进程 + 独立工作目录 + 独立日志**，
绑定一张 GPU。

设备语义（对齐 atst-tools runtime 冻结接口，见 docs/superpowers/specs/
2026-09-21-atst-runtime-interface-design.md §2/§4）：

* ``--devices`` 里的整数是**继承可见集合内的 0-based 逻辑序号**（默认
  ``CUDA_VISIBLE_DEVICES`` 的字面集合，未设置 = 整机可见），不是宿主卡号；
* 也接受完整物理 GPU UUID（``GPU-8-4-4-4-12``）；``MIG-`` 前缀不支持；
* 显式空掩码（``--devices ""``）视为**显式空请求**并拒绝，不会静默退化成 inherit；
* 本进程已设置且非空 ``CUDA_VISIBLE_DEVICES`` = caller-bound：请求必须是继承集合的
  子集，越界 fail-closed；整机可见 + 显式请求则拒绝（无法证明分配身份）。

线程预算（对齐同 SPEC §6）：``--threads`` 在**科学库初始化之前**写入子进程环境
（OMP/MKL/OPENBLAS/NUMEXPR/DP intra+inter-op）。优先级：
显式 ``--threads`` > 调用者已在环境里显式设定的值 > 默认 4。即 `--threads` 只在
调用者没显式给出该键时补齐，不覆盖调用者的显式选择。

失败分类：``timeout`` / ``oom`` / ``import_error`` / ``ccqn_unconverged`` / ``other``
（判据见 ``classify_failure`` 与 README「故障排查」）。分类是**诊断提示**，不是
科学结论；``other`` 一律保留完整日志供人工判读。

清单格式：TSV，列 = ``case_name  case_dir  model  head  extra_args  time_budget_s``，
支持 ``#`` 注释行与空行；``model``/``head``/``extra_args``（空格分隔、可用引号）/
``time_budget_s`` 留空 = 继承默认（``extra_args`` 留空即无附加参数）。

退出码：0 = 全部成功；1 = 至少一个 case 失败；2 = 用法/清单/设备参数错误。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recnet_adapter import resolve  # noqa: E402
from recnet_adapter.runtime_config import resolve_runtime  # noqa: E402

PROG = "farm_runner"
DEFAULT_THREADS = 4
LOG_TAIL_BYTES = 8 * 1024 * 1024  # 失败特征只在日志尾部扫描
#: 超时后先 TERM 整个进程组、等这么久再 KILL（避免残留 Sella/JAX 子进程，又不拖长尾）
KILL_GRACE_S = 3.0
RUN_MANIFEST_NAME = "RUN_MANIFEST.txt"
SUMMARY_NAME = "farm_summary.tsv"
FARM_DIRNAME = "farm"
DRY_MARKER = "@@farm-dry-run"

UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")
INDEX_RE = re.compile(r"^(0|[1-9][0-9]*)$")
MIG_RE = re.compile(r"^MIG-", re.IGNORECASE)

# ---------------------------------------------------------------------------
# 失败特征（诊断提示；判定证据 = 退出的信号/退出码 + 日志尾部匹配）
# ---------------------------------------------------------------------------
OOM_SIGNATURES = (
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "torch.outofmemoryerror",
    "cuda error: out of memory",
    "out of memory:",
    "cannot allocate memory",
    "memoryerror:",
    "std::bad_alloc",
    "insufficient memory",
    "torch.cuda.outofmemoryerror",
)
IMPORT_SIGNATURES = (
    "modulenotfounderror:",
    "importerror:",
    "cannot import name",
    "dll load failed",
    "undefined symbol:",
    "no module named",
)
CCQN_UNCONVERGED_SIGNATURES = (
    # 全断句 / 无 TS 接受（handlers/ts.py:469）
    "no ts accepted on preferred top-x sites",
    # TS 优化未收敛（handlers/ts.py:226/350）
    "optimization failed:",
    "extra sella saddle optimization failed:",
    # 优化器自身报的不收敛
    "non-converged",
    "did not converge",
    "not converged",
    "maximum number of steps",
    "convergence failure",
)


class FarmError(RuntimeError):
    """用法 / 清单 / 设备参数错误（退出码 2）。"""


class DeviceSpecError(FarmError):
    """``--devices`` 语法或可采性问题。"""


# ---------------------------------------------------------------------------
# runtime.devices 解析（对齐 atst SPEC §2/§4）
# ---------------------------------------------------------------------------

@dataclass
class DeviceView:
    """设备请求的 four-layer facts：requested / inherited / effective。"""

    requested: list[str]
    inherited_tokens: list[str] = field(default_factory=list)
    inherited_source: str = "unset"
    inherited_count_only: Optional[int] = None
    touched: bool = False

    @property
    def caller_bound(self) -> bool:
        return bool(self.inherited_tokens)

    def logical_index_of(self, token: str) -> Optional[int]:
        """token 在继承可见集合内的 0-based 逻辑序号（序号原样返回；UUID 查表）。"""
        if INDEX_RE.match(token):
            return int(token)
        if self.inherited_tokens:
            for pos, entry in enumerate(self.inherited_tokens):
                if entry.upper() == token.upper():
                    return pos
        return None

    def pool(self) -> list[str]:
        if not self.requested:
            raise DeviceSpecError(
                "runtime.devices must not be empty; omit the field to inherit all visible devices"
            )
        return list(self.requested)

    @staticmethod
    def _parse_inherited(raw: str) -> list[str]:
        return [tok.strip() for tok in raw.split(",") if tok.strip()]

    def check_admissible(self) -> None:
        """按 SPEC §4 可采性表裁决显式设备请求（fail-closed）。"""
        if not self.requested:
            return
        if self.inherited_source == "unset":
            raise DeviceSpecError(
                "explicit device selection is refused: the visible device set is not "
                "caller-bound and the trusted allocation is unknown "
                "(set CUDA_VISIBLE_DEVICES, e.g. in the sbatch template, or omit --devices "
                "to get the default 0..concurrency-1 binding)"
            )
        if self.inherited_source == "all" and self.inherited_count_only is not None:
            # GPU_REQ_* 类平台变量只给数量、不给身份 → 身份不可验证，不猜测。
            raise DeviceSpecError(
                f"explicit device selection is refused: the visible device set exceeds the "
                f"trusted allocation and device identity is unverified "
                f"(GPU request variables provide count={self.inherited_count_only} only; "
                f"set CUDA_VISIBLE_DEVICES to concrete devices)"
            )
        for tok in self.requested:
            if UUID_RE.match(tok):
                if self.inherited_source == "all":
                    raise DeviceSpecError(
                        "explicit device selection is refused: the visible device set is not "
                        "caller-bound and the trusted allocation is unknown"
                    )
                if not any(entry.upper() == tok.upper() for entry in self.inherited_tokens):
                    raise DeviceSpecError(
                        f"device token '{tok}' is not part of the inherited visible set "
                        f"({','.join(self.inherited_tokens) or '<empty>'})"
                    )
            elif INDEX_RE.match(tok):
                if self.inherited_source == "all":
                    continue  # count-only 已在上面拒绝；这里是防御性分支
                idx = int(tok)
                if idx >= len(self.inherited_tokens):
                    raise DeviceSpecError(
                        f"device index {idx} is outside the inherited visible set "
                        f"(size {len(self.inherited_tokens)})"
                    )
            else:  # pragma: no cover - 归一化阶段已拦截
                raise DeviceSpecError(
                    f"runtime.devices entry '{tok}' is not a valid 0-based device index or full GPU UUID"
                )


def read_device_view(raw: Optional[str]) -> DeviceView:
    """解析 ``--devices``；``None`` = 字段缺省（inherit），``""`` = 显式空请求。"""
    if raw is None:
        requested: list[str] = []
        touched = False
    else:
        touched = True
        tokens = [tok.strip() for tok in raw.split(",")]
        tokens = [tok for tok in tokens if tok != ""]
        if raw.strip() == "":
            raise DeviceSpecError(
                "runtime.devices must not be empty; omit the field to inherit all visible devices"
            )
        requested = validate_device_tokens(tokens, origin="--devices")

    view = DeviceView(requested=requested, touched=touched)
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None:
        if inherited.strip() == "":
            raise DeviceSpecError(
                "CUDA_VISIBLE_DEVICES is set to an empty value; explicit empty selection is "
                "refused. Unset it to inherit all visible devices."
            )
        tokens = DeviceView._parse_inherited(inherited)
        view.inherited_tokens = tokens
        view.inherited_source = "caller-bound" if tokens else "unset"
    else:
        # 记录「数量已知、身份未知」的分配事实（不用于猜测设备身份）
        for var in ("SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS"):
            value = os.environ.get(var)
            if value and value.isdigit() and int(value) > 0:
                view.inherited_source = "all"
                view.inherited_count_only = int(value)
                break
        else:
            view.inherited_source = "unset"
    return view


def validate_device_tokens(tokens: list[str], *, origin: str) -> list[str]:
    """语法校验（SPEC §2 冻结口径）：非负整数或完整 GPU UUID；重复/空/float/MIG 拒绝。"""
    cleaned = [str(tok).strip() for tok in tokens]
    if not cleaned:
        raise DeviceSpecError(
            "runtime.devices must not be empty; omit the field to inherit all visible devices"
        )
    for tok in cleaned:
        if tok == "":
            raise DeviceSpecError(
                "runtime.devices must not be empty; omit the field to inherit all visible devices"
            )
        if MIG_RE.match(tok):
            raise DeviceSpecError(
                f"runtime.devices does not support MIG device selection ('{tok}'); "
                "pass a full physical GPU UUID instead"
            )
        if UUID_RE.match(tok) or INDEX_RE.match(tok):
            continue
        raise DeviceSpecError(
            f"runtime.devices entry '{tok}' is not a valid 0-based device index or full GPU UUID "
            f"(from {origin})"
        )
    seen: set[str] = set()
    for tok in cleaned:
        key = tok.upper()
        if key in seen:
            raise DeviceSpecError(f"runtime.devices must not contain duplicate entries ({tok})")
        seen.add(key)
    return cleaned


# ---------------------------------------------------------------------------
# 清单
# ---------------------------------------------------------------------------

@dataclass
class Case:
    index: int
    name: str
    case_dir: Path
    model: Optional[str] = None
    head: Optional[str] = None
    extra_args: str = ""
    time_budget_s: Optional[float] = None

    @property
    def prepared(self) -> Path:
        return self.case_dir / "prepared_data" / "prepared_rmg_data.yaml"

    def extra_argv(self) -> list[str]:
        text = (self.extra_args or "").strip()
        return shlex.split(text) if text else []


@dataclass
class CaseResult:
    case: Case
    device: str = ""
    status: str = "pending"
    exit_code: Optional[int] = None
    failure_class: str = "none"
    wall_s: float = 0.0
    started_at: str = ""
    ended_at: str = ""
    work_dir: Optional[Path] = None
    log_path: Optional[Path] = None
    detail: str = ""
    dry_run: bool = False
    manifest_path: Optional[Path] = None


def parse_cases(path: Path) -> list[Case]:
    if not path.is_file():
        raise FarmError(f"清单文件不存在: {path}")
    cases: list[Case] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 2:
            raise FarmError(f"{path}:{lineno}: 至少需要 2 列（case_name<TAB>case_dir），实际 {len(fields)} 列")
        fields += [""] * (6 - len(fields))
        name = fields[0].strip()
        case_dir = fields[1].strip()
        if not name:
            raise FarmError(f"{path}:{lineno}: case_name 为空")
        if not case_dir:
            raise FarmError(f"{path}:{lineno}: case_dir 为空")
        if name in seen:
            raise FarmError(f"{path}:{lineno}: case_name 重复: {name}（工作目录会互相覆盖）")
        seen.add(name)
        budget_raw = fields[5].strip()
        budget: Optional[float] = None
        if budget_raw:
            try:
                budget = float(budget_raw)
            except ValueError as exc:
                raise FarmError(f"{path}:{lineno}: time_budget_s 不是数字: {budget_raw!r}") from exc
            if budget <= 0:
                raise FarmError(f"{path}:{lineno}: time_budget_s 必须为正: {budget_raw!r}")
        cases.append(
            Case(
                index=len(cases),
                name=name,
                case_dir=Path(case_dir).expanduser(),
                model=fields[2].strip() or None,
                head=fields[3].strip() or None,
                extra_args=fields[4].strip(),
                time_budget_s=budget,
            )
        )
    if not cases:
        raise FarmError(f"清单为空（只有注释/空行）: {path}")
    return cases


# ---------------------------------------------------------------------------
# 环境 / 元数据
# ---------------------------------------------------------------------------

def recnet_commit(recnet_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(recnet_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - commit 只是证据，不可用则 unknown
        return "unknown"


_SHA_CACHE: dict[tuple[str, int, int], str] = {}
_SHA_LOCK = threading.Lock()


def file_sha256(path: Optional[Path]) -> str:
    if path is None:
        return "unset"
    try:
        stat = path.stat()
    except OSError:
        return "missing"
    key = (str(path), stat.st_size, int(stat.st_mtime))
    with _SHA_LOCK:
        cached = _SHA_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        return f"unreadable({exc.__class__.__name__})"
    value = digest.hexdigest()
    with _SHA_LOCK:
        _SHA_CACHE[key] = value
    return value


def resolve_env_name() -> str:
    for var in ("DPEVA_DPA4_ENV_NAME", "CONDA_DEFAULT_ENV", "CONDA_ENV_PATH"):
        value = os.environ.get(var)
        if value:
            return Path(value).name if var == "CONDA_ENV_PATH" else value
    return "unknown (source dpeva-dpa4.env before launching)"


def child_env(case: Case, args: argparse.Namespace, device: str, model: Optional[str],
              head: Optional[str]) -> dict[str, str]:
    env = dict(os.environ)
    # 设备绑定：子进程的实际掩码
    env["CUDA_VISIBLE_DEVICES"] = device
    env.pop("ATST_RUNTIME_BOUND", None)
    # 线程预算：显式 --threads > 调用者已显式设定 > 默认
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "DP_INTRA_OP_PARALLELISM_THREADS", "DP_INTER_OP_PARALLELISM_THREADS"):
        if args.threads is not None:
            env[var] = str(args.threads)
        elif not env.get(var):
            env[var] = str(DEFAULT_THREADS)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in (str(args.recnet_dir), env.get("PYTHONPATH", "")) if p]
    )
    # "none" is a resolved value, not permission to inherit the caller's model/head.
    # Parent already resolved these files; cwd changes in each worker. Transfer
    # resolved values, not relative selectors that would be reinterpreted there.
    env.pop("RECNET_PROJECT_CONFIG", None)
    env.pop("RECNET_MODEL_CONFIG", None)
    env["RECNET_DP_MODEL"] = str(model) if model is not None else "none"
    env["RECNET_DP_HEAD"] = head if head is not None else "none"
    env["RECNET_FARM_CASE"] = case.name
    env["RECNET_FARM_RUN_ID"] = args.run_id
    env["RECNET_FARM_DEVICE"] = device
    # 每 case 私有 k=v（清单 extra_args 里的 `--env K=V`），用于 case 级环境覆盖
    env.update(case_env_overrides(case))
    return env


def case_env_overrides(case: Case) -> dict[str, str]:
    """从清单 extra_args 提取 case 级环境覆盖：``--env K=V``（farm 消费，不透传管线）。"""
    extra = case.extra_argv()
    overrides: dict[str, str] = {}
    for pos, token in enumerate(extra):
        value: Optional[str] = None
        if token == "--env" and pos + 1 < len(extra):
            value = extra[pos + 1]
        elif token.startswith("--env="):
            value = token.split("=", 1)[1]
        if value is None:
            continue
        if "=" not in value:
            raise FarmError(f"case {case.name}: --env 需要 K=V 形式，收到 {value!r}")
        key, _, val = value.partition("=")
        overrides[key.strip()] = val
    return overrides


def _strip_env_pairs(tokens: list[str]) -> list[str]:
    """移除 ``--env K=V``（farm 消费），其余顺序不变。"""
    out: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token == "--env":
            skip_next = True
            continue
        if token.startswith("--env="):
            continue
        out.append(token)
    return out


def build_pipeline_argv(case: Case, prepared: Path, slab: Optional[str], head: Optional[str]) -> list[str]:
    """构造 run_dp_ts.py 参数；清单里的 extra_args 是可选覆盖（与 01 sbatch 口径一致）。

    合并规则与 01 模板同构：调用者给的键优先。清单里的 ``--env K=V`` 是 farm 自己消费的
    case 级环境覆盖，不透传给管线。
    """
    extra = case.extra_argv()
    extra = _strip_env_pairs(extra)

    def provided(flag: str) -> bool:
        return any(tok == flag or tok.startswith(flag + "=") for tok in extra)

    argv: list[str] = []
    if not provided("--path"):
        argv += ["--path", str(case.case_dir)]
    if not provided("--prepared"):
        argv += ["--prepared", str(prepared)]
    if slab and not provided("--slab"):
        argv += ["--slab", slab]
    return argv + extra


# ---------------------------------------------------------------------------
# 日志与失败分类
# ---------------------------------------------------------------------------

def read_log_tail(log_path: Path, limit: int = LOG_TAIL_BYTES) -> str:
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as fh:
            if size > limit:
                fh.seek(size - limit)
            payload = fh.read()
    except OSError:
        return ""
    text = payload.decode("utf-8", errors="replace").lower()
    if not text.strip():
        return ""
    if text.count("\r") > 200:  # 终端进度条式输出（\r 覆盖行）
        text = " ".join(part for part in re.split(r"[\r\n]+", text) if part.strip())
    return text


def classify_failure(exit_code: Optional[int], timed_out: bool, log_text: str,
                     signals: list[str]) -> tuple[str, str]:
    """返回 (classification, detail)。判据顺序：timeout → import_error → oom → ccqn_unconverged → other。"""
    detail = ""
    if signals:
        detail = "log signatures: " + ", ".join(repr(sig) for sig in signals[:4])
    if timed_out:
        return "timeout", detail or "killed after exceeding time_budget_s"
    for sig in IMPORT_SIGNATURES:
        if sig in log_text:
            return "import_error", detail or f"log signature: {sig!r}"
    for sig in OOM_SIGNATURES:
        if sig in log_text:
            return "oom", detail or f"log signature: {sig!r}"
    for sig in CCQN_UNCONVERGED_SIGNATURES:
        if sig in log_text:
            return "ccqn_unconverged", detail or f"log signature: {sig!r}"
    if exit_code is None:
        return "other", detail or "process did not report an exit code"
    if exit_code < 0:
        return "other", detail or f"terminated by signal {-exit_code}"
    return "other", detail or f"exit code {exit_code}"


def collect_signatures(log_text: str) -> list[str]:
    found: list[str] = []
    for bucket in (OOM_SIGNATURES, IMPORT_SIGNATURES, CCQN_UNCONVERGED_SIGNATURES):
        for sig in bucket:
            if sig in log_text:
                found.append(sig)
    return found


def artifact_lines(work_dir: Path, case_dir: Path) -> list[str]:
    data: list[str] = []
    rxn_dir = case_dir / "rxn"
    if rxn_dir.is_dir():
        try:
            names = sorted(p.name for p in rxn_dir.iterdir())[:12]
        except OSError:
            names = []
        data.append(f"case_rxn_dir: {rxn_dir} ({len(names)} top entries: {', '.join(names)})")
    else:
        data.append(f"case_rxn_dir: {rxn_dir} (absent)")
    if work_dir.is_dir():
        try:
            entries = sorted(p.name for p in work_dir.iterdir())[:12]
        except OSError:
            entries = []
        data.append(f"farm_work_dir: {work_dir} (entries: {', '.join(entries)})")
    return data


# ---------------------------------------------------------------------------
# 每 case 执行
# ---------------------------------------------------------------------------

def prepare_work_dir(case: Case, args: argparse.Namespace) -> tuple[Path, Path]:
    try:
        case.case_dir.mkdir(parents=True, exist_ok=True)
        work_dir = case.case_dir / FARM_DIRNAME / "runs" / args.run_id / case.name
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FarmError(f"无法创建工作目录（case={case.name}, dir={case.case_dir}）: {exc}") from exc
    log_path = work_dir / "case.log"
    return work_dir, log_path


def run_case_dry(case: Case, args: argparse.Namespace, work_dir: Path, log_path: Path,
                 argv: list[str], model: Optional[str], device: str) -> int:
    """CPU 干跑：不启动计算（不建 DP 计算器），只做目录准备 + prepared 结构体检。"""
    notes: list[str] = []
    ok = True
    prepared = case.prepared
    lines = [
        f"{DRY_MARKER}: computed=false",
        f"{DRY_MARKER}: case={case.name}",
        f"{DRY_MARKER}: cwd={work_dir}",
        f"{DRY_MARKER}: device={device}",
        f"{DRY_MARKER}: model={model or '<inherit>'}",
        f"{DRY_MARKER}: argv={shlex.join(argv)}",
    ]
    lines.append(f"{DRY_MARKER}: case_dir_exists={case.case_dir.is_dir()}")
    lines.append(f"{DRY_MARKER}: prepared_exists={prepared.is_file()}")
    if not case.case_dir.is_dir():
        ok = False
        notes.append(f"case 目录不存在: {case.case_dir}")
    if not prepared.is_file():
        ok = False
        notes.append(f"prepared 数据不存在: {prepared}")
    if ok:
        try:
            import yaml  # 延迟导入：只有干跑体检需要

            payload = yaml.safe_load(prepared.read_text(encoding="utf-8")) or {}
            rxns = payload.get("rxns") if isinstance(payload, dict) else None
            species = payload.get("species") if isinstance(payload, dict) else None
            n_rxns = len(rxns) if isinstance(rxns, list) else 0
            n_species = len(species) if isinstance(species, dict) else 0
            if not n_rxns:
                ok = False
                notes.append("rxns 缺失或为空")
            if not n_species:
                ok = False
                notes.append("species 缺失或为空")
            lines.append(f"{DRY_MARKER}: rxns={n_rxns} species={n_species}")
            if isinstance(rxns, list):
                missing = [
                    i for i, rxn in enumerate(rxns)
                    if not isinstance(rxn, dict) or not rxn.get("reactant_species")
                    or not rxn.get("broken_bond")
                ]
                if missing:
                    ok = False
                    notes.append(f"rxn 字段缺失/非法的下标: {missing[:8]}")
        except Exception as exc:  # noqa: BLE001 - 干跑体检失败只降级为 WARN 语义
            lines.append(f"{DRY_MARKER}: yaml_parse_error={exc.__class__.__name__}: {exc}")
            notes.append(f"prepared yaml 无法解析（{exc.__class__.__name__}）——干跑未校验结构")
    skip = [flag for flag in ("--slab",) if not any(
        tok == flag or tok.startswith(flag + "=") for tok in argv)]
    lines.append(f"{DRY_MARKER}: missing_optional_args={skip}")
    lines.append(f"{DRY_MARKER}: structural_ok={ok}")
    lines.append(f"{DRY_MARKER}: notes={' | '.join(notes) if notes else 'none'}")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return 0 if ok else 1


def run_case_subprocess(case: Case, args: argparse.Namespace, work_dir: Path, log_path: Path,
                        env: dict[str, str], argv: list[str]) -> tuple[int, bool]:
    entry = args.recnet_dir / "recnet_adapter" / "run_dp_ts_ft2dp.py"
    cmd = [args.python, str(entry), *argv]
    started = time.monotonic()
    with open(log_path, "wb") as log_fh:
        log_fh.write(("[farm] cmd: " + shlex.join(cmd) + "\n").encode())
        log_fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(work_dir),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timeout = case.time_budget_s if case.time_budget_s else args.default_time_budget_s
        timed_out = False
        try:
            proc.wait(timeout=timeout if timeout else None)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        wall = time.monotonic() - started
        log_fh.write(f"\n[farm] wall_s={wall:.1f} rc={proc.returncode} timed_out={timed_out}\n".encode())
        log_fh.flush()
    return (proc.returncode if proc.returncode is not None else -9), timed_out


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """先 TERM 整个进程组（子进程用 start_new_session 自建组），宽限期后退再 KILL。

    只对「自己没有退出」的进程补 KILL；已退出的不再等（否则会被 shell 包装层
    忽略 SIGTERM 的行为拖出长尾）。
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
    try:
        proc.send_signal(signal.SIGTERM)
    except OSError:
        pass
    if proc.poll() is not None:
        return
    try:
        proc.wait(timeout=KILL_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass
    try:
        proc.send_signal(signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - 最后兜底
        pass


def execute_case(case: Case, args: argparse.Namespace, device: str, model: Optional[str],
                 head: Optional[str]) -> CaseResult:
    result = CaseResult(case=case, device=device, dry_run=args.dry_run)
    started_wall = datetime.now()
    t0 = time.monotonic()
    result.started_at = started_wall.strftime("%F %T")
    try:
        work_dir, log_path = prepare_work_dir(case, args)
    except FarmError as exc:
        result.status = "FAILURE"
        result.failure_class = "other"
        result.detail = str(exc)
        result.ended_at = datetime.now().strftime("%F %T")
        result.wall_s = time.monotonic() - t0
        result.manifest_path = None
        return result
    result.work_dir = work_dir
    result.log_path = log_path
    manifest_path = work_dir / RUN_MANIFEST_NAME
    result.manifest_path = manifest_path

    argv = build_pipeline_argv(case, case.prepared, None, head)
    env = child_env(case, args, device, model, head)
    rc: Optional[int] = None
    timed_out = False
    try:
        if args.dry_run:
            rc = run_case_dry(case, args, work_dir, log_path, argv, model, device)
        else:
            rc, timed_out = run_case_subprocess(case, args, work_dir, log_path, env, argv)
    except Exception as exc:  # noqa: BLE001 - 单 case 失败不拖垮 farm
        rc = None
        result.detail = f"launcher error: {exc.__class__.__name__}: {exc}"
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"\n[farm] launcher error: {exc.__class__.__name__}: {exc}\n")
        except OSError:
            pass
    result.wall_s = time.monotonic() - t0
    result.ended_at = datetime.now().strftime("%F %T")
    result.exit_code = rc

    log_text = read_log_tail(log_path) if log_path.exists() else ""
    signals = collect_signatures(log_text)
    if args.dry_run:
        ok = rc == 0
        result.status = "SUCCESS" if ok else "FAILURE"
        result.failure_class = "none" if ok else "other"
        result.detail = result.detail or ("dry-run structural check passed" if ok else "dry-run structural check failed")
    elif rc is None:
        result.status = "FAILURE"
        result.failure_class = "other"
    else:
        classification, detail = classify_failure(rc, timed_out, log_text, signals)
        result.failure_class = classification
        if rc == 0:
            result.status = "SUCCESS"
            result.failure_class = "none"
            result.detail = result.detail or "exit code 0"
        else:
            result.status = "FAILURE"
            result.detail = result.detail or detail

    write_manifest(result, args, argv, model, head, env)
    return result


# ---------------------------------------------------------------------------
# 汇总 / 证据
# ---------------------------------------------------------------------------

def write_manifest(result: CaseResult, args: argparse.Namespace, argv: list[str],
                   model: Optional[str], head: Optional[str], env: dict[str, str]) -> None:
    case = result.case
    manifest_path = result.manifest_path
    if manifest_path is None:
        return
    model_path = Path(model).expanduser() if model else None
    lines = [
        "",
        f"===== RUN_MANIFEST record {result.ended_at or datetime.now().strftime('%F %T')} =====",
        f"record_schema: farm_runner/1",
        f"case_name: {case.name}",
        f"case_dir: {case.case_dir}",
        f"run_id: {args.run_id}",
        f"model: {model or '<unconfigured>'}",
        f"model_sha256: {file_sha256(model_path)}",
        f"recnet_commit: {args.recnet_commit}",
        f"env_name: {args.env_name}",
        f"argv: {shlex.join(argv)}",
        f"device: {result.device}",
        f"device_inherited_visible: {args.device_inherited_label}",
        f"device_source: {args.device_source}",
        f"device_logical_index: {args.device_view.logical_index_of(result.device) if args.device_view else 'n/a'}",
        f"cuda_visible_devices_effective: {result.device}",
        f"threads: {args.threads if args.threads is not None else DEFAULT_THREADS}",
        f"thread_env: {', '.join(f'{k}={env[k]}' for k in THREAD_KEYS if k in env)}",
        f"head: {head if head is not None else 'none'}",
        f"time_budget_s: {case.time_budget_s if case.time_budget_s else (args.default_time_budget_s or 0)}",
        f"started_at: {result.started_at}",
        f"ended_at: {result.ended_at}",
        f"wall_s: {result.wall_s:.1f}",
        f"exit_code: {result.exit_code if result.exit_code is not None else 'none'}",
        f"failure_class: {result.failure_class}",
        f"status: {result.status}",
        f"dry_run: {result.dry_run}",
        f"work_dir: {result.work_dir}",
        f"log: {result.log_path}",
        f"detail: {result.detail or '-'}",
    ]
    if result.work_dir is not None:
        lines += artifact_lines(result.work_dir, case.case_dir)
    lines.append(f"===== END record =====")
    with _MANIFEST_LOCK:
        with open(manifest_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "DP_INTRA_OP_PARALLELISM_THREADS",
    "DP_INTER_OP_PARALLELISM_THREADS",
)
_MANIFEST_LOCK = threading.Lock()


def print_summary(results: list[CaseResult], args: argparse.Namespace, wall_s: float) -> None:
    counts: dict[str, int] = {}
    for res in results:
        key = "success" if res.status == "SUCCESS" else res.failure_class
        counts[key] = counts.get(key, 0) + 1
    total_case_s = sum(res.wall_s for res in results)
    max_case_s = max((res.wall_s for res in results), default=0.0)
    print("\n" + "=" * 96)
    print(f"[{PROG}] case farm 汇总（run_id={args.run_id}{'，DRY-RUN' if args.dry_run else ''}）")
    print("=" * 96)
    header = f"{'case':<28} {'device':<8} {'status':<9} {'class':<18} {'exit':>5} {'wall_s':>8}  detail"
    print(header)
    print("-" * 96)
    for res in results:
        exit_txt = "-" if res.exit_code is None else str(res.exit_code)
        print(f"{res.case.name:<28} {res.device:<8} {res.status:<9} {res.failure_class:<18} "
              f"{exit_txt:>5} {res.wall_s:>8.1f}  {res.detail[:40]}")
    print("-" * 96)
    for key in ("success", "timeout", "oom", "import_error", "ccqn_unconverged", "other"):
        if counts.get(key):
            print(f"  {key:<18} {counts[key]}")
    print(f"  total cases        {len(results)}")
    print(f"  total wall clock   {wall_s:.1f} s ({wall_s / 60:.1f} min)")
    print(f"  case-seconds       {total_case_s:.1f} s ({total_case_s / 3600:.2f} GPU·h 估计)")
    print(f"  critical path      {max_case_s:.1f} s (最长单 case)")
    print(f"  concurrency        {args.concurrency}  devices={','.join(args.device_pool) or '<inherit>'}")
    print(f"  summary tsv        {args.summary_path}")
    print("=" * 96)


SUMMARY_HEADER = (
    "#farm_runner/1\trun_id\trun_started_at\tdry_run\tcase_name\tcase_dir\tdevice\thead\n"
    "#record\tcase_status\tfailure_class\texit_code\twall_s\tstarted_at\tended_at\tlog\twork_dir\tdetail\n"
)


def write_summary(results: list[CaseResult], args: argparse.Namespace, wall_s: float) -> None:
    path = args.summary_path
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for res in results:
        key = "success" if res.status == "SUCCESS" else res.failure_class
        counts[key] = counts.get(key, 0) + 1
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(SUMMARY_HEADER)
        for res in results:
            fh.write("\t".join([
                "case",
                args.run_id,
                args.run_started_at,
                str(res.dry_run),
                res.case.name,
                str(res.case.case_dir),
                res.device,
                res.case.head or "-",
                res.status,
                res.failure_class,
                "-" if res.exit_code is None else str(res.exit_code),
                f"{res.wall_s:.1f}",
                res.started_at,
                res.ended_at,
                str(res.log_path or "-"),
                str(res.work_dir or "-"),
                (res.detail or "-").replace("\t", " "),
            ]) + "\n")
        fh.write("\t".join([
            "totals", args.run_id, args.run_started_at, str(args.dry_run), "ALL", "-", "-", "-",
            f"cases={len(results)}",
            ",".join(f"{k}={v}" for k, v in sorted(counts.items())),
            "-",
            f"{wall_s:.1f}",
            "-", "-", "-", "-",
            f"case_seconds={sum(r.wall_s for r in results):.1f} concurrency={args.concurrency} "
            f"devices={','.join(args.device_pool)}",
        ]) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="RecNet case farm：单 allocation 内并发跑多个 reaction case（每 case 一卡一进程）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "清单列：case_name<TAB>case_dir<TAB>model<TAB>head<TAB>extra_args<TAB>time_budget_s\n"
            "extra_args 里的 `--env K=V` 由 farm 消费（case 级环境覆盖），不会透传给管线。\n\n"
            "示例：\n"
            "  python farm_runner.py --cases campaign/cases.tsv --concurrency 4 --threads 4\n"
            "  python farm_runner.py --cases cases.tsv --concurrency 2 --dry-run --run-id dryrun01\n"
        ),
    )
    parser.add_argument("--cases", required=True, help="清单 TSV（case_name/case_dir/model/head/extra_args/time_budget_s）")
    parser.add_argument("--concurrency", type=int, default=1, help="并发 case 数（默认 1）")
    parser.add_argument("--devices", default=None,
                        help="可用卡列表（逗号分隔的逻辑序号或 GPU UUID；缺省 = 继承可见集合，未设置则 0..concurrency-1）")
    parser.add_argument("--threads", type=int, default=None,
                        help=f"每 case 进程线程预算（默认 {DEFAULT_THREADS}）；显式给出时覆盖调用者环境值")
    parser.add_argument("--model", default=None, help="清单 model 列为空时的兜底模型路径")
    parser.add_argument("--project-config", default=None, help="显式项目 JSON 配置")
    parser.add_argument("--head", default=None, help="清单 head 列为空时的兜底 head（none/null = 单头）")
    parser.add_argument("--recnet-dir", default=None, help="Recnet 仓库根（默认由脚本位置推断）")
    parser.add_argument("--python", default=None, help="子进程 Python（RECNET_PYTHON 或当前解释器）")
    parser.add_argument("--run-id", default=None, help="运行标识（默认 UTC 时间戳，用于工作目录与汇总）")
    parser.add_argument("--summary", default=None, help=f"汇总 TSV 路径（默认 <cwd>/{SUMMARY_NAME}）")
    parser.add_argument("--default-time-budget-s", type=float, default=None,
                        help="清单 time_budget_s 列缺省时的兜底单 case 墙钟预算（秒；缺省=不限）")
    parser.add_argument("--dry-run", action="store_true",
                        help="干跑：不启动计算、不建 DP 计算器，仍完成目录准备/manifest/汇总")
    return parser


def default_recnet_dir() -> Path:
    return resolve_runtime().recnet_dir


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime = resolve_runtime(recnet_dir=args.recnet_dir, python=args.python,
                              project_config=args.project_config)
    args.recnet_dir = runtime.recnet_dir
    args.python = runtime.python

    if args.concurrency < 1:
        raise FarmError(f"--concurrency 必须 >= 1（收到 {args.concurrency}）")
    if args.threads is not None and args.threads < 1:
        raise FarmError(f"--threads 必须 >= 1（收到 {args.threads}）")

    if not (args.recnet_dir / "recnet_adapter" / "run_dp_ts_ft2dp.py").is_file():
        raise FarmError(
            f"在 {args.recnet_dir} 下找不到 recnet_adapter/run_dp_ts_ft2dp.py；"
            "用 --recnet-dir 指定 Recnet 仓库根"
        )

    view = read_device_view(args.devices)
    view.check_admissible()
    if view.requested:
        pool = view.pool()
        if len(pool) < args.concurrency:
            print(f"[{PROG}] WARNING: 设备数 {len(pool)} < 并发数 {args.concurrency}；"
                  f"实际并发将被压到 {len(pool)}", file=sys.stderr)
        args.device_source = "explicit --devices"
    elif view.caller_bound:
        pool = list(view.inherited_tokens)
        args.device_source = "inherit (CUDA_VISIBLE_DEVICES)"
    else:
        pool = [str(i) for i in range(args.concurrency)]
        args.device_source = "default 0..concurrency-1 (CUDA_VISIBLE_DEVICES unset)"
        print(f"[{PROG}] WARNING: CUDA_VISIBLE_DEVICES 未设置且未给 --devices；"
              f"按 0..{args.concurrency - 1} 绑定（仅适合已知分到连续卡号的场景，"
              "GPU 作业请让 sbatch 模板设置 CUDA_VISIBLE_DEVICES）", file=sys.stderr)
    args.device_pool = pool
    args.device_view = view
    args.device_inherited_label = (
        ",".join(view.inherited_tokens) if view.inherited_tokens
        else (f"<unset; allocation count={view.inherited_count_only}>" if view.inherited_count_only
              else "<unset>")
    )

    cases = parse_cases(Path(args.cases).expanduser())

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    args.run_id = run_id
    args.run_started_at = datetime.now().strftime("%F %T")
    args.summary_path = Path(args.summary).expanduser().resolve() if args.summary else (
        Path.cwd() / SUMMARY_NAME
    )
    args.recnet_commit = recnet_commit(args.recnet_dir)
    args.env_name = resolve_env_name()
    selected_config = args.project_config or os.environ.get("RECNET_PROJECT_CONFIG")
    model_config = os.environ.get("RECNET_MODEL_CONFIG")
    fallback = resolve(model=args.model, head=args.head,
                       config_path=(None if selected_config or model_config else
                                    args.recnet_dir / "recnet_adapter/model_config.json"),
                       project_config=selected_config,
                       repo_root=None if selected_config or model_config else args.recnet_dir)
    fallback_model = str(fallback.model) if fallback.model is not None else None
    fallback_head = fallback.head

    print(f"[{PROG}] run_id={run_id} cases={len(cases)} concurrency={args.concurrency} "
          f"dry_run={args.dry_run}")
    print(f"[{PROG}] recnet_dir={args.recnet_dir} commit={args.recnet_commit} env={args.env_name}")
    print(f"[{PROG}] devices={','.join(pool)} source={args.device_source} "
          f"inherited_visible={args.device_inherited_label}")
    print(f"[{PROG}] threads={args.threads if args.threads is not None else DEFAULT_THREADS} "
          f"model={fallback_model or '<unconfigured>'} head={fallback_head if fallback_head is not None else 'none'}")
    if not args.dry_run:
        hint = _cuda_hint()
        if hint:
            print(f"[{PROG}] {hint}", file=sys.stderr)

    available = min(args.concurrency, len(pool))
    work_items = list(cases)
    t0 = time.monotonic()
    results: list[CaseResult] = []
    device_lock = threading.Lock()
    device_cursor = 0

    def next_device() -> str:
        nonlocal device_cursor
        with device_lock:
            device = pool[device_cursor % len(pool)]
            device_cursor += 1
            return device

    with concurrent.futures.ThreadPoolExecutor(max_workers=available) as pool_exec:
        future_map = {}
        pending = list(work_items)
        # 并发受设备池限制：先把 available 个 case 排上队
        in_flight: dict[concurrent.futures.Future, Case] = {}
        while pending or in_flight:
            while pending and len(in_flight) < available:
                case = pending.pop(0)
                device = next_device()
                resolved = resolve(model=case.model or fallback_model or "none",
                                   head=case.head if case.head is not None else (fallback_head or "none"),
                                   config_path=args.recnet_dir / "recnet_adapter/model_config.json",
                                   repo_root=args.recnet_dir)
                model = str(resolved.model) if resolved.model is not None else None
                head = resolved.head
                fut = pool_exec.submit(execute_case, case, args, device, model, head)
                in_flight[fut] = case
                future_map[fut] = (case, device)
            done, _ = concurrent.futures.wait(
                list(in_flight), return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                case, device = future_map.pop(fut)
                in_flight.pop(fut, None)
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    result = CaseResult(case=case, device=device, status="FAILURE",
                                        failure_class="other",
                                        detail=f"executor error: {exc.__class__.__name__}: {exc}")
                results.append(result)
                print(f"[{PROG}] {result.status:<7} {result.case.name:<28} device={result.device:<6} "
                      f"class={result.failure_class:<18} wall={result.wall_s:>7.1f}s "
                      f"log={result.log_path}")
    wall = time.monotonic() - t0
    results.sort(key=lambda res: res.case.index)

    write_summary(results, args, wall)
    print_summary(results, args, wall)
    failed = [res for res in results if res.status != "SUCCESS"]
    return 1 if failed else 0


def _cuda_hint() -> str:
    if os.environ.get("SLURM_JOB_ID"):
        return ""
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "提示：本进程未在 Slurm 作业内且 CUDA_VISIBLE_DEVICES 未设置；GPU 节点上应经 03_case_farm.sbatch 提交。"
    return ""


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FarmError as exc:
        print(f"[{PROG}] ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print(f"[{PROG}] interrupted", file=sys.stderr)
        sys.exit(130)
