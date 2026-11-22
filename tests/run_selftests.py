"""分阶段、串行运行现有 selftest 脚本（纯标准库，不联网）。

用法::

    python tests/run_selftests.py --phase p1
    python tests/run_selftests.py --phase all --python C:/path/python.exe
    python tests/run_selftests.py --phase p1 --report tests/reports/p1.json

设计约束：

* 只执行 tests/ 下已存在的脚本；不自动 SSH、不 pip install、不联网；
* 每个脚本串行执行，单独 timeout；子进程统一 UTF-8 输出；
* 未集成/未来 phase 的脚本明确记为 ``not-covered``，绝不算通过；
* 有失败/超时 -> exit 1；无失败但有未覆盖 -> exit 2；
  全部已覆盖且通过 -> exit 0；runner 自身配置/报告错误 -> exit 3；
* 默认不写任何文件；只有显式 ``--report PATH`` 才写 JSON 报告。
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass

SCHEMA_VERSION = 1
DEFAULT_TIMEOUT = 300.0
OUTPUT_TAIL_CHARS = 12000

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TestSpec:
    label: str
    filename: str
    globs: tuple = ()
    note: str = ""


# P0：权重/环境/存储。文件名以 git 历史中的实际名称为准；
# 旧名 _weights_review.py 已不存在，exact 后仍用 glob 兼容旧 worktree。
P0_SPECS = (
    TestSpec("environment_probe", "_selftest_environment_probe.py"),
    TestSpec("weights_logic", "_selftest_weights_logic.py"),
    TestSpec("storage", "_selftest_storage_inventory.py"),
    TestSpec("partial", "_selftest_partial_lifecycle.py"),
    TestSpec("h3", "_selftest_h3_adapter.py"),
    TestSpec("real_catalog", "_selftest_real_header_catalog.py"),
    TestSpec(
        "weights_review",
        "_selftest_registry_review.py",
        ("_selftest_*weights*review*.py", "_selftest_*registry*review*.py", "_weights_review.py"),
        "旧 _weights_review.py 名称缺失；优先实际 registry_review，glob 兜底",
    ),
)

# P1：启动/HTTP/executor/边界。仅运行当前 worktree 可见文件。
P1_SPECS = (
    TestSpec("empty_startup", "_selftest_empty_startup.py"),
    TestSpec("control_http", "_selftest_control_http.py"),
    TestSpec("operation_admission", "_selftest_operation_admission.py"),
    TestSpec("execution_boundary", "_selftest_execution_boundary.py"),
    TestSpec("check_call_sites", "_check_call_sites.py"),
)

PHASES = {"p0": P0_SPECS, "p1": P1_SPECS,
          "p2": (
              TestSpec("comfy_bindings", "_selftest_comfy_bindings.py"),
              TestSpec("te_residency", "_selftest_te_residency.py"),
              TestSpec("clipproj_resident", "_selftest_clipproj_resident.py"),
              TestSpec("te_manager", "_selftest_te_manager.py"),
              TestSpec("te_probe", "_selftest_te_probe.py"),
              TestSpec("weight_api", "_selftest_weight_api.py"),
              TestSpec("h3_patcher", "_selftest_h3_patcher.py"),
              TestSpec("attn_api", "_selftest_attn_api.py"),
          )}
PHASE_ORDER = ("p0", "p1", "p2")
STATUS_ORDER = ("passed", "failed", "timeout", "error", "not-covered")


def _decode_output(data):
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return str(data)


def _safe_print(text="", end="\n"):
    try:
        print(text, end=end, flush=True)
    except UnicodeEncodeError:
        # 极老终端兜底；正常路径已在 main() 里 reconfigure 成 UTF-8。
        print(text.encode("ascii", "backslashreplace").decode("ascii"),
              end=end, flush=True)


def resolve_script(spec, tests_dir):
    """返回 (Path|None, resolution)。exact 优先，glob 只做兼容兜底。"""
    tests_dir = Path(tests_dir)
    exact = tests_dir / spec.filename
    if exact.is_file():
        return exact, "exact"
    tried = [spec.filename]
    for pattern in spec.globs:
        tried.append(pattern)
        matches = sorted(
            path for path in tests_dir.glob(pattern)
            if path.is_file() and path.name not in
            ("run_selftests.py", "_selftest_test_runner.py")
        )
        if matches:
            return matches[0], "glob:%s" % pattern
    return None, "missing (tried: %s)" % ", ".join(tried)


def _relative(path, root):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def execute_spec(phase, spec, interpreter, timeout, root, tests_dir):
    """执行一个脚本（或明确记录 missing），返回可 JSON 化的记录。"""
    path, resolution = resolve_script(spec, tests_dir)
    record = {
        "phase": phase,
        "label": spec.label,
        "expected": spec.filename,
        "resolution": resolution,
        "resolved": None if path is None else _relative(path, root),
        "status": "not-covered",
        "reason": None,
        "returncode": None,
        "duration_s": 0.0,
        "output_tail": "",
        "output_chars": 0,
        "output_truncated": False,
    }
    if path is None:
        record["reason"] = (
            "文件不存在于本 worktree（未来/未集成）；明确 skip，不计为通过"
        )
        return record

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (str(root) if not old_pythonpath
                         else str(root) + os.pathsep + old_pythonpath)
    cmd = [str(interpreter), str(path)]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=env,
            check=False,
        )
        output = _decode_output(completed.stdout)
        record["status"] = "passed" if completed.returncode == 0 else "failed"
        record["returncode"] = completed.returncode
        if completed.returncode != 0:
            record["reason"] = "脚本返回 %s" % completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = _decode_output(exc.stdout or exc.stderr)
        record["status"] = "timeout"
        record["reason"] = "超过 %.2f 秒，已终止子进程" % timeout
    except OSError as exc:
        output = ""
        record["status"] = "error"
        record["reason"] = "无法启动解释器/脚本: %s" % exc
    except Exception as exc:
        # 单个脚本/子进程层的意外错误不终止整个 runner。
        output = ""
        record["status"] = "error"
        record["reason"] = "runner 执行异常: %s: %s" % (type(exc).__name__, exc)
    finally:
        record["duration_s"] = round(time.monotonic() - started, 3)

    record["output_chars"] = len(output)
    if len(output) > OUTPUT_TAIL_CHARS:
        record["output_truncated"] = True
        output = output[-OUTPUT_TAIL_CHARS:]
    record["output_tail"] = output
    return record


def summarize(records):
    counts = {status: 0 for status in STATUS_ORDER}
    for record in records:
        status = record.get("status")
        if status in counts:
            counts[status] += 1
    covered = (counts["passed"] + counts["failed"] + counts["timeout"]
               + counts["error"])
    hard_failures = counts["failed"] + counts["timeout"] + counts["error"]
    if hard_failures:
        overall, exit_code = "failed", 1
    elif covered == 0:
        overall, exit_code = "not-covered", 2
    elif counts["not-covered"]:
        overall, exit_code = "partial", 2
    else:
        overall, exit_code = "passed", 0
    counts.update({
        "total": len(records),
        "covered": covered,
        "overall": overall,
        "exit_code": exit_code,
    })
    return counts


def build_report(phase, interpreter, timeout, root, tests_dir, records):
    summary = summarize(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "python": str(interpreter),
        "timeout_seconds": timeout,
        "root": str(Path(root).resolve()),
        "tests_dir": str(Path(tests_dir).resolve()),
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "overall": summary["overall"],
        "exit_code": summary["exit_code"],
        "summary": summary,
        "tests": records,
    }


def run_selection(phase, interpreter, timeout, root, tests_dir, emit=None,
                  verbose=False):
    phase_names = PHASE_ORDER if phase == "all" else (phase,)
    records = []
    for phase_name in phase_names:
        for spec in PHASES[phase_name]:
            record = execute_spec(phase_name, spec, interpreter, timeout,
                                  root, tests_dir)
            records.append(record)
            if emit is not None:
                emit(record, verbose)
    return build_report(phase, interpreter, timeout, root, tests_dir, records)


def write_report(path, report):
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2,
                  sort_keys=True)
        handle.write("\n")
    return target


def _format_record(record, verbose=False):
    label = record["label"]
    status = record["status"]
    duration = record["duration_s"]
    if status == "passed":
        line = "[PASS] %-20s %6.2fs" % (label, duration)
    elif status == "not-covered":
        line = "[SKIP] %-20s not-covered" % label
    else:
        line = "[%s] %-20s %6.2fs" % (status.upper(), label, duration)
    lines = [line]
    if record.get("reason"):
        lines.append("       %s" % record["reason"])
    if record.get("resolved"):
        lines.append("       %s (%s)" % (record["resolved"],
                                         record["resolution"]))
    if record.get("output_tail") and (verbose or status != "passed"):
        lines.append("       ---- output tail ----")
        lines.extend("       " + item for item in
                     record["output_tail"].splitlines())
    return "\n".join(lines)


def list_selection(phase, tests_dir):
    phase_names = PHASE_ORDER if phase == "all" else (phase,)
    for phase_name in phase_names:
        _safe_print("== %s ==" % phase_name)
        for spec in PHASES[phase_name]:
            path, resolution = resolve_script(spec, tests_dir)
            if path is None:
                _safe_print("[SKIP] %-20s not-covered (%s)"
                            % (spec.label, resolution))
            else:
                _safe_print("[AVAILABLE] %-15s %s (%s)"
                            % (spec.label, path.name, resolution))
    _safe_print("LIST ONLY：未执行任何测试，退出码 0 不代表测试通过。")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="串行运行现有 selftest（纯 stdlib，不 SSH/pip/联网）")
    parser.add_argument("--phase", choices=PHASE_ORDER + ("all",), default="all",
                        help="p0=权重/环境，p1=启动/HTTP/executor/边界")
    parser.add_argument("--python", default=sys.executable,
                        help="运行测试脚本的解释器（默认当前解释器）")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                        help="每个脚本的超时秒数（默认 %.0f）"
                             % DEFAULT_TIMEOUT)
    parser.add_argument("--report", default=None,
                        help="显式 JSON 报告路径；不传则不写文件")
    parser.add_argument("--verbose", action="store_true",
                        help="通过脚本也打印输出尾部")
    parser.add_argument("--list", action="store_true",
                        help="只列出计划/可见性，不执行测试")
    args = parser.parse_args(argv)

    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    if args.list and args.report:
        parser.error("--list 不能与 --report 同用（list 不执行测试）")

    root = ROOT
    tests_dir = TESTS_DIR
    if not tests_dir.is_dir():
        _safe_print("ERROR: tests 目录不存在: %s" % tests_dir, end="\n")
        return 3
    if args.list:
        list_selection(args.phase, tests_dir)
        return 0

    def emit(record, verbose):
        _safe_print(_format_record(record, verbose=verbose))

    try:
        report = run_selection(
            args.phase, args.python, args.timeout, root, tests_dir,
            emit=emit, verbose=args.verbose,
        )
    except Exception:
        _safe_print("ERROR: runner 内部异常:")
        _safe_print(traceback.format_exc())
        return 3

    summary = report["summary"]
    _safe_print("")
    _safe_print(
        "SUMMARY total=%d covered=%d passed=%d failed=%d timeout=%d "
        "error=%d not-covered=%d"
        % (summary["total"], summary["covered"], summary["passed"],
           summary["failed"], summary["timeout"], summary["error"],
           summary["not-covered"]))
    _safe_print("RESULT %s (exit %d)"
                % (report["overall"], report["exit_code"]))

    if args.report:
        try:
            target = write_report(args.report, report)
        except Exception as exc:
            _safe_print("ERROR: 写入报告失败: %s" % exc)
            return 3
        _safe_print("REPORT %s" % target)
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
