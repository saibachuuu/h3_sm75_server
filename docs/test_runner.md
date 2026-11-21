# 分阶段 selftest runner

`tests/run_selftests.py` 是纯标准库入口，串行运行**当前 worktree 已存在**的
selftest 脚本。它不自动 SSH、不执行 `pip install`、不联网，也不修改代码。

## 用法

```powershell
# 只跑 P1 可见测试
python tests/run_selftests.py --phase p1

# 指定 ComfyUI / 项目虚拟环境解释器
python tests/run_selftests.py --phase p1 --python C:\path\to\python.exe

# 全部阶段，每个脚本 600 秒超时，并显式写 JSON 报告
python tests/run_selftests.py --phase all --timeout 600 --report tests\reports\all.json

# 只检查哪些脚本在本 worktree 可见（不执行测试）
python tests/run_selftests.py --phase all --list
```

默认解释器是当前 `sys.executable`；默认不写任何文件。只有显式传入
`--report PATH` 才写报告（父目录不存在会自动创建）。测试产物仍应放在
`tests/` 下已有 gitignore 覆盖的位置，不要把报告提交进仓库。

## 阶段与文件

P0（权重/环境/存储）：

| label | 实际文件 |
|---|---|
| `environment_probe` | `tests/_selftest_environment_probe.py` |
| `weights_logic` | `tests/_selftest_weights_logic.py` |
| `storage` | `tests/_selftest_storage_inventory.py` |
| `partial` | `tests/_selftest_partial_lifecycle.py` |
| `h3` | `tests/_selftest_h3_adapter.py` |
| `real_catalog` | `tests/_selftest_real_header_catalog.py` |
| `weights_review` | `tests/_selftest_registry_review.py` |

旧名字 `_weights_review.py` 已不存在；runner 先找实际
`_selftest_registry_review.py`，找不到时才用
`_selftest_*review*.py` / `_weights_review.py` glob 兼容旧 worktree。

P1（启动/HTTP/executor/边界）：

| label | 实际文件 |
|---|---|
| `empty_startup` | `tests/_selftest_empty_startup.py` |
| `control_http` | `tests/_selftest_control_http.py` |
| `operation_admission` | `tests/_selftest_operation_admission.py` |
| `execution_boundary` | `tests/_selftest_execution_boundary.py` |
| `check_call_sites` | `tests/_check_call_sites.py` |

## 状态与退出码

每个脚本独立串行执行；stdout/stderr 合并，统一 UTF-8 解码，并带单独
timeout。子进程环境设置 `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`、
`PYTHONDONTWRITEBYTECODE=1`，且 `PYTHONPATH` 前置仓库根目录。

| 状态 | 含义 |
|---|---|
| `passed` | 子进程退出码 0 |
| `failed` | 子进程退出码非 0 |
| `timeout` | 超过 `--timeout`，子进程被终止 |
| `error` | runner 无法启动脚本/解释器，或子进程层异常 |
| `not-covered` | 当前 worktree 没有该文件（未来/未集成）；**明确 skip，不计为通过** |

| 总退出码 | 含义 |
|---|---|
| `0` | 所有计划脚本都存在并 passed |
| `1` | 至少一个 failed/timeout/error |
| `2` | 无硬失败，但存在 not-covered（整体 `partial` 或 `not-covered`） |
| `3` | runner 配置/内部/报告写入错误 |

因此“缺未来 phase”不会被静默当成成功：P0 在当前 executor worktree 尚
未集成时会全部显示 `[SKIP] not-covered`，并以退出码 2 结束；P1 只跑可见
脚本，缺失项同样列出。`--list` 仅做可见性检查，不代表测试通过。

## JSON 报告

只有 `--report` 会写文件，内容包含：

```json
{
  "schema_version": 1,
  "phase": "p1",
  "python": "...",
  "timeout_seconds": 180.0,
  "root": "...",
  "tests_dir": "...",
  "generated_at": "2026-...",
  "overall": "partial",
  "exit_code": 2,
  "summary": {
    "total": 5, "covered": 2, "passed": 2, "failed": 0,
    "timeout": 0, "error": 0, "not-covered": 3,
    "overall": "partial", "exit_code": 2
  },
  "tests": [
    {
      "phase": "p1",
      "label": "operation_admission",
      "expected": "_selftest_operation_admission.py",
      "resolution": "exact",
      "resolved": "tests/_selftest_operation_admission.py",
      "status": "passed",
      "reason": null,
      "returncode": 0,
      "duration_s": 3.2,
      "output_tail": "...",
      "output_chars": 1234,
      "output_truncated": false
    }
  ]
}
```

`output_tail` 只保留最后 12000 字符，避免报告无限膨胀；`output_truncated`
标明是否截断。

## runner 自测

```powershell
python tests/_selftest_test_runner.py
```

自测覆盖 phase 列表、`weights_review` 的 exact/glob 解析、UTF-8、失败/超时/
缺失状态、退出码语义、显式报告路径，以及在本 worktree 实际运行可见 P1
子集并确认缺失项是 `not-covered`（不是隐性成功）。自测不需要 torch，也不
会复制或执行其他 worktree 的文件。
