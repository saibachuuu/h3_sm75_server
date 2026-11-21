# 统一 operation executor 与原子 admission（P1 底座）

本文描述 `dit_svc/job_queue.py` + `dit_svc/operations.py` 的**已实现接口**。
它只负责执行器/回执/互斥，不包含 weights、attention、infer 的具体业务；
主代理在 P2+ 通过 runtime 可选 hook 接入。

## 1. 边界

- 只有一个 worker（`QueueManager._worker`）。旧 GPU 任务没有另开线程，也没有
  HTTP 线程直接跑 GPU 的旁路。
- 旧 `Job`（`video / warmup / vae_probe / director`）与新 `Operation` 共用
  同一个 worker、同一把 `threading.Condition`（admission lock）。
- `Job.public()` 字段、`snapshot()` 形状、旧提交端点行为保持不变。
- 新长操作返回 `operation_id` 回执，客户端轮询
  `GET /api/operations/<id>`，不根据 HTTP 连接是否断开判断成功/失败。
- 所有结构仅标准库；`operations.py` 不 import torch，队列模块在真正执行旧 job
  时才延迟 import `pipeline/vae_probe/director`。

## 2. admission 规则（原子）

所有检查都在同一个 `self._cond` 内完成：

| 请求 | 空闲 | 旧 job 运行/排队 | 其他 control 运行/排队 |
|---|---|---|---|
| 旧 job 提交 | 接受 | 接受并继续排队 | 409 `BUSY`（`BusyError`） |
| `submit_operation` | 接受并占用唯一保留位 | 409 `BUSY` | 409 `BUSY` |

- 两个并发 control 最多一个获准；另一个在锁内看到 active operation 后拒绝。
- 旧 job 运行期间提交旧 job 仍可排队（不因“队列非空”拒绝）。
- control operation 运行/排队期间旧 job 一律 409。
- 旧 job 的 readiness 检查仍在锁外（`Runtime.ready/error` + 可选
  `require_job` hook）；因此模型未 ready 不妨碍 weights control operation。
- 入队与检查之间没有“先看队列为空再入队”的窗口。

## 3. 公共接口

```python
queue.submit_operation(kind, payload, run_callback,
                       preflight=None, request_id=None) -> Operation
queue.get_operation(operation_id) -> Operation | None
queue.operations_snapshot(limit=20) -> dict
queue.cancel_operation(operation_id) -> dict          # public view
queue.executor_state() -> dict                        # status 端点摘要
queue.stop(timeout=5.0) -> bool                       # shutdown/测试
queue.shutdown(timeout=5.0) -> bool                   # stop 别名
```

旧的 `submit / submit_cond / submit_director / submit_vae_probe / submit_warmup /
drop / clear_queued / get / snapshot` 全部保留。

### 3.1 提交与回执

`submit_operation` 成功返回 `Operation` 对象：

```python
op.receipt()
# {"ok": true, "operation_id": "op_<uuid>", "status": "queued",
#  "operation_url": "/api/operations/op_<uuid>"}
```

`Operation.public()`（GET/快照使用）：

```json
{
  "id": "op_<uuid>",
  "kind": "weights.disk2ram",
  "status": "queued|running|done|failed|cancelled",
  "stage": "排队中",
  "progress": null,
  "created": 0.0,
  "started": null,
  "finished": null,
  "elapsed": 0.0,
  "request_id": "client-1",
  "result": null,
  "error": null,
  "operation_url": "/api/operations/op_<uuid>"
}
```

- `public()` **不包含** payload、callback、bytes 或 tensor；快照可以安全返回。
- `public(include_log=True)` 额外给最近 40 行日志，只建议单条 GET 使用。
- `result` 是 `run_callback` 的返回值，必须能被 HTTP JSON 序列化。
- callback 可以用 `op.set_stage("...", progress=0.5)` 更新进度；不要获取
  admission lock，也不要在 callback 内调用 `queue.stop()`。
- callback 抛异常不会杀死 worker：operation 变 `failed`，队列继续接受/执行
  下一个任务。普通异常归类为 `EXECUTION_FAILED/500`；`OperationError`
  原样保留 code/http_status/retryable/details。

### 3.2 查询快照

```python
queue.operations_snapshot(limit=20)
# {"operations": [...最新在前...],
#  "counts": {"queued":0,"running":0,"done":0,"failed":0,"cancelled":0},
#  "running": "op_..." | None}
```

- 只读内存 metadata，不等待运行中的 callback，也不触发任何 GPU 同步。
- `snapshot()`（旧 jobs）与 `operations_snapshot()` 互不影响；取消 operation
  不会误改旧 job。
- `executor_state()` 返回
  `{"state":"idle|queued|running","active_operation","active_job","queued",
    "queued_operations","queued_jobs","control_reserved"}`，供 `/api/status` 使用。

### 3.3 取消

- 仅 `queued` 可取消 → `status=cancelled`，返回 public dict。
- 对已取消 operation 再取消是幂等的（仍返回 cancelled）。
- `running / done / failed` → 409 `OPERATION_NOT_CANCELLABLE`；首版不承诺打断
  GPU kernel。
- 不存在 → 404 `OPERATION_NOT_FOUND`。
- 取消后保留位立即释放，可以提交下一个 control。

### 3.4 request_id 去重

- 只保证**当前进程内**去重；重启后客户端要按 operation/job 状态重新规划。
- 相同 `request_id` + 相同 `kind` + 规范化 payload → 返回**同一个**
  `Operation`（同 operation_id），不会重复执行。
- 相同 `request_id` + 不同请求 → 409 `REQUEST_ID_CONFLICT`，details 带已有
  operation_id。payload 的 dict key 顺序不影响指纹；bytes 按 sha256 参与指纹。
- 重复请求在 preflight 之前就返回原回执，不会重复做耗时扫描。
- 去重索引有界，与 operation 保留一致（默认 200 条）：只淘汰最老的 terminal
  operation，绝不丢 queued/running；被淘汰的 request_id 再次出现视为新请求。

## 4. preflight / 执行前重验

提交阶段：

1. 锁外调用 `preflight(kind, payload)`（显式参数优先）；返回 token（任意对象，
   仅传给 revalidate hook）。
2. 锁内查 request_id、检查 control/job 占用、调用可选
   `runtime.operation_admission(kind, payload)`、创建 operation。

执行阶段（worker 内、锁外）：

1. **再次调用同一个 preflight**（执行前重验；文件/状态可能已变化）。
2. 调用可选 `runtime.operation_revalidate(kind, payload, token)`。
3. 最后才调用 `run_callback(op)`。

preflight 建议抛 `OperationError`；抛 `ValueError` 会归类为
`INVALID_REQUEST/400`，其他异常归类为 `EXECUTION_FAILED/500`。
执行阶段失败只把 operation 标 `failed`，不会回滚已接受的回执。

## 5. runtime 可选 hooks（主代理接入点）

尚未接入 weights 模块时，以下属性全部不存在即可，行为与旧版一致：

| hook | 调用时机 | 锁 | 要求 |
|---|---|---|---|
| `runtime.require_job(payload)` | 旧 job 校验后、入队前 | 锁外 | 可慢；抛 `OperationError` 会转成旧 `BusyError`/`ValueError` |
| `runtime.operation_preflight(kind, payload)` | 提交时 + 执行前 | 锁外 | 可慢；建议幂等纯检查 |
| `runtime.operation_admission(kind, payload)` | admission 锁内 | **持锁** | 必须廉价；不得 GPU/文件扫描，不得回调 QueueManager，否则死锁 |
| `runtime.operation_revalidate(kind, payload, token)` | 执行前 preflight 之后 | 锁外 | 复核 state revision/recipe fingerprint 等 |

旧 job 的 readiness 仍是原来的 `Runtime.ready/error` 检查；`require_job`
只作为增量 hook，用来做 weights 相关前置检查。control operation 不要求
`Runtime.ready`，只由 `operation_admission` 决定（P1 门槛：ready=false 不妨碍
weights API）。

## 6. 错误与旧接口兼容

`dit_svc.operations.OperationError`：

```python
OperationError(code, message, http_status=500, retryable=False, details=None)
# .to_dict() -> {"code","message","http_status","retryable","details"}
```

`dit_svc.job_queue.BusyError` 现在继承 `OperationError`，默认
`code="BUSY", http_status=409, retryable=True`；`str(BusyError("x")) == "x"`，
旧 HTTP handler 的 `except BusyError` 与中文错误字符串保持不变。
新 HTTP 层可统一用 `err.to_dict()` 生成 error envelope；执行阶段失败时
operation GET 仍返回 200，`status=failed`，`error` 是同一个结构。

## 7. shutdown

`stop()` 会：

- 设置停止标志并通知 worker；
- 把仍 `queued` 的 operation/job 标为 `cancelled`（stage=`服务关闭`）；
- 之后的新提交返回 `SHUTTING_DOWN/503`（`submit_operation` 抛
  `OperationError`，旧 job 抛 `BusyError`）；
- 等待 worker 退出（默认 5 秒，返回是否已退出）。

不强行杀死正在运行的 callback；服务进程退出时由 daemon 线程结束。
测试中应始终在 `finally` 调用 `stop()`，避免残留线程。

## 8. 测试

```powershell
& C:\Users\88888888\AppData\Local\Temp\opencode\explicit_api_env_20261121\Scripts\python.exe tests\_selftest_operation_admission.py
```

当前覆盖：回执/public view、preflight 两次、两个 control 并发、control/job
多轮 barrier、job 运行中继续排队、control 占用时 job 409、queued/running
取消、request_id 去重/冲突/有界淘汰、callback 异常不封锁、提交/执行 preflight
失败、runtime hooks、admission 拒绝、require_job、快照不阻塞、客户端断线后
服务继续、shutdown 干净退出。测试不 import torch。

## 9. 已知限制（P1）

- 只在进程内去重，无持久化；operation 不跨重启恢复。
- 不提供 running operation 的强制中断；取消只针对 queued。
- `operations_snapshot` 是内存 metadata 快照，不是执行进度事件流。
- 去重指纹基于规范化 JSON + bytes sha256；未知对象用 `type+repr`，仅保证同一
  进程内稳定性。
- 具体 operation kind/payload schema、HTTP 路由和 error→HTTP 映射由主代理在
  P2/P5 接入；本底座只提供 `OperationError` 的 code/http_status。
