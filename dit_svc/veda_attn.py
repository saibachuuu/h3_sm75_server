"""veda warmup/bench 的执行框架（与 attention 实现无关的"定时+校验"部分）。

路线 B 变更（2026-10-11）：fork 专属逻辑（build_veda_override /
install_veda_probe / gate_snapshot / VEDA_PROBE）已全部移除：
  * bridge 现在在 dit_svc/veda_star7.py；
  * override 的就地安装与统计在 star7 的 VedaPatch（见
    h3_sm75_server/veda/comfy_patch.py）；
  * 探针统计走 dit_svc/veda_attn_star7.snapshot()。

保留的 run_single_warmup / bench_attention 由 /api/warmup 使用。
**口径警告**：POST /api/warmup 以裸调（无 transformer_options）执行，
Veda 的 gate 必然判 dense，所以 ms 数是 dense 回退耗时，不是稀疏加速比；
稀疏是否真跑请看 /api/status.veda_calls。
"""

# warmup 分级：每级 = (名, q_tokens, 说明)。负载逐级放大，默认只跑 probe。
WARMUP_TIERS = {
    "probe": (256, "最小 probe（与现有 sage/int8 probe 同量级）"),
    "tier1": (2048, "短序列（文本段量级）"),
    "tier2": (8192, "中序列（短视频段量级）"),
    "tier3": (24500, "全尺寸（124 帧 packed ~24.5k，sage 崩溃量级——仅手动指定才跑）"),
}
WARMUP_ORDER = ["probe", "tier1", "tier2", "tier3"]


def check_nvml_sane(torch):
    """NVML 健康检查：sage 事故判据——全零/断流 = 驱动已病。返回 (ok, 说明)。"""
    try:
        free, total = torch.cuda.mem_get_info()
        if free == 0 or total == 0:
            return False, "NVML 返回全零（free=%d total=%d），驱动可能已病，中止" % (free, total)
        return True, "free=%.2fGB total=%.2fGB" % (free / 1e9, total / 1e9)
    except Exception as e:  # noqa: BLE001
        return False, "NVML 调用失败（断流？）: %s" % e


def run_single_warmup(torch, device, attn_fn, tokens, log=print, repeat=1):
    """单发 warmup：指定 token 数跑 repeat 次。返回 dict（每轮 ms/finite/amax + 显存）。

    给 POST /api/warmup 用（队列内串行执行，不重启）。失败抛 RuntimeError。
    上限 tokens 40000（超过 packed 上限太多无意义，且防手滑炸卡），
    repeat 上限 50（防长时间占卡；真任务 4 步约等于 repeat=4×56 层）。
    QKV 只分配一次、循环复用（allocator 热态，贴近真任务步间状态，轮间无停顿）；
    empty_cache 只放最后，NVML 每轮查。
    """
    import math
    if not isinstance(tokens, int) or not (16 <= tokens <= 40000):
        raise ValueError("tokens 须为 16–40000 的整数，当前 %r" % (tokens,))
    if not isinstance(repeat, int) or not (1 <= repeat <= 50):
        raise ValueError("repeat 须为 1–50 的整数，当前 %r" % (repeat,))
    heads, dim = 56, 128  # H3 真实形态
    ok, vram0 = check_nvml_sane(torch)
    if not ok:
        raise RuntimeError("NVML 异常，中止：%s" % vram0)
    log("[warmup once] n=%d x%d | %s" % (tokens, repeat, vram0))
    # 一次分配、循环复用：首轮建图后 allocator 全热，后续轮无分配停顿
    parts = []
    for s in range(0, tokens, 4096):
        m = min(4096, tokens - s)
        parts.append(torch.randn(1, heads, m, dim, device=device, dtype=torch.float16))
    q = torch.cat(parts, dim=2)
    del parts
    v = q.clone()
    out_probe = attn_fn(q, q, v, heads)  # 建图轮（不计时，排除编译抖动）
    del out_probe
    torch.cuda.synchronize()
    rounds = []
    for i in range(repeat):
        ok, vram = check_nvml_sane(torch)
        if not ok:
            raise RuntimeError("[warmup %d/%d] NVML 异常，中止：%s" % (i + 1, repeat, vram))
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out = attn_fn(q, q, v, heads)
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1)
        finite = bool(torch.isfinite(out.float()).all())
        amax = float(out.float().abs().max())
        ok2, vram2 = check_nvml_sane(torch)
        del out
        r = {"ms": round(ms, 1), "finite": finite,
             "amax": amax if math.isfinite(amax) else None}
        log("[warmup %d/%d] %s | %s" % (i + 1, repeat, r, vram2))
        if not finite or r["amax"] is None:
            raise RuntimeError("[warmup %d/%d] 输出非 finite: %s" % (i + 1, repeat, r))
        if not ok2:
            raise RuntimeError("[warmup %d/%d] 跑后 NVML 异常：%s" % (i + 1, repeat, vram2))
        rounds.append(r)
    del q, v
    torch.cuda.empty_cache()
    result = {"tokens": tokens, "repeat": repeat, "rounds": rounds,
              "ms_min": min(r["ms"] for r in rounds),
              "ms_max": max(r["ms"] for r in rounds),
              "vram_before": vram0, "vram_after": vram2}
    log("[warmup once] done %s" % result)
    return result


def bench_attention(torch, device, veda_call, dense_call, tokens, repeat=3,
                    profile=False, log=print):
    """同一对 Q/K/V 上对比两组 attention 的耗时（可选 profiler 明细）。

    veda_call / dense_call：fn(q, k, v) -> out，q/k/v 为 (1, heads, tokens, dim) fp16。
    每档先跑一轮建图（不计时），再跑 repeat 轮取 min/max；每轮校验输出 finite。
    profile=True 时额外用 torch.profiler 抓 CUDA kernel 明细（按 cuda_time_total 排序），
    这是把"到底谁吃时间"从推测变成数字的唯一手段。

    **口径警告**：若两组调用都没有 transformer_options，则 veda 一侧测的是
    dense 回退，ratio≈1 是必然（不是 bug，是 gate 行为）。
    """
    import time as _time
    heads, dim = 56, 128  # H3 真实形态
    parts = []
    for s in range(0, tokens, 4096):
        m = min(4096, tokens - s)
        parts.append(torch.randn(1, heads, m, dim, device=device, dtype=torch.float16))
    q = torch.cat(parts, dim=2)
    del parts
    k = q.clone()
    v = q.clone()

    def _run(fn):
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out = fn(q, k, v)
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1)
        finite = bool(torch.isfinite(out.float()).all())
        del out
        return round(ms, 2), finite

    result = {"tokens": tokens, "heads": heads, "dim": dim, "repeat": repeat}
    measured = {}
    for name, fn in (("veda", veda_call), ("int8", dense_call)):
        try:
            _run(fn)                       # 建图轮：allocator/内核都热起来，不计时
        except Exception as e:
            result[name] = {"error": "%s: %s" % (type(e).__name__, e)}
            log("[bench] %s 预热失败: %s" % (name, e))
            continue
        rounds = []
        for _ in range(max(1, repeat)):
            ms, finite = _run(fn)
            if not finite:
                raise RuntimeError("%s 注意力输出含 NaN/Inf（n=%d）" % (name, tokens))
            rounds.append(ms)
            log("[bench %s] n=%d round %d: %.1f ms" % (name, tokens, len(rounds), ms))
        measured[name] = rounds
        result[name] = {"rounds": rounds, "ms_min": min(rounds), "ms_max": max(rounds),
                        "ms_avg": round(sum(rounds) / len(rounds), 2)}
    if "veda" in measured and "int8" in measured:
        result["ratio_veda_over_int8"] = round(
            result["veda"]["ms_min"] / max(result["int8"]["ms_min"], 1e-9), 2)
    if profile:
        for name, fn in (("veda", veda_call), ("int8", dense_call)):
            if name not in measured:
                continue
            t0 = _time.perf_counter()
            try:
                from torch.profiler import ProfilerActivity, profile as _profile
                with _profile(activities=[ProfilerActivity.CUDA]) as prof:
                    fn(q, k, v)
                    torch.cuda.synchronize()
                table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
            except Exception as e:
                table = "profiler 失败: %s: %s" % (type(e).__name__, e)
            result[name + "_profile"] = table
            log("[bench %s] profile (%.1fs):\n%s" % (name, _time.perf_counter() - t0, table))
    del q, k, v
    torch.cuda.empty_cache()
    log("[bench] 汇总: veda=%s int8=%s ratio=%s"
        % (result.get("veda", {}).get("ms_min"), result.get("int8", {}).get("ms_min"),
           result.get("ratio_veda_over_int8")))
    return result


def run_tiered_warmup(torch, device, attn_fn, impl_name, level, log=print):
    """分级 warmup：从 probe 逐级跑到 level，每级同步 + 水位检查 + NVML 检查。

    attn_fn(q, k, v, heads) 为待测 attention（含 skip_reshape=True 调用形态）。
    任一级失败/水位异常即抛 RuntimeError，调用方回退。level='off' 直接返回。
    注意：无 transformer_options 的调用只验证 dense 回退通路与 allocator 热态。
    """
    import math
    if level == "off":
        log("[warmup] 跳过（--veda-warmup off）")
        return
    if level not in WARMUP_TIERS:
        raise ValueError("未知 warmup 级别: %s（可选 off/probe/tier1/tier2/tier3）" % level)
    target = WARMUP_ORDER.index(level)
    heads, dim = 56, 128  # H3 真实形态
    for name in WARMUP_ORDER[:target + 1]:
        n, desc = WARMUP_TIERS[name]
        ok, vram = check_nvml_sane(torch)
        if not ok:
            raise RuntimeError("[warmup %s] NVML 异常，中止：%s" % (name, vram))
        log("[warmup %s] n=%d（%s）| %s" % (name, n, desc, vram))
        # 分块构造，避免一次性大分配误伤：每 4k tokens 一块拼起来
        parts = []
        for s in range(0, n, 4096):
            m = min(4096, n - s)
            parts.append(torch.randn(1, heads, m, dim, device=device, dtype=torch.float16))
        q = torch.cat(parts, dim=2)
        del parts
        k = q
        v = q.clone()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out = attn_fn(q, k, v, heads)
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1)
        finite = bool(torch.isfinite(out.float()).all())
        amax = float(out.float().abs().max())
        ok, vram2 = check_nvml_sane(torch)
        log("[warmup %s] %.1fms finite=%s amax=%.4g | %s"
            % (name, ms, finite, amax, vram2))
        del q, k, v, out
        torch.cuda.empty_cache()
        if not finite or not math.isfinite(amax):
            raise RuntimeError("[warmup %s] 输出非 finite，中止" % name)
        if not ok:
            raise RuntimeError("[warmup %s] 跑后 NVML 异常，中止：%s" % (name, vram2))
    log("[warmup] 到 %s 全过（dense 回退口径）" % level)
