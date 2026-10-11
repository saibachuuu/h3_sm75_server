# 回归自检

全部需要 torch 解释器；本地可用 `~/comfyui/.venv/Scripts/python.exe`：

```
cd h3_sm75_server/tests
python _check_call_sites.py          # runtime.py self.X 调用点一致性（秒级）
python _selftest_star7_port.py       # vendored 包 import 链 / backend 候选 / CLI 映射 / 探针契约
python _selftest_star7_cpu.py        # tiling/selection 纯 CPU 数学 / PackedLayout 解析
python _selftest_bridge.py           # bridge 编排顺序与守卫（GPU 相关已打桩）
python _selftest_bundle_compat.py    # bundle 兼容层（合成 bundle，含 fp8 反量化数值）
```

改 `Runtime` 类内代码后必跑 `_check_call_sites.py`；改 `veda/bundle_compat.py` 或 vendored veda 后必跑其余四个。

这些脚本不能替代远端 smoke：SM75 kernel 的数值与性能只能在 2080 Ti 上验。
