"""真实 H3 Comfy adapter（P0 构造证据落地，P2 选择性绑定）。

Comfy 第三方模块全部经构造参数注入（torch / ops / MiniMaxH3Model /
model_detection），本文件 import 时不碰任何 Comfy 包，远端 probe 传真模块，
本机单测传 stub。绝不复用 stock 不完整 sd 路径：
每个 unit 按子模块 `load_state_dict(strict=True, assign=True)` 精确绑定，
缺 key 直接抛（关掉 strict=False 留随机参数的门）。

已核实快照依据（只读，未改）：
- detect 形状集：comfy/model_detection.py:401-433
- stock 全量 eager：comfy/sd.py:2431-2524；strict=False：model_base.py:381
- 量化绑定：comfy/ops.py:_load_quantized_module:1256-1397（comfy_quant 描述符
  + per-format scales；缺 weight 只 warning -> 本 adapter 用 strict=True 关门）
- 构造 device 透传：minimax/model.py:476-518；blocks ModuleList 索引稳定
- adaln_basis/mean 在 minimax/model.py 中无消费点（仅 adaln_t_table 被用），
  本 adapter 显式 waive（见 WAIVED_KEYS），不静默丢弃。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..weights import UnitCatalog, WeightsError, audit_unit

from .fake import AdapterError, ModelAdapter, SkeletonNoStorage


# ---------------------------------------------------------------------------
# 常量：版本门 + waive 清单
# ---------------------------------------------------------------------------

QUANT_META_KEY = "_quantization_metadata"
QUANT_FORMAT_VERSION = "1.0"

#: 有真实 tensor 但模型前向不消费、stock loader 只会 warn-unexpected 的 keys。
#: 显式 waive（绑定报告中记录），不是静默丢弃；模型类变更时重审。
WAIVED_KEYS = {
    "adaln_basis": "curve basis host-side only; MiniMaxH3Model consumes only adaln_t_table",
    "adaln_mean": "curve basis host-side only; MiniMaxH3Model consumes only adaln_t_table",
}

#: H3 推理必需的绑定 keys（waived 除外全要齐；由 catalog 派生，禁止手写）。
REQUIRED_TOP_KEYS = ("adaln_t_table", "rope.inv_freq")


class ComfyUnavailable(WeightsError):
    code = "COMFY_UNAVAILABLE"


def _need(name: str, value: Any) -> Any:
    if value is None:
        raise ComfyUnavailable(
            "real H3 adapter needs injected %r (pass real comfy module in probe, stub in test)" % (name,),
            recovery=["远端 probe 传真 comfy 模块；本机单测传 stub"],
            details={"missing": name})
    return value


# ---------------------------------------------------------------------------
# header-only detect 形状代理（不读权重值）
# ---------------------------------------------------------------------------


class _ShapeProxy:
    __slots__ = ("shape",)

    def __init__(self, shape: Sequence[int]):
        self.shape = tuple(shape)


class ShapeProxySD(Mapping):
    """只读 header 形状的 sd 代理：支持 detect 用的 keys/in/[]/.shape。"""

    def __init__(self, shapes: Mapping[str, Sequence[int]]):
        self._shapes = dict(shapes)

    def __getitem__(self, key: str) -> _ShapeProxy:
        return _ShapeProxy(self._shapes[key])

    def __iter__(self):  # type: ignore[override]
        return iter(self._shapes)

    def __len__(self) -> int:
        return len(self._shapes)

    def keys(self):  # type: ignore[override]
        return self._shapes.keys()


# ---------------------------------------------------------------------------
# 真实 H3 adapter
# ---------------------------------------------------------------------------


class H3ComfyAdapter(ModelAdapter):
    family = "dit-h3"

    def __init__(self, *, torch_mod: Any = None, ops_mod: Any = None,
                 model_cls: Any = None, detection_mod: Any = None):
        self.torch = torch_mod
        self.ops = ops_mod
        self.model_cls = model_cls
        self.detection = detection_mod

    # -- 版本/quant 配置 --
    def quant_config_from_metadata(self, metadata: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """从 checkpoint header metadata 建 quant_config。

        只接受 format_version == "1.0"；layers 为 {layer: {format, ...}}。
        返回 (quant_config, report)。未知版本直接拒绝，不猜。
        """
        raw = metadata.get(QUANT_META_KEY, None) if metadata else None
        if raw is None:
            return {}, {"quantized": False, "reason": "no _quantization_metadata"}
        import json as _json
        try:
            parsed = _json.loads(raw) if isinstance(raw, str) else dict(raw)
        except Exception as e:
            raise AdapterError("bad _quantization_metadata: %s" % (e,)) from None
        ver = parsed.get("format_version", None)
        if ver != QUANT_FORMAT_VERSION:
            raise AdapterError(
                "unsupported quantization format_version %r (want %r)" % (ver, QUANT_FORMAT_VERSION),
                recovery=["等主代理确认新格式后再扩展 adapter，不静默按旧格式绑"],
                details={"format_version": ver})
        layers = parsed.get("layers", {})
        if not isinstance(layers, dict):
            raise AdapterError("quant layers is not a dict")
        formats = sorted(set(v.get("format") for v in layers.values()
                             if isinstance(v, dict)))
        return (dict(layers), {"quantized": True, "format_version": ver,
                               "num_layers": len(layers), "formats": formats})

    def detect_unet_config(self, shapes: Mapping[str, Sequence[int]],
                           metadata: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """header-only detect：只给形状，不给权重值。"""
        detection = _need("model_detection", self.detection)
        proxy = ShapeProxySD(shapes)
        try:
            cfg = detection.detect_unet_config(proxy, "", metadata)
        except Exception as e:
            raise AdapterError("detect_unet_config failed: %s: %s"
                               % (type(e).__name__, e)) from None
        if not cfg or cfg.get("image_model") != "minimax_h3":
            raise AdapterError("header is not minimax_h3: %r" % (cfg,),
                               details={"config": cfg})
        return (dict(cfg), {"image_model": cfg.get("image_model"),
                            "num_layers": cfg.get("num_layers"),
                            "token_refiner_num_layers": cfg.get("token_refiner_num_layers"),
                            "hidden_size": cfg.get("hidden_size")})

    # -- 骨架 --
    def build_skeleton(self, catalog: UnitCatalog,
                       unet_config: Optional[Mapping[str, Any]] = None,
                       quant_config: Optional[Mapping[str, Any]] = None,
                       compute_dtype: Any = None,
                       device: str = "meta") -> Any:
        """无存储骨架：device="meta" 零真实 storage；device 仅 skeleton 阶段用。

        注意 MiniMaxH3Model 构造即定 factory device；选择性物化时 materialize
        会把选中子模块 to_empty 到目标设备并修正 factory_kwargs（见下）。
        """
        torch_mod = _need("torch", self.torch)
        ops_mod = _need("comfy.ops", self.ops)
        model_cls = _need("MiniMaxH3Model", self.model_cls)
        if unet_config is None:
            raise AdapterError("build_skeleton needs detected unet_config (header-only)")
        dtype = compute_dtype if compute_dtype is not None else torch_mod.bfloat16
        if quant_config:
            operations = ops_mod.mixed_precision_ops(dict(quant_config), dtype)
        else:
            operations = ops_mod.disable_weight_init
        cfg = dict(unet_config)
        try:
            # 真实 MiniMaxH3Model 内部有无 device 分配（adaln_t_table /
            # rope.inv_freq 用 torch.empty(dtype=float32)），构造参数 device
            # 覆盖不到它们；用 torch.device 上下文兜底，保证骨架零真实 storage。
            with torch_mod.device(device):
                model = model_cls(**cfg, device=device, operations=operations,
                                  dtype=dtype)
        except Exception as e:
            raise AdapterError("skeleton construct failed: %s: %s"
                               % (type(e).__name__, e),
                               details={"device": device}) from None
        for _m in model.modules():
            if hasattr(_m, "requires_grad_"):
                try:
                    _m.requires_grad_(False)
                except Exception:
                    pass
        model.eval()
        return model

    # -- unit -> 模型持有人路径 --
    def unit_holders(self, unit_id: str, model_id: str) -> List[Tuple[str, str]]:
        """返回 [(holder_path, kind)]；holder_path 为模型下点路径（""=模型自身）。

        kind: "module"（子模块整体绑定）/ "buffer"（holder 上的单个 buffer）。
        """
        if unit_id == "%s.refiner" % model_id:
            return [("token_refiner", "module"), ("condition_proj", "module")]
        if unit_id.startswith("%s.blocks." % model_id):
            idx = unit_id.rsplit(".", 1)[1]
            return [("blocks.%s" % idx, "module")]
        if unit_id.startswith("%s.modules." % model_id):
            leaf = unit_id.rsplit(".", 1)[1]
            if leaf == "adaln_t_table":
                return [("", "buffer:adaln_t_table")]
            return [(leaf, "module")]
        if unit_id == "%s.root" % model_id:
            return [("", "module-rootscalars")]
        raise AdapterError("H3 adapter has no holder mapping for %r" % (unit_id,),
                           details={"unit": unit_id})

    def _resolve(self, model: Any, path: str) -> Any:
        cur = model
        if not path:
            return cur
        for part in path.split("."):
            if part.isdigit() and hasattr(cur, "__getitem__"):
                try:
                    cur = cur[int(part)]
                    continue
                except Exception:
                    pass
            cur = getattr(cur, part)
        return cur

    def _sync_factory_kwargs(self, holder: Any, device: str) -> int:
        """递归同步 holder 子树全部 factory_kwargs['device']。

        真实 MixedPrecision Linear 构造期 device 固定为 meta（骨架阶段），
        而 _load_quantized_module 按此重建权重；to_empty 不碰该元数据，
        不显式同步则量化权重回绑到 meta（远端 probe 20 实测 4 个 meta）。
        只动 holder 子树，不碰未选 unit。返回同步个数（审计用）。
        """
        n = 0
        seen = set()
        for m in holder.modules():
            if id(m) in seen:
                continue
            seen.add(id(m))
            fk = getattr(m, "factory_kwargs", None)
            if isinstance(fk, dict) and fk.get("device") != device:
                fk["device"] = device
                n += 1
        return n

    def _stale_holders(self, model: Any,
                       holders: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """返回子树仍有 meta storage 的 holders（重绑候选）。"""
        out = []
        for holder_path, kind in holders:
            if kind.startswith("buffer:") or kind == "module-rootscalars":
                continue
            if self._meta_leaves(self._resolve(model, holder_path)):
                out.append((holder_path, kind))
        return out

    def _meta_leaves(self, holder: Any) -> List[str]:
        """holder 子树仍带真实 meta storage 的 params/buffers 路径。"""
        out: List[str] = []
        for name, p in holder.named_parameters(recurse=True):
            if p is not None and getattr(p, "is_meta", False):
                out.append("param:" + name)
        for name, b in holder.named_buffers(recurse=True):
            if b is not None and getattr(b, "is_meta", False):
                out.append("buffer:" + name)
        return out

    def _bind_holder(self, holder: Any, holder_path: str, kind: str,
                     key_tensors: Mapping[str, Any],
                     device: str) -> Tuple[list, list, dict, int]:
        """单个 holder：to_empty + factory_kwargs 同步 + strict 绑定。"""
        torch_mod = _need("torch", self.torch)
        bound: List[str] = []
        waived: List[str] = []
        quant_formats: Dict[str, str] = {}
        if kind.startswith("buffer:"):
            name = kind.split(":", 1)[1]
            full = name
            if full not in key_tensors:
                raise AdapterError("missing buffer key %r" % (full,),
                                   details={"key": full})
            holder.register_buffer(
                name, torch_mod.as_tensor(key_tensors[full]).to(device))
            bound.append(full)
            return bound, waived, quant_formats, 0
        if kind == "module-rootscalars":
            raise AdapterError("root unit has no module home; expand catalog first")
        prefix = holder_path + "."
        sub = {k: v for k, v in key_tensors.items() if k == holder_path
               or k.startswith(prefix)}
        if holder_path == "token_refiner":
            sub = {k: v for k, v in key_tensors.items()
                   if k.startswith("token_refiner.")}
        for w in WAIVED_KEYS:
            if w in sub:
                sub.pop(w)
                waived.append(w)
        try:
            holder.to_empty(device=device)
        except Exception as e:
            raise AdapterError("to_empty(%s) failed: %s" % (device, e),
                               details={"holder": holder_path}) from None
        synced = self._sync_factory_kwargs(holder, device)
        rel = {}
        for k, v in sub.items():
            if k == holder_path:
                rel[k] = v
            elif k.startswith(prefix):
                rel[k[len(prefix):]] = v
            else:
                raise AdapterError("key %r not under holder %r"
                                   % (k, holder_path))
        try:
            missing, unexpected = holder.load_state_dict(rel, strict=True,
                                                         assign=True)
        except RuntimeError as e:
            raise AdapterError("strict bind failed for %r: %s"
                               % (holder_path, e),
                               details={"holder": holder_path,
                                        "error": str(e)[:300]}) from None
        assert not missing and not unexpected, (missing, unexpected)
        bound.extend(k for k in sub.keys())
        for k, v in sub.items():
            if k.endswith("comfy_quant"):
                quant_formats[k] = self._read_quant_format(v)
        return bound, waived, quant_formats, synced

    # -- 选择性物化：只读所选 unit keys，未选不动 --
    def materialize_unit(self, model: Any, unit_id: str,
                         key_tensors: Mapping[str, Any],
                         device: str = "cpu",
                         expected_keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """把该 unit 的 checkpoint keys 精确绑定进骨架（strict=True）。

        key_tensors: {checkpoint全key: tensor}。任何未被任一 holder 消费、
        也未被显式 waive 的 key 直接拒绝（防跨 unit 多读静默忽略）。
        expected_keys: 调用方按 catalog 传入的该 unit 全 key 集；传入时做
        集合精确校验（多/少都拒）。未选 unit 的模块保持 meta/None。
        waived-only 请求（全 key 被 waive、无 holder 可绑）返回 noncompute
        报告，不触模型、不读权重体、不计驻留。
        返回绑定报告（含 waived、per-layer quant format 日志、
        factory_kwargs 同步数、meta 重绑记录）。
        """
        _need("torch", self.torch)
        model_id = unit_id.split(".")[0]
        waived_pre = sorted(k for k in key_tensors
                            if k.split(".")[-1] in WAIVED_KEYS or k in WAIVED_KEYS)
        if waived_pre and len(waived_pre) == len(key_tensors):
            return {"unit": unit_id, "bound_keys": [], "waived": waived_pre,
                    "quant_formats": {}, "device": device, "noncompute": True}
        if expected_keys is not None:
            exp, got = set(expected_keys), set(key_tensors.keys())
            if exp != got:
                raise AdapterError(
                    "key set mismatch for %r: missing=%r extra=%r"
                    % (unit_id, sorted(exp - got), sorted(got - exp)),
                    details={"unit": unit_id, "missing": sorted(exp - got),
                             "extra": sorted(got - exp)})
        bound: List[str] = []
        waived: List[str] = list(waived_pre)
        quant_formats: Dict[str, str] = {}
        synced_total = 0
        repair_log: List[Dict[str, Any]] = []
        holders = self.unit_holders(unit_id, model_id)
        # 预检：每个 key 必须被某 holder 消费或被显式 waive；
        # 跨 unit 多余 key 在此拒绝，不等绑定污染模型。
        def _covered(k: str) -> bool:
            if k.split(".")[-1] in WAIVED_KEYS or k in WAIVED_KEYS:
                return True
            for hp, kind in holders:
                if kind.startswith("buffer:"):
                    if k == kind.split(":", 1)[1]:
                        return True
                elif k == hp or k.startswith(hp + "."):
                    return True
            return False
        uncovered = sorted(k for k in key_tensors.keys() if not _covered(k))
        if uncovered:
            raise AdapterError(
                "unconsumed keys for %r (cross-unit extra?): %r"
                % (unit_id, uncovered),
                recovery=["只传该 unit 的 keys；跨 unit key 必须走各自 unit 绑定"],
                details={"unit": unit_id, "unconsumed": uncovered})
        for holder_path, kind in holders:
            holder = self._resolve(model, holder_path)
            # holder 消费的 keys：holder_path 点转 checkpoint 前缀
            prefix = holder_path.replace(".", ".") + "."
            # 顶层模块名与 checkpoint 顶层名一致（blocks.N / token_refiner /
            # condition_proj / video_patch_proj / ...），rope 除外：
            sub = {k: v for k, v in key_tensors.items() if k == holder_path
                   or k.startswith(prefix)}
            if kind == "module" and holder_path in ("token_refiner",):
                # token_refiner holder 消费 token_refiner.* 全体（refiner unit 含
                # condition_proj 时由另一 holder 条目处理，此处只取本前缀）
                sub = {k: v for k, v in key_tensors.items()
                       if k.startswith("token_refiner.")}
            b, w, q, n_synced = self._bind_holder(holder, holder_path, kind,
                                                 dict(sub), device)
            bound.extend(b)
            waived.extend(x for x in w if x not in waived)
            quant_formats.update(q)
            synced_total += n_synced
        # 绑定后扫 meta：量化子层若仍按陈旧 factory_kwargs 重建会落 meta；
        # 发现即重同步后定向重绑（只动本 unit holders），仍有残留则硬失败。
        if device != "meta":
            stale = self._stale_holders(model, holders)
            if stale:
                repaired = []
                for holder_path, kind in stale:
                    holder = self._resolve(model, holder_path)
                    n = self._sync_factory_kwargs(holder, device)
                    prefix = holder_path + "."
                    sub = {k: v for k, v in key_tensors.items()
                           if k == holder_path or k.startswith(prefix)}
                    if kind == "module" and holder_path in ("token_refiner",):
                        sub = {k: v for k, v in key_tensors.items()
                               if k.startswith("token_refiner.")}
                    b, w, q, n2 = self._bind_holder(holder, holder_path, kind,
                                                   dict(sub), device)
                    repaired.append({"holder": holder_path,
                                     "resynced": n, "rebound": len(b)})
                stale = self._stale_holders(model, holders)
                if stale:
                    raise AdapterError(
                        "meta leaves remain after rebind for %r: %r"
                        % (unit_id, stale),
                        recovery=["检查该 holder 子树 factory_kwargs 同步逻辑"],
                        details={"unit": unit_id, "stale": stale,
                                 "repaired": repaired})
                repair_log.extend(repaired)
        consumed = set(bound) | set(waived)
        unconsumed = sorted(k for k in key_tensors.keys() if k not in consumed)
        if unconsumed:
            raise AdapterError(
                "unconsumed keys for %r (cross-unit extra?): %r"
                % (unit_id, unconsumed),
                recovery=["只传该 unit 的 keys；跨 unit key 必须走各自 unit 绑定"],
                details={"unit": unit_id, "unconsumed": unconsumed})
        return {"unit": unit_id, "bound_keys": sorted(bound),
                "waived": sorted(waived), "quant_formats": quant_formats,
                "device": device, "factory_kwargs_synced": synced_total,
                "meta_repair": repair_log}

    @staticmethod
    def _read_quant_format(descriptor_tensor: Any) -> str:
        try:
            import json as _json
            raw = descriptor_tensor.numpy().tobytes()
            return str(_json.loads(raw).get("format", "unknown"))
        except Exception as e:  # noqa: BLE001
            return "unreadable:%s" % type(e).__name__

    # -- purge：换 meta，不删槽位 --
    def purge_unit(self, model: Any, unit_id: str) -> Dict[str, Any]:
        """释放该 unit 全部真实 storage（params/buffers 置 meta 空壳）。

        不删除 ModuleList 索引、不碰未选 unit。同步把 holder 子树
        factory_kwargs['device'] 置回 meta（与当前 meta 空壳状态一致，
        防下次绑定沿用陈旧 cpu/cuda 重建）。返回释放报告。
        """
        torch_mod = _need("torch", self.torch)
        model_id = unit_id.split(".")[0]
        touched: List[str] = []
        for holder_path, kind in self.unit_holders(unit_id, model_id):
            if kind.startswith("buffer:"):
                name = kind.split(":", 1)[1]
                holder = self._resolve(model, holder_path)
                old = holder.get_buffer(name)
                holder.register_buffer(
                    name, torch_mod.empty(old.shape, device="meta", dtype=old.dtype))
                touched.append(holder_path or "model" + ":" + name)
                continue
            holder = self._resolve(model, holder_path)
            with torch_mod.no_grad():
                for _name, p in list(holder.named_parameters(recurse=False)):
                    if p is not None and not getattr(p, "is_meta", False):
                        holder._parameters[_name] = torch_mod.nn.Parameter(
                            torch_mod.empty(p.shape, device="meta",
                                            dtype=p.dtype),
                            requires_grad=False)
                for _name, b in list(holder.named_buffers(recurse=False)):
                    if b is not None and not getattr(b, "is_meta", False):
                        holder.register_buffer(
                            _name, torch_mod.empty(b.shape, device="meta", dtype=b.dtype))
                # 递归子模块（quant Linear 的 weight=None 保持 None）
                for _m in holder.modules():
                    if _m is holder:
                        continue
                    for _name, p in list(_m.named_parameters(recurse=False)):
                        if p is not None and not getattr(p, "is_meta", False):
                            _m._parameters[_name] = torch_mod.nn.Parameter(
                                torch_mod.empty(p.shape, device="meta",
                                                dtype=p.dtype),
                                requires_grad=False)
                    for _name, b in list(_m.named_buffers(recurse=False)):
                        if b is not None and not getattr(b, "is_meta", False):
                            _m.register_buffer(
                                _name, torch_mod.empty(b.shape, device="meta", dtype=b.dtype))
            touched.append(holder_path)
        for _m in [holder] + [m for m in holder.modules()]:
            _fk = getattr(_m, "factory_kwargs", None)
            if isinstance(_fk, dict):
                _fk["device"] = "meta"
        return {"unit": unit_id, "purged_holders": touched}

    # -- move：整 unit 子树 .to(device)，随后由 inventory 核对 --
    def move_unit(self, model: Any, unit_id: str, device: str) -> Dict[str, Any]:
        """把该 unit 子树整体搬到 device（数据搬运，非权重管理绕过）。

        不做跨 unit 事务回滚（plan 02.6）；调用方按 unit 逐个 move，
        每 unit 后用 inventory 校验，OOM 即停并 mark_error。
        """
        _need("torch", self.torch)
        model_id = unit_id.split(".")[0]
        touched: List[str] = []
        for holder_path, kind in self.unit_holders(unit_id, model_id):
            if kind.startswith("buffer:"):
                name = kind.split(":", 1)[1]
                holder = self._resolve(model, holder_path)
                buf = holder.get_buffer(name)
                holder.register_buffer(name, buf.to(device))
                touched.append((holder_path or "model") + ":" + name)
                continue
            holder = self._resolve(model, holder_path)
            holder.to(device)
            self._sync_factory_kwargs(holder, device)
            touched.append(holder_path)
        return {"unit": unit_id, "device": device, "moved_holders": touched}

    # -- inventory 证据：unit 子树真实张量 --
    def resident_tensors(self, model: Any, unit_id: str) -> Dict[str, Any]:
        """返回该 unit 子树的真实张量容器，供 storage_inventory 测量。"""
        model_id = unit_id.split(".")[0]
        out: Dict[str, Any] = {}
        for holder_path, kind in self.unit_holders(unit_id, model_id):
            if kind.startswith("buffer:"):
                name = kind.split(":", 1)[1]
                holder = self._resolve(model, holder_path)
                out[(holder_path or "model") + "." + name] = holder.get_buffer(name)
                continue
            out[holder_path] = self._resolve(model, holder_path)
        return out

    def audit_after_bind(self, catalog: UnitCatalog, model: Any,
                         unit_id: str, bind_report: Mapping[str, Any],
                         expect_loaded: bool = True) -> Dict[str, Any]:
        """绑定后交叉自证：catalog × 真实绑定记录 × 物理 inventory。"""
        from ..weights import audit_unit, storage_inventory
        if bind_report.get("noncompute"):
            # waived-only：无 holder、无 storage、不在 catalog units；
            # 其“永不驻留”由 catalog 分类（noncompute_keys）+ probe 只读
            # catalog keys 保证，此处只确认报告自洽。
            verdict = "ok" if not expect_loaded else "NONCOMPUTE_LOADED"
            return {"unit": unit_id, "noncompute": True,
                    "reason": "waived-only keys; no holder, never materialized",
                    "waived": list(bind_report.get("waived", [])),
                    "expect_loaded": expect_loaded, "verdict": verdict}
        bound = list(bind_report.get("bound_keys", []))
        waived = list(bind_report.get("waived", []))
        inv = storage_inventory(self.resident_tensors(model, unit_id),
                                owner=unit_id)
        return audit_unit(catalog, unit_id, bound, inv,
                          expect_loaded=expect_loaded, waived=waived)