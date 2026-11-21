"""P0/P2 权重底座：recipe / catalog / selector / state / storage inventory。

只依赖标准库即可 import（torch 按需延迟导入，见 storage_inventory）。
真实 H3 checkpoint 与远端 ComfyUI 版本尚未提供，本模块的 H3 命名规则为
「待真实 header 验证的暂行规则」，详见 docs/p0_weights.md。

与主代理接口约定（本文件只做底座实现，不做 HTTP/queue/runtime 改动）：
  WeightRegistry.snapshot()      -> 可 JSON 序列化全量摘要（无 Tensor/bytes）
  WeightRegistry.model_state()   -> 单模型摘要（含 unit 明细与缺失数）
  WeightRegistry.resolve_units() -> mode/units 选择器展开（all 只是展开）
  apply_disk2ram / apply_ram2vram / apply_vram2ram / apply_purgefromram
                                 -> 返回可 JSON 序列化操作摘要
错误类型为本 stage 局部类型；主代理后续统一映射到公共 OperationError。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 局部错误类型（主代理后续映射到公共 OperationError）
# ---------------------------------------------------------------------------


class WeightsError(Exception):
    """权重底座所有局部错误的基类。code 供主代理映射。"""

    code = "WEIGHTS_ERROR"

    def __init__(self, message: str, *, recovery: Optional[List[str]] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.recovery = list(recovery or [])
        self.details = dict(details or {})


class RecipeError(WeightsError):
    code = "RECIPE_ERROR"


class CatalogError(WeightsError):
    code = "CATALOG_ERROR"


class UnmappedKeyError(CatalogError):
    code = "UNMAPPED_KEY"


class SelectorError(WeightsError):
    code = "UNKNOWN_SELECTOR"


class TransitionError(WeightsError):
    code = "ILLEGAL_TRANSITION"


class BudgetError(WeightsError):
    code = "RESOURCE_EXHAUSTED"


class StateError(WeightsError):
    code = "STATE_ERROR"


# ---------------------------------------------------------------------------
# safetensors header（标准库解析，不读权重体）
# ---------------------------------------------------------------------------

_DTYPE_NBYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1,
    "I32": 4, "U32": 4, "F32": 4,
    "F64": 8, "I64": 8, "U64": 8,
}

# 非计算元数据白名单：header 顶层 "__metadata__" 不是张量，跳过不计账。
HEADER_META_KEY = "__metadata__"

# 已知可忽略的非计算 key 前缀（可解释白名单；命中则不建账但记录）。
IGNORED_KEY_PREFIXES = ("__",)


def dtype_nbytes(dtype: str) -> int:
    try:
        return _DTYPE_NBYTES[dtype]
    except KeyError:
        raise CatalogError("unknown dtype %r" % (dtype,),
                           details={"dtype": dtype}) from None


def tensor_key_bytes(dtype: str, shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * dtype_nbytes(dtype)


def read_safetensors_header(path: str) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """只读 safetensors header，返回 (tensors, header_len)。

    tensors: {key: {"dtype": str, "shape": [...], "data_offsets": [b, e]}}。
    顶层 "__metadata__" 被剥离（见返回值第二元组之外的元信息不返回）。
    不读取任何权重体；枚举 catalog 不需要 safetensors/torch 包。
    """
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise CatalogError("file too small for safetensors header",
                               details={"path": path})
        (header_len,) = struct.unpack("<Q", raw)
        if header_len > 256 * 1024 * 1024:
            raise CatalogError("safetensors header implausibly large",
                               details={"path": path, "header_len": header_len})
        blob = f.read(header_len)
        if len(blob) < header_len:
            raise CatalogError("truncated safetensors header",
                               details={"path": path})
    try:
        header = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise CatalogError("safetensors header is not valid JSON: %s" % e,
                           details={"path": path}) from None
    if not isinstance(header, dict):
        raise CatalogError("safetensors header is not an object",
                           details={"path": path})
    tensors: Dict[str, Dict[str, Any]] = {}
    for k, v in header.items():
        if k == HEADER_META_KEY:
            continue
        if not isinstance(v, dict) or "dtype" not in v or "shape" not in v:
            raise CatalogError("malformed tensor entry %r" % (k,),
                               details={"key": k})
        tensors[k] = {"dtype": v["dtype"], "shape": list(v["shape"]),
                      "data_offsets": list(v.get("data_offsets", [0, 0]))}
    return tensors, header_len


def header_digest(path: str, header_len: int) -> str:
    """header 原始字节的 sha256（变更检测用，不冒充全文件哈希）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.read(8)
        remaining = header_len
        while remaining > 0:
            chunk = f.read(min(65536, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# ModelRecipe：知道从哪读，不意味着已读
# ---------------------------------------------------------------------------


@dataclass
class ModelRecipe:
    model_id: str
    kind: str  # dit | te | vae.video | vae.audio | veda.bundle ...
    path: str  # 主权重文件（绝对路径）
    extra_paths: Tuple[str, ...] = ()
    options: Dict[str, Any] = field(default_factory=dict)
    fingerprint: Dict[str, Any] = field(default_factory=dict)
    adapter_version: str = "weights-v0"

    def summary(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "kind": self.kind,
            "path": self.path,
            "extra_paths": list(self.extra_paths),
            "options": dict(self.options),
            "fingerprint": dict(self.fingerprint),
            "adapter_version": self.adapter_version,
        }

    def same_source(self, other: "ModelRecipe") -> bool:
        return (self.path == other.path
                and tuple(self.extra_paths) == tuple(other.extra_paths)
                and self.fingerprint == other.fingerprint)


def fingerprint_of(path: str) -> Dict[str, Any]:
    """文件指纹：规范化路径 + 大小 + mtime_ns + header 摘要（变更检测）。"""
    ap = os.path.abspath(path)
    st = os.stat(ap)
    tensors, header_len = read_safetensors_header(ap)
    return {
        "path": ap,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "header_len": header_len,
        "header_sha256": header_digest(ap, header_len),
        "num_tensors": len(tensors),
    }


def make_recipe(model_id: str, kind: str, path: str,
                extra_paths: Sequence[str] = (),
                options: Optional[Dict[str, Any]] = None) -> ModelRecipe:
    ap = os.path.abspath(path)
    fps = fingerprint_of(ap)
    extra_fps = []
    for p in extra_paths:
        extra_fps.append(fingerprint_of(os.path.abspath(p)))
    fp = dict(fps)
    if extra_fps:
        fp["extras"] = extra_fps
    return ModelRecipe(model_id=model_id, kind=kind, path=ap,
                       extra_paths=tuple(os.path.abspath(p) for p in extra_paths),
                       options=dict(options or {}), fingerprint=fp)


def recipe_from_header_dict(model_id: str, kind: str, path_label: str,
                            tensors: Mapping[str, Mapping[str, Any]]) -> ModelRecipe:
    """无真实文件时（如单测 fake header）构造轻量 recipe（指纹标 fake）。"""
    canon = json.dumps({k: {"dtype": v["dtype"], "shape": list(v["shape"])}
                        for k, v in sorted(tensors.items())},
                       sort_keys=True).encode("utf-8")
    fp = {"path": path_label, "size": -1, "mtime_ns": -1,
          "header_sha256": "fake-" + hashlib.sha256(canon).hexdigest()[:16],
          "num_tensors": len(tensors), "fake": True}
    return ModelRecipe(model_id=model_id, kind=kind, path=path_label,
                       fingerprint=fp)


# ---------------------------------------------------------------------------
# UnitCatalog：完整、稳定、与驻留状态无关
# ---------------------------------------------------------------------------

VALID_UNIT_STATES = ("absent", "ram", "vram", "mixed", "error")


@dataclass
class UnitInfo:
    id: str
    checkpoint_keys: Tuple[str, ...]
    expected_bytes: int
    groups: Tuple[str, ...] = ()
    factory: str = "rule-generic-v0"
    capabilities: Dict[str, Any] = field(default_factory=dict)
    shared_group: Optional[str] = None  # 同 storage 所有权组（不可分拆时同名）

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "num_keys": len(self.checkpoint_keys),
            "expected_bytes": self.expected_bytes,
            "groups": list(self.groups),
            "factory": self.factory,
            "capabilities": dict(self.capabilities),
            "shared_group": self.shared_group,
        }


class UnitCatalog:
    """完整 catalog：覆盖 header 全部带权 key（白名单忽略除外）。"""

    def __init__(self, model_id: str, units: Sequence[UnitInfo],
                 ignored_keys: Sequence[str] = ()):
        self.model_id = model_id
        self.units: Dict[str, UnitInfo] = {}
        self.key_to_unit: Dict[str, str] = {}
        self.groups: Dict[str, List[str]] = {}
        for u in units:
            if u.id in self.units:
                raise CatalogError("duplicate unit id %r" % (u.id,))
            self.units[u.id] = u
            for k in u.checkpoint_keys:
                if k in self.key_to_unit:
                    raise CatalogError(
                        "key %r owned by both %r and %r" % (
                            k, self.key_to_unit[k], u.id),
                        details={"key": k})
                self.key_to_unit[k] = u.id
            for g in u.groups:
                self.groups.setdefault(g, []).append(u.id)
        self.ignored_keys = tuple(ignored_keys)

    def unit_ids(self) -> List[str]:
        return list(self.units.keys())

    def keys_of(self, unit_ids: Iterable[str]) -> List[str]:
        out: List[str] = []
        for uid in unit_ids:
            out.extend(self.units[uid].checkpoint_keys)
        return out

    def expected_bytes_of(self, unit_ids: Iterable[str]) -> int:
        return sum(self.units[uid].expected_bytes for uid in unit_ids)

    def check_cover(self, header_keys: Iterable[str]) -> None:
        """全量覆盖校验：未识别 key 直接失败（不允许 leftover 后继续）。"""
        missing = [k for k in header_keys
                   if k not in self.key_to_unit
                   and not any(k.startswith(p) for p in IGNORED_KEY_PREFIXES)]
        if missing:
            raise UnmappedKeyError(
                "%d unmapped checkpoint keys (e.g. %r)" % (len(missing), missing[:5]),
                recovery=["补齐该 model family 的 key->unit 命名规则后重建 catalog"],
                details={"missing_sample": missing[:20], "missing_count": len(missing)})

    def summary(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "units": [self.units[uid].summary() for uid in self.unit_ids()],
            "groups": {g: list(v) for g, v in self.groups.items()},
            "ignored_keys": list(self.ignored_keys),
        }


_BLOCK_RE = re.compile(r"(?:^|[._])(?:blocks?|transformer_blocks?)[._-]?(\d+)(?:[._]|$)")
_LAYER_RE = re.compile(r"(?:^|[._])(?:layers?|predictor)[._-]?(\d+)(?:[._]|$)")


def _is_ignored(key: str) -> bool:
    return any(key.startswith(p) for p in IGNORED_KEY_PREFIXES)


def build_catalog_from_header(model_id: str,
                              header: Mapping[str, Mapping[str, Any]],
                              family: str = "dit") -> UnitCatalog:
    """从 header 全量建 catalog（暂行命名规则，未经真实 H3 验证）。

    family: dit | te | vae | veda。规则只看 key 字符串，不读任何权重体。
    同一 storage 的共享归属（shared_group）在真实布局核实前保守地按 unit
    独立建账，并在 docs/p0_weights.md 登记为 unknown。
    """
    buckets: Dict[str, List[str]] = {}
    unit_groups: Dict[str, Tuple[str, ...]] = {}
    ignored: List[str] = []
    for key in header.keys():
        if _is_ignored(key):
            ignored.append(key)
            continue
        uid = _assign_unit(model_id, key, family)
        buckets.setdefault(uid, []).append(key)
    if family == "dit" and not buckets:
        raise CatalogError("empty header for family dit")
    units: List[UnitInfo] = []
    for uid in sorted(buckets.keys()):
        keys = tuple(sorted(buckets[uid]))
        total = 0
        for k in keys:
            total += tensor_key_bytes(header[k]["dtype"], header[k]["shape"])
        groups = _groups_for(model_id, uid, family)
        units.append(UnitInfo(id=uid, checkpoint_keys=keys, expected_bytes=total,
                              groups=groups,
                              factory="rule-%s-v0-UNVERIFIED" % family,
                              capabilities=_capabilities_for(uid, family)))
    cat = UnitCatalog(model_id, units, ignored)
    cat.check_cover(list(header.keys()))
    return cat


# H3 detect 所需的 header-only 形状描述键（Comfy 5f415e31a
# model_detection.py:401-433 实测）。detect 只读这些 key 的存在性与
# .shape，不读权重值；header 枚举（get_shape）即可满足，无需物化张量。
# count_blocks 另需 "blocks.{N}." / "token_refiner.blocks.{N}." 前缀枚举。
H3_DESCRIPTOR_KEYS = (
    "video_patch_proj.weight",
    "audio_patch_proj.weight",
    "final_layer.video_out.weight",
    "final_layer.audio_out.weight",
    "blocks.0.attn.q_norm.weight",
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.mlp.fc1.weight",
    "condition_proj.weight",
    "rope.inv_freq",
    # 二选一：adaln_t_table（曲线基）或 time_embedder.proj_in/out
    "adaln_t_table",
    "time_embedder.proj_in.weight",
    "time_embedder.proj_out.weight",
)

#  stock loader 铁证（同快照）：sd.py:2517 get_model 后
#  model_base.py:381 load_state_dict(..., strict=False)。
#  部分 sd 喂给原函数会留下随机初始化参数并仅打 warning ——
#  因此 partial disk2ram 绝不能复用原函数喂不完整 sd，必须走
#  adapter 的 build_skeleton + materialize_unit 路径。


def _assign_unit(model_id: str, key: str, family: str) -> str:
    """暂行命名规则。dit 部分已按 Comfy 5f415e31a 快照的真实 H3 布局校准
    （model_detection.py:401-433），但仍需真实 checkpoint header 复核；
    详情见 docs/p0_weights.md，未复核前 factory 保持 -UNVERIFIED 后缀。
    """
    lk = key.lower()
    if family == "dit":
        m = _BLOCK_RE.match(lk) or _BLOCK_RE.search(lk)
        # 真实 H3 block 形如 blocks.{N}.attn.qkv_proj / mlp.fc1，
        # refiner 形如 token_refiner.blocks.{N}.*（不得与主 blocks 混淆）。
        if lk.startswith("token_refiner.") or lk.startswith("condition_proj"):
            return "%s.refiner" % model_id
        if m and not lk.startswith("token_refiner."):
            # _BLOCK_RE 也会命中 token_refiner.blocks.N，先排除后取主 blocks
            if ".blocks." in lk and lk.split(".blocks.")[0].endswith("token_refiner"):
                return "%s.refiner" % model_id
            return "%s.blocks.%d" % (model_id, int(m.group(1)))
        top = key.split(".")[0]
        if top in ("video_patch_proj", "audio_patch_proj",
                   "time_embedder", "final_layer", "rope", "adaln_t_table"):
            return "%s.modules.%s" % (model_id, top)
        if top in ("", "_root"):
            return "%s.root" % model_id
        return "%s.modules.%s" % (model_id, top)
    if family == "te":
        if any(s in lk for s in ("proj", "clip_g", "text_projection", "mmproj")):
            return "%s.proj" % model_id
        if any(s in lk for s in ("vision", "visual", "image_tower")):
            return "%s.vision" % model_id
        return "%s.model" % model_id
    if family == "vae":
        if "encoder" in lk:
            return "%s.encoder" % model_id
        if "decoder" in lk:
            return "%s.decoder" % model_id
        if "shared" in lk or "quant" in lk:
            return "%s.shared" % model_id
        # 无法判断 encoder/decoder 独立性时收敛到 shared，不虚构可分 unit。
        return "%s.shared" % model_id
    if family == "veda":
        m = _LAYER_RE.search(lk)
        if m:
            return "%s.layers.%d" % (model_id, int(m.group(1)))
        return "%s.layers.0" % model_id
    raise CatalogError("unknown family %r" % (family,), details={"family": family})


def _groups_for(model_id: str, uid: str, family: str) -> Tuple[str, ...]:
    if family == "dit":
        if ".blocks." in uid:
            return (model_id, "%s.blocks" % model_id)
        if uid.endswith(".refiner"):
            return (model_id,)
        return (model_id,)
    if family == "te":
        return (model_id,)
    if family == "vae":
        return (model_id,)
    if family == "veda":
        return (model_id, "%s.bundle" % model_id)
    return (model_id,)


def _capabilities_for(uid: str, family: str) -> Dict[str, Any]:
    if family == "dit":
        return {"sample_needs": True, "partial_load_supported": True,
                "movable": True, "purgeable": True}
    if family == "veda":
        return {"sparse_needs": True, "partial_load_supported": True,
                "movable": True, "purgeable": True}
    return {"partial_load_supported": True, "movable": True, "purgeable": True}


# ---------------------------------------------------------------------------
# 选择器：稳定 unit id / 组 / 闭区间范围；不支持 glob
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"^(?P<prefix>.+?)[._-](?P<lo>\d+)-(?P<hi>\d+)$")


def resolve_units(catalog: UnitCatalog, mode: str,
                  units: Sequence[str] = ()) -> List[str]:
    """展开选择器。mode="all" 忽略 units 并返回全部 unit（稳定 catalog 序）。

    支持：精确 unit id；组别名（catalog.groups 的 key，如 dit.blocks）；
    闭区间范围（如 dit.blocks.12-15，按尾部数字展开）。
    不支持 glob / 正则 / 空选择。返回去重后 catalog 序列表。
    """
    order = catalog.unit_ids()
    if mode == "all":
        return list(order)
    if mode != "units":
        raise SelectorError("unknown mode %r (want all|units)" % (mode,),
                            details={"mode": mode})
    if not units:
        raise SelectorError("empty units with mode=units",
                            recovery=["mode=all 或给出显式 unit 选择器"])
    picked: List[str] = []

    def _expand_numeric_range(sel: str) -> Optional[List[str]]:
        m = _RANGE_RE.match(sel)
        if not m:
            return None
        prefix, lo, hi = m.group("prefix"), int(m.group("lo")), int(m.group("hi"))
        if hi < lo:
            raise SelectorError("reversed range %r" % (sel,), details={"sel": sel})
        cands = ["%s.%d" % (prefix, i) for i in range(lo, hi + 1)]
        known = [c for c in cands if c in catalog.units]
        if not known:
            raise SelectorError("range %r matches no unit" % (sel,),
                                details={"sel": sel})
        if len(known) != len(cands):
            missing = [c for c in cands if c not in catalog.units]
            raise SelectorError("range %r partly missing: %r" % (sel, missing[:5]),
                                details={"sel": sel, "missing": missing})
        return known

    for sel in units:
        if sel in catalog.units:
            picked.append(sel)
        elif sel in catalog.groups:
            picked.extend(catalog.groups[sel])
        else:
            ranged = _expand_numeric_range(sel)
            if ranged is not None:
                picked.extend(ranged)
            else:
                raise SelectorError(
                    "unknown selector %r" % (sel,),
                    recovery=["用 GET catalog 查看合法 unit id / 组名"],
                    details={"sel": sel, "hint": "no glob; use id, group or N-M range"})
    # 去重并恢复 catalog 稳定序
    seen = set()
    deduped = []
    for u in picked:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    rank = {uid: i for i, uid in enumerate(order)}
    return sorted(deduped, key=lambda u: rank[u])


# ---------------------------------------------------------------------------
# 真实 storage inventory：views 去重 / 量化内部张量 / buffers / dict 投影
# ---------------------------------------------------------------------------


def _torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    return torch


def _storage_id_of(t: Any) -> Optional[Tuple[str, int, int]]:
    """(device_type, data_ptr, storage_nbytes)。meta 张量返回 None。"""
    try:
        dev = str(t.device)
    except Exception:
        return None
    if dev == "meta":
        return None
    try:
        st = t.untyped_storage()
    except Exception:
        return None
    try:
        return (dev.split(":")[0], int(st.data_ptr()), int(st.nbytes()))
    except Exception:
        return None


def _walk_tensors(root: Any, _seen: Optional[set] = None,
                  _depth: int = 0, _path: str = "root"):
    """通用张量 walk：Tensor / dict / list+tuple / nn.Module / 量化包装。

    产出 (path, tensor) 二元组，path 记录对象图中的属性/key 路径，
    供 inventory owners 使用。torch 可选：无 torch 时只识别
    dict/list 中鸭子类型张量（有 untyped_storage 方法的对象）。
    """
    if _seen is None:
        _seen = set()
    if _depth > 12 or root is None:
        return
    oid = id(root)
    if oid in _seen:
        return
    _seen.add(oid)
    torch_mod = _torch()
    is_tensor = False
    if torch_mod is not None and isinstance(root, torch_mod.Tensor):
        is_tensor = True
    elif hasattr(root, "untyped_storage") and callable(root.untyped_storage):
        try:
            root.untyped_storage()
            is_tensor = True
        except Exception:
            is_tensor = False
    if is_tensor:
        yield (_path, root)
        # 视图链：base tensor 可能挂在 _base_attr 上？标准 Tensor 不需要；
        # 仍尝试展开 _base 以防封装类型。
        base = getattr(root, "_base", None)
        if base is not None and id(base) not in _seen and _depth < 12:
            yield from _walk_tensors(base, _seen, _depth + 1, _path + "._base")
        return
    if isinstance(root, Mapping):
        for k, v in root.items():
            yield from _walk_tensors(v, _seen, _depth + 1, "%s.%s" % (_path, k))
        return
    if isinstance(root, (list, tuple)):
        for i, v in enumerate(root):
            yield from _walk_tensors(v, _seen, _depth + 1, "%s[%d]" % (_path, i))
        return
    torch_nn = getattr(torch_mod, "nn", None) if torch_mod else None
    if torch_nn is not None and isinstance(root, torch_nn.Module):
        try:
            params = list(root.named_parameters(recurse=True))
        except Exception:
            params = []
        try:
            bufs = list(root.named_buffers(recurse=True))
        except Exception:
            bufs = []
        for name, p in params:
            yield from _walk_tensors(p, _seen, _depth + 1,
                                     "%s.param.%s" % (_path, name))
        for name, b in bufs:
            if b is not None:
                yield from _walk_tensors(b, _seen, _depth + 1,
                                         "%s.buf.%s" % (_path, name))
        return
    # 量化包装 / dict 投影对象：走 __dict__（防循环用 _seen）。
    d = getattr(root, "__dict__", None)
    if isinstance(d, dict) and _depth < 12:
        for k, v in d.items():
            if k.startswith("__"):
                continue
            yield from _walk_tensors(v, _seen, _depth + 1, "%s.%s" % (_path, k))


@dataclass
class StorageInventory:
    entries: List[Dict[str, Any]] = field(default_factory=list)
    total_cpu: int = 0
    total_cuda: int = 0
    meta_tensors: int = 0
    num_tensors_seen: int = 0

    @property
    def num_storages(self) -> int:
        return len(self.entries)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "num_storages": self.num_storages,
            "num_tensors_seen": self.num_tensors_seen,
            "meta_tensors": self.meta_tensors,
            "total_cpu": self.total_cpu,
            "total_cuda": self.total_cuda,
            "shared": sum(1 for e in self.entries if len(e["owners"]) > 1),
            "storages": [e["storage_id"] for e in self.entries],
            "entries": self.entries,
        }


def storage_inventory(root: Any, owner: str = "") -> StorageInventory:
    """真实 storage 清单：按 (device, data_ptr) 去重，meta 计零字节。

    同一 storage 被多个 tensor（视图/别名/量化内部复用）引用只计一次，
    owners 列出全部引用路径。无 torch 时退化为空清单（meta_tables=0），
    调用方据此判定本机不可做物理账。
    """
    inv = StorageInventory()
    seen_storage: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for path, t in _walk_tensors(root):
        inv.num_tensors_seen += 1
        sid = _storage_id_of(t)
        if sid is None:
            inv.meta_tensors += 1
            continue
        dev, ptr, nbytes = sid
        key = (dev, ptr)
        # 所有者路径：owner + 张量在对象图中的属性路径。
        label = "%s/%s" % (owner or "root", path)
        entry = seen_storage.get(key)
        if entry is None:
            entry = {"storage_id": "%s:%d:%d" % (dev, ptr, nbytes),
                     "device": dev, "nbytes": nbytes, "owners": []}
            seen_storage[key] = entry
            inv.entries.append(entry)
            if dev == "cuda":
                inv.total_cuda += nbytes
            else:
                inv.total_cpu += nbytes
        entry["owners"].append(label)
    return inv


# ---------------------------------------------------------------------------
# WeightRegistry：recipe/catalog/unit 状态/generation + 四动作编排
# ---------------------------------------------------------------------------


def _new_unit_record(info: UnitInfo) -> Dict[str, Any]:
    return {
        "id": info.id,
        "state": "absent",
        "transition": None,
        "device": None,
        "generation": 0,
        "expected_bytes": info.expected_bytes,
        "resident_bytes_cpu": 0,
        "resident_bytes_cuda": 0,
        "storage_digest": [],
        "last_error": None,
        "recovery_actions": [],
    }


class WeightRegistry:
    """单进程权重注册表：JSON 摘要快照，不持有真实 Tensor 强引用。"""

    def __init__(self):
        self.recipes: Dict[str, ModelRecipe] = {}
        self.catalogs: Dict[str, UnitCatalog] = {}
        self.units: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.model_generation: Dict[str, int] = {}

    # -- recipe / catalog --
    def register_recipe(self, recipe: ModelRecipe) -> None:
        mid = recipe.model_id
        cur = self.recipes.get(mid)
        if cur is not None and self._has_resident(mid) and not cur.same_source(recipe):
            raise RecipeError(
                "model %r has resident weights; purge all units before "
                "switching recipe" % (mid,),
                recovery=["purgefromram 该模型全部 unit 后再换 recipe"],
                details={"model": mid})
        self.recipes[mid] = recipe
        self.units.setdefault(mid, {})
        self.model_generation.setdefault(mid, 0)

    def attach_catalog(self, catalog: UnitCatalog) -> None:
        mid = catalog.model_id
        if mid not in self.recipes:
            raise RecipeError("no recipe for model %r" % (mid,),
                              details={"model": mid})
        self.catalogs[mid] = catalog
        store = self.units.setdefault(mid, {})
        for uid, info in catalog.units.items():
            if uid not in store:
                store[uid] = _new_unit_record(info)
            else:
                store[uid]["expected_bytes"] = info.expected_bytes

    # -- 查询 --
    def _require_model(self, model_id: str) -> UnitCatalog:
        try:
            return self.catalogs[model_id]
        except KeyError:
            raise StateError("unknown model %r (no catalog)" % (model_id,),
                             details={"model": model_id}) from None

    def _has_resident(self, model_id: str) -> bool:
        return any(r["state"] in ("ram", "vram", "mixed", "error")
                   for r in self.units.get(model_id, {}).values())

    def unit_record(self, model_id: str, unit_id: str) -> Dict[str, Any]:
        try:
            return self.units[model_id][unit_id]
        except KeyError:
            raise StateError("unknown unit %r of model %r" % (unit_id, model_id),
                             details={"model": model_id, "unit": unit_id}) from None

    def resolve_units(self, model_id: str, mode: str,
                      units: Sequence[str] = ()) -> List[str]:
        return resolve_units(self._require_model(model_id), mode, units)

    def model_state(self, model_id: str) -> Dict[str, Any]:
        cat = self._require_model(model_id)
        store = self.units.get(model_id, {})
        unit_list = []
        absent = ram = vram = mixed = error = 0
        for uid in cat.unit_ids():
            r = store.get(uid)
            if r is None:
                s = "absent"
                unit_list.append({"id": uid, "state": s, "device": None,
                                  "generation": 0, "resident_bytes_cpu": 0,
                                  "resident_bytes_cuda": 0,
                                  "expected_bytes": cat.units[uid].expected_bytes,
                                  "last_error": None, "recovery_actions": []})
                absent += 1
                continue
            unit_list.append({k: r[k] for k in (
                "id", "state", "device", "generation", "resident_bytes_cpu",
                "resident_bytes_cuda", "expected_bytes", "last_error",
                "recovery_actions")})
            if r["state"] == "absent":
                absent += 1
            elif r["state"] == "ram":
                ram += 1
            elif r["state"] == "vram":
                vram += 1
            elif r["state"] == "mixed":
                mixed += 1
            elif r["state"] == "error":
                error += 1
        if mixed or error:
            mstate = "error"
        elif absent and (ram or vram):
            mstate = "partial"
        elif not absent and ram and not vram:
            mstate = "ram"
        elif not absent and vram and not ram:
            mstate = "vram"
        elif not absent and ram and vram:
            mstate = "partial"
        else:
            mstate = "absent"
        return {
            "model_id": model_id,
            "state": mstate,
            "generation": self.model_generation.get(model_id, 0),
            "recipe": self.recipes[model_id].summary(),
            "absent_units": absent,
            "units": unit_list,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {"models": {mid: self.model_state(mid)
                           for mid in self.catalogs.keys()}}

    # -- 状态写入（由 adapter 物化/搬运/释放成功后调用） --
    def set_unit_loaded(self, model_id: str, unit_id: str, device: str,
                        resident_bytes_cpu: int = 0,
                        resident_bytes_cuda: int = 0,
                        storage_digest: Optional[List[str]] = None) -> None:
        r = self.unit_record(model_id, unit_id)
        if device == "cpu":
            r["state"] = "ram"
        elif device == "cuda:0":
            r["state"] = "vram"
        else:
            raise StateError("unknown device %r" % (device,),
                             details={"device": device})
        r["device"] = device
        r["transition"] = None
        r["resident_bytes_cpu"] = int(resident_bytes_cpu)
        r["resident_bytes_cuda"] = int(resident_bytes_cuda)
        r["storage_digest"] = list(storage_digest or [])
        r["last_error"] = None
        r["recovery_actions"] = []
        r["generation"] += 1
        self.model_generation[model_id] = self.model_generation.get(
            model_id, 0) + 1

    def set_unit_absent(self, model_id: str, unit_id: str) -> None:
        r = self.unit_record(model_id, unit_id)
        r["state"] = "absent"
        r["transition"] = None
        r["device"] = None
        r["resident_bytes_cpu"] = 0
        r["resident_bytes_cuda"] = 0
        r["storage_digest"] = []
        r["last_error"] = None
        r["recovery_actions"] = []
        r["generation"] += 1
        self.model_generation[model_id] = self.model_generation.get(
            model_id, 0) + 1

    def mark_error(self, model_id: str, unit_id: str, message: str,
                   recovery: Optional[List[str]] = None) -> None:
        r = self.unit_record(model_id, unit_id)
        r["state"] = "error"
        r["transition"] = None
        r["last_error"] = message
        r["recovery_actions"] = list(recovery or ["用 recover 修复后重试"])
        self.model_generation[model_id] = self.model_generation.get(
            model_id, 0) + 1

    def recover_unit(self, model_id: str, unit_id: str) -> Dict[str, Any]:
        """显式 recover：仅 error/mixed 且实际完全在 CPU 才可回 absent。

        仍有 CUDA 残留时先 recover→ram（调用方搬运后再次 recover），
        不能把恢复当成推理隐式 offload。
        """
        r = self.unit_record(model_id, unit_id)
        if r["state"] not in ("error", "mixed"):
            raise TransitionError("unit %r state %r needs no recover"
                                  % (unit_id, r["state"]),
                                  details={"unit": unit_id,
                                           "state": r["state"]})
        if r["resident_bytes_cuda"]:
            raise TransitionError(
                "unit %r still holds CUDA storage; vram2ram first" % (unit_id,),
                recovery=["vram2ram %s 后再次 recover" % unit_id],
                details={"unit": unit_id})
        self.set_unit_absent(model_id, unit_id)
        return {"model": model_id, "unit": unit_id, "state": "absent"}

    # -- 四动作迁移门卫 --
    def check_transition(self, action: str, model_id: str,
                         unit_ids: Sequence[str]) -> bool:
        """参数/状态/指纹冲突在修改前拒绝；合法 no-op 返回 True。"""
        cat = self._require_model(model_id)
        if model_id not in self.recipes:
            raise RecipeError("no recipe for model %r" % (model_id,))
        if action not in ("disk2ram", "ram2vram", "vram2ram", "purgefromram"):
            raise TransitionError("unknown action %r" % (action,),
                                  details={"action": action})
        store = self.units.get(model_id, {})
        if action == "disk2ram":
            # 非 absent 且 recipe 变化时已在 register_recipe 拒绝；
            # 同 recipe 已驻留 = no-op 成功。
            need = [u for u in unit_ids
                    if store.get(u, {}).get("state", "absent") == "absent"]
            blocked = [u for u in unit_ids
                       if store.get(u, {}).get("state", "absent")
                       in ("mixed", "error")]
            if blocked:
                raise TransitionError(
                    "disk2ram blocked by mixed/error units %r" % (blocked,),
                    recovery=["recover %s 后重试" % ", ".join(blocked)],
                    details={"blocked": blocked})
            return True
        if action == "ram2vram":
            bad = [u for u in unit_ids
                   if store.get(u, {}).get("state", "absent") != "ram"
                   and store.get(u, {}).get("state", "absent") != "vram"]
            if bad:
                raise TransitionError(
                    "ram2vram needs ram units; not-ready: %r" % (bad,),
                    recovery=["先 disk2ram %s" % ", ".join(bad)],
                    details={"not_ready": bad})
            return True
        if action == "vram2ram":
            bad = [u for u in unit_ids
                   if store.get(u, {}).get("state", "absent") not in
                   ("vram", "ram", "absent")]
            if bad:
                raise TransitionError(
                    "vram2ram blocked by mixed/error units %r" % (bad,),
                    recovery=["recover 后重试"],
                    details={"blocked": bad})
            return True
        if action == "purgefromram":
            cuda_hold = [u for u in unit_ids
                         if store.get(u, {}).get("resident_bytes_cuda", 0) > 0
                         or store.get(u, {}).get("device") == "cuda:0"]
            if cuda_hold:
                raise TransitionError(
                    "purgefromram refused: units still hold CUDA %r"
                    % (cuda_hold,),
                    recovery=["先 vram2ram %s" % ", ".join(cuda_hold)],
                    details={"cuda_hold": cuda_hold})
            return True
        raise TransitionError("unreachable")  # pragma: no cover

    # -- 操作摘要 --
    def summarize_operation(self, action: str, model_id: str,
                            requested: Sequence[str],
                            completed: Sequence[str] = (),
                            failed: Sequence[str] = (),
                            not_attempted: Sequence[str] = (),
                            dry_run: bool = False,
                            extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        op = {
            "action": action,
            "model": model_id,
            "mode": "all" if set(requested) == set(
                self._require_model(model_id).unit_ids()) else "units",
            "requested": list(requested),
            "completed": list(completed),
            "failed": list(failed),
            "not_attempted": list(not_attempted),
            "dry_run": bool(dry_run),
            "generation": self.model_generation.get(model_id, 0),
            "model_state": self.model_state(model_id)["state"],
        }
        if extra:
            op.update(extra)
        json.dumps(op)  # 必须可 JSON；不可序列化在此直接炸
        return op


# ---------------------------------------------------------------------------
# dry-run 预算（只读 recipe/header/catalog 与资源摘要，不物化权重）
# ---------------------------------------------------------------------------


def dry_run_estimate(catalog: UnitCatalog, unit_ids: Sequence[str],
                     action: str,
                     host_free_bytes: Optional[int] = None,
                     device_free_bytes: Optional[int] = None) -> Dict[str, Any]:
    """预算估算。超限直接抛 BudgetError（修改前拒绝）。"""
    want = catalog.expected_bytes_of(unit_ids)
    est: Dict[str, Any] = {
        "action": action,
        "units": list(unit_ids),
        "expected_resident_bytes": want,
        # 转换峰值按 2x 常驻估（普通 fp32->bf16 无整模型 fp32 副本；
        # 量化布局以真实 adapter 为准，此处标记估算可信度）。
        "peak_bytes": want * 2 if action == "disk2ram" else want,
        "confidence": "low-UNVERIFIED-layout",
    }
    if action == "disk2ram" and host_free_bytes is not None:
        est["host_free_bytes"] = host_free_bytes
        if est["peak_bytes"] > host_free_bytes:
            raise BudgetError(
                "disk2ram peak %d > host free %d" % (
                    est["peak_bytes"], host_free_bytes),
                recovery=["减少本次 unit 选择，或释放主机内存后重试"],
                details=est)
    if action == "ram2vram" and device_free_bytes is not None:
        est["device_free_bytes"] = device_free_bytes
        if want > device_free_bytes:
            raise BudgetError(
                "ram2vram %d > device free %d" % (want, device_free_bytes),
                recovery=["减少本次 unit 选择"],
                details=est)
    return est
