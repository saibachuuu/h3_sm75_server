"""Explicit ClipProj residency without copying the plugin's encoder/MLP math.

Only the inspected plugin encoder is supported. GPU coefficients, residual MLP
and sink are prepared during ram2vram. The original _encode runs against a sealed
cache and cannot enter its lazy-weight path. Imports stay torch/Comfy-free.
"""
import hashlib
import inspect
import re
import sys
import textwrap

from dit_svc import weights as W
from dit_svc.operations import OperationError

_STATS = ("mean_in", "std_in", "mean_out", "std_out")
_ENCODE_SHA256 = "f1da7ee8cee419a69e174c062be6f083388ade5612e1b6c1cfb3d013ca15beea"
_RESIDENT_CLASSES = {}


class ResidentGpuCache(dict):
    """Read-only dict accepted by both the plugin and storage_inventory."""
    is_frozen = True

    def _refuse(self, *args, **kwargs):
        raise OperationError("CACHE_MUTATION_FORBIDDEN", "推理不得创建、替换或清空投影权重",
                             409, False)

    __setitem__ = __delitem__ = clear = pop = popitem = update = setdefault = __ior__ = _refuse


def validate_projection_dict(proj, projection_name=""):
    import torch
    def invalid(message):
        raise OperationError("INVALID_PROJECTION_CONFIG", message, 422, False,
                             {"projection": projection_name})
    if not isinstance(proj, dict):
        invalid("投影必须是张量字典")
    if "control" in proj or projection_name.startswith("<control:"):
        raise OperationError("UNSUPPORTED_PROJECTION", "resident 推理只支持已训练的投影",
                             400, False)
    if type(proj.get("tap")) is not int or proj["tap"] < -1:
        invalid("缺少或无效的 tap")
    def tensor(key, shape=None):
        value = proj.get(key)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            invalid("%s 必须是有存储的 CPU Tensor" % key)
        if shape is not None and tuple(value.shape) != shape:
            invalid("%s 形状 %s 与 %s 不符" % (key, tuple(value.shape), shape))
        if not value.is_floating_point():
            invalid("%s 必须是浮点权重" % key)
        return value
    for key in _STATS:
        if tensor(key).ndim != 1 or tensor(key).numel() == 0:
            invalid("%s 必须是非空一维向量" % key)
        if tensor(key).dtype != torch.float32:
            invalid("%s 必须保持插件的 float32 归一化格式" % key)
        if not torch.isfinite(tensor(key)).all():
            invalid("%s 含非有限值" % key)
    d_in, d_out = proj["mean_in"].numel(), proj["mean_out"].numel()
    tensor("std_in", (d_in,))
    tensor("std_out", (d_out,))
    if not (proj["std_in"] > 0).all() or not (proj["std_out"] > 0).all():
        invalid("投影标准差必须为正数")
    if "W" in proj:
        if tensor("W", (d_in, d_out)).dtype != torch.float32:
            invalid("W 必须保持插件的 float32 矩阵格式")
    layers = set()
    for key in proj:
        if key.startswith("mlp."):
            match = re.fullmatch(r"mlp\.(\d+)\.(weight|bias)", key)
            if match is None:
                invalid("无效的 MLP 键 %s" % key)
            layers.add(int(match[1]))
    previous = d_in
    for index in sorted(layers):
        weight = tensor("mlp.%d.weight" % index)
        if weight.ndim != 2 or weight.shape[1] != previous:
            invalid("MLP 第 %d 层输入不匹配" % index)
        previous = weight.shape[0]
        key = "mlp.%d.bias" % index
        if key in proj:
            tensor(key, (previous,))
    if layers and previous != d_out:
        invalid("MLP 输出与投影输出维度不符")
    if not layers and "W" not in proj:
        invalid("投影既没有 W，也没有 MLP")
    if "sink_out" in proj:
        sink = tensor("sink_out")
        if tuple(sink.shape) not in ((d_out,), (1, d_out)):
            invalid("sink_out 形状不符")
    inv = W.storage_inventory(proj)
    if inv.uninspectable or inv.meta_tensors:
        invalid("投影存在无法检查的存储")
    return d_in, d_out


def _check_plugin(base_cls):
    try:
        source = textwrap.dedent(inspect.getsource(base_cls._encode)).strip()
        digest = hashlib.sha256(source.encode()).hexdigest()
    except (OSError, TypeError) as exc:
        raise OperationError("UNSUPPORTED_CLIPPROJ", "无法检查 ClipProj encoder", 503, False) from exc
    if digest != _ENCODE_SHA256:
        raise OperationError("UNSUPPORTED_CLIPPROJ", "ClipProj encoder 与已核对版本不符",
                             503, False, {"actual_sha256": digest})


def _check_encoder_config(config, proj):
    width = getattr(config, "hidden_size", None)
    layers = getattr(config, "num_hidden_layers", None)
    if type(width) is not int or type(layers) is not int:
        raise OperationError("UNSUPPORTED_CLIPPROJ", "无法核对 encoder 维度", 503, False)
    if width != proj["mean_in"].numel() or proj["tap"] >= layers:
        raise OperationError("INCOMPATIBLE_PROJECTION", "投影维度或 tap 与 encoder 不符",
                             422, False, {"encoder_width": width, "layers": layers})


def get_resident_projected_class(base_cls=None):
    # Explicit base injection is for CPU contract tests, never an API option.
    if base_cls is None:
        from clipproj.clipproj_nodes import ProjectedCLIP
        base_cls = ProjectedCLIP
        _check_plugin(base_cls)
    if base_cls in _RESIDENT_CLASSES:
        return _RESIDENT_CLASSES[base_cls]

    class ResidentProjectedCLIP(base_cls):
        def __init__(self, base, projection_name, *, build_residual_fn=None):
            super().__init__(base, projection_name)
            proj = self.__dict__["_proj"]
            # Adopt only this projection's cache entry; do not clear unrelated data.
            module = sys.modules.get("clipproj.clipproj_projection")
            cache = getattr(module, "_CACHE", {})
            for key, value in list(cache.items()):
                if value is proj:
                    del cache[key]
            validate_projection_dict(proj, projection_name)
            if build_residual_fn is None:
                from clipproj.clipproj_nodes import _submodel
                _check_encoder_config(_submodel(base).transformer.model.config, proj)
            self.__dict__.update(_proj_cpu=proj, _gpu=ResidentGpuCache(),
                                 _prepared_device=None, _build_residual_fn=build_residual_fn)

        @property
        def projection_cpu(self):
            return self.__dict__["_proj_cpu"]

        @property
        def prepared_device(self):
            return self.__dict__["_prepared_device"]

        @property
        def is_prepared(self):
            return self.prepared_device is not None

        def prepare_projection(self, device):
            import torch
            dev = torch.device(device)
            if self.is_prepared:
                self.require_projection(dev)
                return self
            proj = self.projection_cpu
            validate_projection_dict(proj, self._proj_name)
            coefficients = ResidentGpuCache({key: proj[key].to(dev)
                                             for key in ("W",) + _STATS if key in proj})
            build = self.__dict__["_build_residual_fn"]
            if build is None:
                from clipproj.clipproj_projection import build_residual
                build = build_residual
            mlp = build(proj, dev)
            sink = proj.get("sink_out")
            sink = sink.to(device=dev, dtype=torch.float32) if sink is not None else None
            runtime = dict(proj)
            if sink is not None:
                runtime["sink_out"] = sink
            cache = ResidentGpuCache(device=dev, p=coefficients, mlp=mlp, sink_out=sink)
            # Validate the temporary whole cache before publishing anything.
            self._validate_cache(cache, dev)
            self.__dict__.update(_gpu=cache, _proj=runtime, _prepared_device=dev)
            return self

        def _validate_cache(self, cache, device):
            import torch
            def missing():
                raise OperationError("MISSING_UNITS", "投影计算副本不完整或设备不符",
                                     409, False, {"missing": ["te.proj"], "reason": "cache_incomplete"})
            proj = self.projection_cpu
            keys = {key for key in ("W",) + _STATS if key in proj}
            if not isinstance(cache, ResidentGpuCache) or cache.get("device") != device:
                missing()
            coefficients = cache.get("p")
            if not isinstance(coefficients, ResidentGpuCache) or set(coefficients) != keys:
                missing()
            for key in keys:
                value = coefficients[key]
                if value.device != device or value.shape != proj[key].shape or value.dtype != proj[key].dtype:
                    missing()
            has_mlp = any(key.startswith("mlp.") for key in proj)
            mlp = cache.get("mlp")
            if has_mlp != (mlp is not None):
                missing()
            if mlp is not None:
                if any(t.device != device for t in list(mlp.parameters()) + list(mlp.buffers())):
                    missing()
                indices = sorted({int(key.split(".")[1]) for key in proj if key.startswith("mlp.")})
                linear = [layer for layer in mlp if isinstance(layer, torch.nn.Linear)]
                if len(linear) != len(indices):
                    missing()
                dtype = proj["mlp.%d.weight" % indices[0]].dtype
                for layer, index in zip(linear, indices):
                    weight = proj["mlp.%d.weight" % index]
                    bias = proj.get("mlp.%d.bias" % index)
                    if layer.weight.shape != weight.shape or layer.weight.dtype != dtype:
                        missing()
                    if (layer.bias is None) != (bias is None):
                        missing()
                    if bias is not None and (layer.bias.shape != bias.shape or layer.bias.dtype != dtype):
                        missing()
            sink = cache.get("sink_out")
            if ("sink_out" in proj) != (sink is not None):
                missing()
            if sink is not None and (sink.device != device or sink.dtype != torch.float32):
                missing()
            inv = W.storage_inventory(cache)
            if inv.uninspectable or inv.meta_tensors:
                missing()

        def require_projection(self, device):
            import torch
            if not self.is_prepared:
                raise OperationError("MISSING_UNITS", "投影尚未显式上卡", 409, False,
                                     {"missing": ["te.proj"], "reason": "not_prepared"})
            dev = torch.device(device)
            if self.prepared_device != dev:
                raise OperationError("MISSING_UNITS", "投影与执行设备不符", 409, False,
                                     {"missing": ["te.proj"], "reason": "device_mismatch"})
            self._validate_cache(self.__dict__["_gpu"], dev)
            if "sink_out" in self.projection_cpu:
                if self.__dict__["_proj"].get("sink_out") is not self.__dict__["_gpu"]["sink_out"]:
                    raise OperationError("MISSING_UNITS", "sink 未使用已准备副本", 409, False)
            return True

        def clear_projection(self):
            old = self.__dict__["_gpu"]
            self.__dict__.update(_gpu=ResidentGpuCache(), _proj=self.projection_cpu,
                                 _prepared_device=None)
            dict.clear(old)  # explicit release only; encoder never gets this capability

        def _encode(self, tokens):
            self.require_projection(self._base.patcher.load_device)
            return super()._encode(tokens)

        def clone(self):
            return type(self)(self._base.clone(), self._proj_name,
                              build_residual_fn=self.__dict__["_build_residual_fn"])

    _RESIDENT_CLASSES[base_cls] = ResidentProjectedCLIP
    return ResidentProjectedCLIP


def wrap_resident_projected(base, projection_name, *, base_cls=None, build_residual_fn=None):
    return get_resident_projected_class(base_cls)(
        base, projection_name, build_residual_fn=build_residual_fn)
