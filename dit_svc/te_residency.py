"""TE-only native Comfy bookkeeping and resident execution (lazy imports).

The global loader is never used to register TE: its free_memory/clone eviction
policy is inappropriate for explicit ownership. Only native LoadedModel.model_load
runs, during the explicit TE move. Inference only validates that registration.
"""
import contextlib
import hashlib
import json

from dit_svc import weights as W
from dit_svc.comfy_bindings import (ComfyBindings, PHASE_EXPLICIT_MOVE,
                                    PHASE_RESIDENT_INFER)
from dit_svc.operations import OperationError


class TeResidency:
    def __init__(self, runtime):
        self.runtime = runtime
        self.bindings = ComfyBindings(log=lambda message: print(message, flush=True))

    def ensure(self):
        mm = self.runtime.mm
        if mm is None:
            raise OperationError("ENV_NOT_READY", "Comfy 环境未就绪", 503, True)
        self.bindings.install(mm)
        record = self.bindings._record
        for name, wrapper in record.wrappers.items():
            if getattr(mm, name) is not wrapper:
                raise OperationError("BINDINGS_CONFLICT", "%s 被第三方替换" % name,
                                     503, False)
        aliases = self.bindings.audit_aliases()
        if aliases:
            raise OperationError("UNSAFE_COMFY_ALIAS", "存在绕过驻留保护的裸函数别名",
                                 503, False, {"aliases": aliases})
        return mm

    def _register(self, backend):
        patcher = backend.obj.patcher
        registry = self.bindings.registry
        def devices():
            units = backend.inventory()["units"]
            return {name: item["device"] for name, item in units.items()}
        if registry.get("te") is None:
            registry.register(patcher, "te", devices(), probe=devices)
        elif registry.resolve(patcher) is not registry.get("te"):
            raise OperationError("BOOKKEEPING_CONFLICT", "TE 对象与登记不符", 503, False)

    def _bookings(self, mm, patcher):
        return [entry for entry in mm.current_loaded_models
                if entry.model is patcher]

    def move(self, backend, device):
        mm = self.ensure()
        self._register(backend)
        patcher = backend.obj.patcher
        if patcher.is_dynamic():
            raise OperationError("UNSUPPORTED_TE_PATCHER", "resident TE 不支持动态卸载",
                                 409, False)
        if patcher.model_patches_models():
            raise OperationError("UNSUPPORTED_TE_PATCHER", "TE 带有额外模型依赖",
                                 409, False)
        before = self.snapshot(exclude_te=True)
        with self.bindings.execution(PHASE_EXPLICIT_MOVE, "te.%s" % device,
                                     allowed_models=("te",),
                                     declared_units=backend.inventory()["units"]):
            try:
                bookings = self._bookings(mm, patcher)
                if len(bookings) > 1:
                    raise OperationError("BOOKKEEPING_CONFLICT", "TE 重复登记", 503, False)
                if device == "cuda:0":
                    if str(patcher.load_device) != device:
                        raise OperationError("UNSUPPORTED_TE_DEVICE", "TE load_device 不匹配",
                                             409, False)
                    if not bookings:
                        entry = mm.LoadedModel(patcher)
                        entry.model_load()  # native full load, no global eviction
                        mm.current_loaded_models.insert(0, entry)
                    inv = backend.inventory()
                    item = inv["units"]["te.model"]
                    if inv["uninspectable"] or item["device"] != device or item["cpu"]:
                        raise OperationError("MOVE_FAILED", "TE 未完整上卡", 500, False, inv)
                    if getattr(backend, "_projected", False):
                        if backend.obj.is_prepared:
                            backend.obj.require_projection(device)
                        else:
                            backend.obj.prepare_projection(device)
                else:
                    for entry in bookings:
                        entry.model_unload()
                        mm.current_loaded_models.remove(entry)
                    if not bookings:
                        patcher.detach()  # clean partial native-load bookkeeping too
                    patcher.model.to(device)
                    backend.drop_caches()
                self.assert_unchanged(before, self.snapshot(exclude_te=True))
            except BaseException:
                self.assert_unchanged(before, self.snapshot(exclude_te=True))
                raise

    def unregister(self):
        if self.bindings.registry.get("te") is not None:
            self.bindings.registry.unregister("te")

    def require(self, backend):
        mm = self.ensure()
        if self.bindings.registry.resolve(backend.obj.patcher) is None:
            raise OperationError("MISSING_RESIDENCE", "TE 未显式登记，先 ram2vram",
                                 409, False)
        bookings = self._bookings(mm, backend.obj.patcher)
        if len(bookings) != 1 or str(bookings[0].device) != "cuda:0":
            raise OperationError("MISSING_RESIDENCE", "TE 没有有效 CUDA 登记",
                                 409, False)
        if getattr(backend, "_projected", False):
            backend.obj.require_projection("cuda:0")
            gpu = {"cache": backend.obj.__dict__["_gpu"],
                   "sink": backend.obj.__dict__["_proj"].get("sink_out")}
            inv = W.storage_inventory(gpu, owner="te.proj.gpu")
            if inv.uninspectable or inv.meta_tensors or inv.total_cpu or not inv.total_cuda:
                raise OperationError("MISSING_UNITS", "投影计算副本未完整驻留 CUDA",
                                     409, False, {"missing": ["te.proj"]})

    @contextlib.contextmanager
    def inference(self, backend, operation=None):
        self.require(backend)
        before = self.snapshot()
        proof = {}
        required = tuple(backend.inventory()["units"])
        with self.bindings.execution(PHASE_RESIDENT_INFER,
                                     getattr(operation, "id", None) or "te.encode",
                                     declared_units=required,
                                     expected_devices={name: "cuda:0" for name in required}):
            try:
                yield proof
            finally:
                after = self.snapshot()
                self.assert_unchanged(before, after)
                self.require(backend)
                proof.update({"unchanged": True, "models": sorted(after),
                              "storage_sha256": self._digest(after)})

    def snapshot(self, exclude_te=False):
        """Read physical storage, not registry device labels. Partial DiT is legal."""
        result = {}
        for name, backend in self.runtime.weights.backends.items():
            if name == "te":
                if exclude_te:
                    continue
                tensors = {"model": backend._clip_model()}
                if getattr(backend, "_projected", False):
                    tensors["projection_cpu"] = backend.obj.projection_cpu
                    tensors["projection_gpu"] = backend.obj.__dict__["_gpu"]
                    tensors["projection_runtime"] = backend.obj.__dict__["_proj"]
                result[name] = self._storage(W.storage_inventory(tensors, owner="te"))
            else:
                inv = backend.inventory()
                if inv.get("uninspectable"):
                    raise OperationError("UNINSPECTABLE_WEIGHTS", "无法检查 %s" % name,
                                         503, False, {"model": name})
                result[name] = {"units": inv["units"],
                                "storages": sorted(inv["storage_digest"])}
        mm = self.runtime.mm
        if mm is not None:
            for entry in list(mm.current_loaded_models):
                patcher = entry.model
                if patcher is None:
                    raise OperationError("BOOKKEEPING_CONFLICT", "失效的 Comfy 模型登记",
                                         503, False)
                if getattr(self.runtime.weights.te.obj, "patcher", None) is patcher:
                    continue
                result["comfy.%x" % id(patcher)] = self._storage(
                    W.storage_inventory(patcher.model, owner="comfy"))
        return result

    @staticmethod
    def _storage(inv):
        if inv.uninspectable:
            raise OperationError("UNINSPECTABLE_WEIGHTS", "无法检查真实权重存储",
                                 503, False, {"uninspectable": inv.uninspectable})
        return {"cpu": inv.total_cpu, "cuda": inv.total_cuda,
                "meta": inv.meta_tensors,
                "storages": sorted(entry["storage_id"] for entry in inv.entries)}

    @staticmethod
    def _digest(snapshot):
        return hashlib.sha256(json.dumps(snapshot, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()

    def assert_unchanged(self, before, after):
        changed = [name for name in set(before) | set(after)
                   if before.get(name) != after.get(name)]
        if changed:
            self.runtime.device_healthy = False
            raise OperationError("WEIGHT_MUTATED", "TE 操作期间权重或登记被改变",
                                 500, False, {"models": sorted(changed)})
