"""统一权重服务：五类模型的四动作分派、状态快照与 TE 编码入口。

API 层只和本模块打交道；HTTP 不再按模型写分支。每个模型是一个 backend：

| model | backend | 粒度 |
|---|---|---|
| dit | DitBackend | 50 blocks + refiner + 顶层模块（unit 级部分物化） |
| vae.video | VaeBackend | 整组（首版） |
| vae.audio | VaeBackend | 整组（首版） |
| te | TeBackend | te.model + te.proj（整组构造） |
| veda.bundle | VedaBackend | 整组（首版） |

未配置的模型保持 absent 占位；四动作对它返回结构化 409，而不是泛泛 501。
"""
from dit_svc import weights as W
from dit_svc.dit_weights import DitBackend
from dit_svc.operations import OperationError
from dit_svc.te_manager import TeBackend
from dit_svc.vae_weights import VaeBackend
from dit_svc.veda_weights import VedaBackend

MODEL_IDS = ("dit", "vae.video", "vae.audio", "te", "veda.bundle")


class WeightService:
    def __init__(self, runtime):
        self.runtime = runtime
        self.registry = W.WeightRegistry()
        self.backends = {
            "dit": DitBackend(runtime, self.registry),
            "vae.video": VaeBackend(runtime, self.registry, "vae.video", True),
            "vae.audio": VaeBackend(runtime, self.registry, "vae.audio", False),
            "te": TeBackend(runtime, self.registry),
            "veda.bundle": VedaBackend(runtime, self.registry),
        }

    # ---------------------------------------------------------- 生命周期
    def prepare(self):
        """登记所有已配置模型的 recipe/catalog（不读权重、不构造模型）。"""
        result = {}
        for model_id, backend in self.backends.items():
            try:
                result[model_id] = backend.prepare()
            except Exception as exc:
                result[model_id] = False
                backend.config_error = "%s: %s" % (type(exc).__name__, exc)
                print("[weights] %s prepare 失败: %s"
                      % (model_id, backend.config_error), flush=True)
        return result

    def backend(self, model_id):
        backend = self.backends.get(model_id)
        if backend is None:
            raise OperationError("MODEL_NOT_FOUND", "未知模型 %r" % (model_id,),
                                 404, False, {"model": model_id})
        return backend

    def ensure_catalog(self, model_id):
        backend = self.backend(model_id)
        if hasattr(backend, "ensure_catalog"):
            backend.ensure_catalog()
        return backend

    def unit_ids(self, model_id, request):
        return self.backend(model_id).unit_ids(request)

    # ---------------------------------------------------------- 快照
    def snapshot(self):
        snapshot = self.registry.snapshot()
        models = snapshot.setdefault("models", {})
        for model_id in MODEL_IDS:
            models.setdefault(model_id, {
                "model_id": model_id, "state": "absent", "generation": 0,
                "catalog_ready": False, "units": [], "bytes_cpu": 0,
                "bytes_cuda": 0,
            })
        for model_id, model in models.items():
            backend = self.backends.get(model_id)
            model["addressable"] = (backend is not None
                                    and backend.configured())
            if model.get("recipe"):
                model.setdefault("catalog_ready", True)
                model.setdefault("units", [])
                model.setdefault("absent_units", 0)
                model.setdefault("counts", {})
                bytes_info = model.get("bytes") or {}
                model.setdefault("bytes_cpu",
                                 int(bytes_info.get("resident_cpu", 0)))
                model.setdefault("bytes_cuda",
                                 int(bytes_info.get("resident_cuda", 0)))
        snapshot["ok"] = True
        return snapshot

    def model_summary(self, model_id):
        models = self.snapshot()["models"]
        if model_id not in models:
            raise OperationError("MODEL_NOT_FOUND", "未知模型", 404, False)
        return models[model_id]

    def state(self):
        return {model_id: backend.state()
                for model_id, backend in self.backends.items()}

    # ---------------------------------------------------------- 四动作
    def run_action(self, model_id, action, request, operation=None):
        return self.backend(model_id).run_action(action, request, operation)

    # ---------------------------------------------------------- 推理
    def encode(self, request, operation=None):
        return self.backends["te"].run_encode(request, operation)

    def probe(self, request, operation=None):
        from dit_svc.te_probe import run_probe
        return run_probe(self.backends["te"], request, operation)

    def preflight_encode(self, request=None):
        return self.backends["te"].preflight_encode(request)

    def prepare_vae(self, model_id):
        """把已 disk2ram 的 VAE 放到显存（量化 VAE 走目标设备重建）。

        供 decode 阶段使用；未配置/未加载直接 409，不自动 disk2ram。
        """
        backend = self.backend(model_id)
        if not backend.configured() or backend.obj is None:
            raise OperationError(
                "MISSING_UNITS", "%s 未在 RAM，先 disk2ram" % model_id,
                409, False, {"missing": [model_id]})
        record = self.registry.unit_record(model_id, backend.PRIMARY_UNIT)
        if record["state"] != "vram":
            backend.move_to("cuda:0")
            backend._refresh()
        return backend.obj

    @property
    def te(self):
        return self.backends["te"]
