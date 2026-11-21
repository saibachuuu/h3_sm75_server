"""DiT（H3）显式权重后端：按 unit 的部分 disk2ram / 搬运 / purge。

这里 finally 把 A 分支已经真机验证过的 ``H3ComfyAdapter`` 路径接成生产：
header -> catalog -> meta 骨架 -> 只读所选 keys -> strict 绑定 -> 物理 audit。
首版只负责“权重存储”，不构造 ModelPatcher / sampling 包装（B 的 h3_patcher
需要在真实包装验证后再接）。

约束：
* catalog 在首次动作时才读 header（启动仍是零权重空壳）；
* disk2ram 只物化所选 unit，未选 unit 保持 meta 零 storage；
* 每 unit 绑完立即 audit，verdict != ok 就标记 error 并停止；
* purge 保留 ModuleList 槽位（adapter 用 to_empty/meta placeholder）；
* 全部 unit absent 后释放骨架与 adapter，仅保留 recipe/catalog。
"""
import gc
import os

from dit_svc import weights as W
from dit_svc.group_weights import GroupBackend
from dit_svc.model_adapters import H3ComfyAdapter
from dit_svc.operations import OperationError

MODEL_ID = "dit"
ADAPTER_VERSION = "dit-h3-comfy-v1"
NONCOMPUTE_KEYS = ("adaln_basis", "adaln_mean")
NONCOMPUTE_REASON = (
    "checkpoint-only keys with no forward consumer in minimax/model.py "
    "(adaln_basis/mean); the consumed curve table is adaln_t_table. "
    "Never materialized, never counted resident, excluded from all-load.")


class DitBackend(GroupBackend):
    MODEL_ID = MODEL_ID
    ADAPTER_VERSION = ADAPTER_VERSION
    VALID_KINDS = ("checkpoint",)
    PRIMARY_UNIT = "dit.blocks.0"

    def __init__(self, runtime, registry):
        super().__init__(runtime, registry)
        self.model = None
        self.adapter = None
        self.unet_config = None
        self.quant_config = None
        self.shapes = None
        self.metadata = None
        self._catalog_ready = False
        self._patcher_pair = None

    # ---------------------------------------------------------- recipe/catalog
    def configured(self):
        return self.config is not None and self.config_error is None

    def default_kind(self, cfg):
        return "checkpoint"

    def prepare(self):
        """只登记 recipe（stat），不读 header；首次动作时才建 catalog。"""
        with self._lock:
            if self._prepared:
                return self.configured()
            cfg = dict((getattr(self.runtime.args, "model_recipes", {}) or {})
                       .get(MODEL_ID) or {})
            if not cfg:
                self._prepared = True
                return False
            try:
                self.config = self._clean_config(cfg)
                self.registry.register_recipe(self._make_recipe(self.config))
                self.config_error = None
            except Exception as exc:
                self.config_error = "%s: %s" % (type(exc).__name__, exc)
                self.config = None
                print("[dit] 配置无效: %s" % self.config_error, flush=True)
            self._prepared = True
            return self.configured()

    def state(self):
        out = {
            "backend": type(self).__name__,
            "configured": self.configured(),
            "config_error": self.config_error,
            "kind": (self.config or {}).get("kind"),
            "path": (self.config or {}).get("path"),
            "loaded": self.model is not None,
            "catalog_ready": self._catalog_ready,
            "last_error": self.last_error,
        }
        if self._catalog_ready:
            record = self.registry.model_state(MODEL_ID)
            out.update({"model_state": record["state"],
                        "generation": record["generation"],
                        "units": record["units"]})
        return out

    def ensure_catalog(self):
        """首次访问时读 header/metadata 并建立 catalog + adapter。"""
        with self._lock:
            if self._catalog_ready:
                return self.catalog
            if not self.configured():
                self._require_configured()
            from safetensors import safe_open

            path = self.config["path"]
            print("[dit] 读 header: %s" % os.path.basename(path), flush=True)
            tensors, _header_len = W.read_safetensors_header(path)
            with safe_open(path, framework="pt") as handle:
                metadata = dict(handle.metadata() or {})
            shapes = {k: tuple(v["shape"]) for k, v in tensors.items()}
            catalog = W.build_catalog_from_header(
                MODEL_ID, tensors, "dit",
                noncompute_exact=list(NONCOMPUTE_KEYS),
                noncompute_reason=NONCOMPUTE_REASON)
            catalog.check_cover(list(tensors.keys()))
            self.registry.attach_catalog(catalog)
            self.catalog = catalog
            self.shapes = shapes
            self.metadata = metadata
            self._catalog_ready = True
            print("[dit] catalog: %d units, %d keys, %d noncompute"
                  % (len(catalog.units), len(catalog.key_to_unit),
                     len(catalog.noncompute_keys)), flush=True)
            return catalog

    def _ensure_adapter(self):
        if self.adapter is not None:
            return
        import torch
        import comfy.model_detection as detection
        import comfy.ops as ops
        try:
            from comfy.ldm.minimax.model import MiniMaxH3Model
        except ImportError:
            import sys
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(
                    detection.__file__))), "ldm"))
            from minimax.model import MiniMaxH3Model  # type: ignore
        self.adapter = H3ComfyAdapter(torch_mod=torch, ops_mod=ops,
                                      model_cls=MiniMaxH3Model,
                                      detection_mod=detection)

    def _ensure_model(self, operation=None):
        self.ensure_catalog()
        if self.model is not None:
            return self.model
        import torch
        self._ensure_adapter()
        self._stage("DiT meta 骨架", operation)
        self.unet_config, _ = self.adapter.detect_unet_config(
            self.shapes, self.metadata)
        self.quant_config, _ = self.adapter.quant_config_from_metadata(
            self.metadata)
        self.model = self.adapter.build_skeleton(
            self.catalog, self.unet_config, self.quant_config,
            torch.bfloat16, device="meta")
        return self.model

    # ---------------------------------------------------------- 四动作
    def run_action(self, action, request, operation=None):
        from dit_svc.group_weights import WEIGHT_ACTIONS
        if action not in WEIGHT_ACTIONS:
            raise OperationError("INVALID_REQUEST", "未知动作 %r" % (action,),
                                 400, False)
        with self._lock:
            self.ensure_catalog()
            units = self.unit_ids(request)
            self.registry.check_transition(action, MODEL_ID, units)
            if request.get("dry_run"):
                return self._dry_run(action, units)
            self._stage("dit %s" % action, operation)
            if action == "disk2ram":
                return self._disk2ram(units, operation)
            if action == "ram2vram":
                return self._ram2vram(units, operation)
            if action == "vram2ram":
                return self._vram2ram(units, operation)
            return self._purge(units, operation)

    def _disk2ram(self, units, operation=None):
        from safetensors import safe_open

        records = {u: self.registry.unit_record(MODEL_ID, u) for u in units}
        pending = [u for u in units if records[u]["state"] == "absent"]
        if not pending:
            return self._summary("disk2ram", units, completed=list(units),
                                 extra={"noop": True,
                                        "reason": "already resident"})
        self._ensure_model(operation)
        completed = []
        failed = None
        try:
            keys = {uid: list(self.catalog.units[uid].checkpoint_keys)
                    for uid in pending}
            with safe_open(self.config["path"], framework="pt") as handle:
                for uid in pending:
                    sub = {k: handle.get_tensor(k) for k in keys[uid]}
                    report = self.adapter.materialize_unit(
                        self.model, uid, sub, device="cpu",
                        expected_keys=self.catalog.units[uid].checkpoint_keys)
                    audit = self.adapter.audit_after_bind(
                        self.catalog, self.model, uid, report,
                        expect_loaded=True)
                    if audit["verdict"] != "ok":
                        self.registry.mark_error(
                            MODEL_ID, uid,
                            "bind audit %s" % audit["verdict"],
                            recovery=["检查 header/量化布局后重新 disk2ram"])
                        raise OperationError(
                            "BIND_AUDIT_FAILED",
                            "%s 绑定后 audit=%s" % (uid, audit["verdict"]),
                            500, False, audit)
                    inventory = W.storage_inventory(
                        self.adapter.resident_tensors(self.model, uid),
                        owner=uid)
                    self.registry.set_unit_loaded(
                        MODEL_ID, uid, "cpu",
                        resident_bytes_cpu=inventory.total_cpu,
                        resident_bytes_cuda=inventory.total_cuda,
                        storage_digest=[e["storage_id"]
                                        for e in inventory.entries])
                    completed.append(uid)
                    print("[dit] disk2ram %s | %.3f GB | keys=%d"
                          % (uid, inventory.total_cpu / 1e9, len(sub)),
                          flush=True)
                    del sub
        except Exception as exc:
            failed = exc
            self.last_error = "%s: %s" % (type(exc).__name__, exc)
            if not isinstance(exc, OperationError):
                self.registry.mark_error(
                    MODEL_ID, pending[len(completed)] if len(completed) < len(pending)
                    else pending[-1], "%s: %s" % (type(exc).__name__, exc),
                    recovery=["检查磁盘/内存后重试该 unit"])
            raise
        finally:
            gc.collect()
        self._ensure_patchers()
        return self._summary(
            "disk2ram", units, completed=completed,
            extra={"not_attempted": [u for u in units if u not in completed],
                   "audit": self._unit_audit(completed)})

    def _ram2vram(self, units, operation=None):
        import torch
        if self.model is None:
            raise OperationError(
                "MISSING_UNITS", "DiT 骨架未构造，先 disk2ram", 409, False,
                {"missing": list(units)})
        before = self._cuda_bytes(torch)
        completed = []
        for uid in units:
            record = self.registry.unit_record(MODEL_ID, uid)
            if record["state"] == "absent":
                raise OperationError(
                    "MISSING_UNITS", "%s 未在 RAM，先 disk2ram" % uid, 409,
                    False, {"missing": [uid]})
            if record["state"] == "vram":
                completed.append(uid)
                continue
            self.adapter.move_unit(self.model, uid, "cuda:0")
            audit = self._refresh_unit(uid, expect_loaded=True)
            if audit["device"] != "cuda:0":
                self.registry.mark_error(
                    MODEL_ID, uid, "move to cuda left %s" % audit["device"],
                    recovery=["先 vram2ram，再 purgefromram 后重试"])
                raise OperationError(
                    "MOVE_FAILED", "%s 上卡后设备为 %s"
                    % (uid, audit["device"]), 500, True, audit)
            completed.append(uid)
        after = self._cuda_bytes(torch)
        self._refresh_patcher_sizes()
        print("[dit] ram2vram %d units | cuda %.2f -> %.2f GB"
              % (len(completed), before / 1e9, after / 1e9), flush=True)
        return self._summary("ram2vram", units, completed=completed,
                             extra={"cuda_bytes_before": before,
                                    "cuda_bytes_after": after})

    def _vram2ram(self, units, operation=None):
        import torch
        if self.model is None:
            return self._summary("vram2ram", units, completed=list(units),
                                 extra={"noop": True, "reason": "not loaded"})
        before = self._cuda_bytes(torch)
        completed = []
        for uid in units:
            record = self.registry.unit_record(MODEL_ID, uid)
            if record["state"] == "absent":
                continue
            self.adapter.move_unit(self.model, uid, "cpu")
            audit = self._refresh_unit(uid, expect_loaded=True)
            if audit["device"] != "cpu":
                self.registry.mark_error(
                    MODEL_ID, uid, "move to cpu left %s" % audit["device"],
                    recovery=["先 purgefromram 后重新 disk2ram"])
                raise OperationError(
                    "MOVE_FAILED", "%s 下卡后设备为 %s"
                    % (uid, audit["device"]), 500, True, audit)
            completed.append(uid)
        torch.cuda.empty_cache()
        after = self._cuda_bytes(torch)
        print("[dit] vram2ram %d units | cuda %.2f -> %.2f GB"
              % (len(completed), before / 1e9, after / 1e9), flush=True)
        return self._summary("vram2ram", units, completed=completed,
                             extra={"cuda_bytes_before": before,
                                    "cuda_bytes_after": after})

    def _purge(self, units, operation=None):
        if self.model is None:
            records = {u: self.registry.unit_record(MODEL_ID, u) for u in units}
            if all(r["state"] == "absent" for r in records.values()):
                return self._summary("purgefromram", units,
                                     completed=list(units),
                                     extra={"noop": True})
        completed = []
        for uid in units:
            record = self.registry.unit_record(MODEL_ID, uid)
            if record["resident_bytes_cuda"] > 0 or record["device"] == "cuda:0":
                raise OperationError(
                    "CUDA_RESIDENT", "%s 仍有 CUDA 驻留，先 vram2ram" % uid,
                    409, False, {"unit": uid})
            if record["state"] == "absent":
                completed.append(uid)
                continue
            self.adapter.purge_unit(self.model, uid)
            audit = self.adapter.audit_after_bind(
                self.catalog, self.model, uid,
                {"bound_keys": [], "waived": []}, expect_loaded=False)
            if audit["verdict"] != "ok":
                self.registry.mark_error(
                    MODEL_ID, uid, "purge audit %s" % audit["verdict"],
                    recovery=["先 vram2ram，再重试 purge"])
                raise OperationError(
                    "PURGE_INCOMPLETE", "%s purge 后 audit=%s"
                    % (uid, audit["verdict"]), 500, True, audit)
            self.registry.set_unit_absent(MODEL_ID, uid)
            completed.append(uid)
            print("[dit] purge %s | meta=%d" % (uid, audit["meta_tensors"]),
                  flush=True)
        remaining = [u for u in self.catalog.unit_ids()
                     if self.registry.unit_record(MODEL_ID, u)["state"] != "absent"]
        if not remaining:
            self._release_model()
        return self._summary("purgefromram", units, completed=completed,
                             extra={"all_absent": not remaining})

    # ---------------------------------------------------------- patcher 外壳
    def _all_units_resident(self):
        if self.catalog is None:
            return False
        return all(self.registry.unit_record(MODEL_ID, unit)["state"]
                   in ("ram", "vram") for unit in self.catalog.unit_ids())

    def _ensure_patchers(self):
        """全量物化后包装成可采样的 (unet, model_shifted)。

        只做 metadata/采样外壳：不重读权重、不 load、不 cast。B 的
        wrap_h3_skeleton 用私有 disable 开关跳过第二次大模型构造，再把现有
        skeleton 绑到 BaseModel 上。
        """
        if self._patcher_pair is not None or self.model is None:
            return self._patcher_pair
        if not self._all_units_resident():
            return None
        import torch
        import comfy.supported_models as supported
        import comfy.model_base as model_base
        import comfy.model_patcher as model_patcher
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
        from dit_svc.h3_patcher import wrap_h3_skeleton

        self._stage("DiT 采样外壳（BaseModel/ModelPatcher + SigmaShift）")
        pair = wrap_h3_skeleton(
            self.model, self.unet_config, torch_mod=torch,
            supported_models_mod=supported, model_base_mod=model_base,
            model_patcher_mod=model_patcher,
            sigma_shift_node=MiniMaxH3SigmaShift,
            load_device=torch.device("cuda:0"),
            offload_device=torch.device("cpu"),
            inference_dtype=torch.bfloat16,
            manual_cast_dtype=torch.bfloat16,
            quant_config=self.quant_config)
        self._patcher_pair = pair
        self.runtime.unet = pair.unet
        self.runtime.model_shifted = pair.model_shifted
        self.runtime.diff_model = self.model
        print("[dit] patcher 外壳就绪: unet=%s shifted=%s"
              % (type(pair.unet).__name__, type(pair.model_shifted).__name__),
              flush=True)
        return pair

    def _refresh_patcher_sizes(self):
        if self._patcher_pair is None:
            return
        from dit_svc.h3_patcher import invalidate_patcher_sizes
        cuda_bytes = sum(
            int(self.registry.unit_record(MODEL_ID, unit)["resident_bytes_cuda"])
            for unit in self.catalog.unit_ids())
        invalidate_patcher_sizes(
            self._patcher_pair.unet, self._patcher_pair.model_shifted,
            model_loaded_weight_memory=cuda_bytes)

    # ---------------------------------------------------------- 账目
    def _refresh_unit(self, uid, expect_loaded=True):
        inventory = W.storage_inventory(
            self.adapter.resident_tensors(self.model, uid), owner=uid)
        if inventory.uninspectable:
            self.registry.mark_error(
                MODEL_ID, uid, "uninspectable: %s"
                % inventory.uninspectable[:3],
                recovery=["检查量化包装/storage 展开后重新 disk2ram"])
            raise OperationError("UNINSPECTABLE_WEIGHTS",
                                 "%s 有无法审计的张量" % uid, 500, False,
                                 {"unit": uid})
        cpu, cuda = inventory.total_cpu, inventory.total_cuda
        if cpu and cuda:
            device = "mixed"
        elif cuda:
            device = "cuda:0"
        elif cpu:
            device = "cpu"
        else:
            device = "absent"
        audit = {"unit": uid, "device": device, "cpu": cpu, "cuda": cuda,
                 "meta_tensors": inventory.meta_tensors,
                 "num_storages": inventory.num_storages}
        if expect_loaded and device == "absent":
            raise OperationError("META_MISSING", "%s 没有任何真实存储" % uid,
                                 500, False, audit)
        if device == "mixed":
            raise OperationError("MIXED_DEVICE", "%s 跨设备" % uid, 500, False,
                                 audit)
        if device in ("cpu", "cuda:0"):
            self.registry.set_unit_loaded(
                MODEL_ID, uid, device, resident_bytes_cpu=cpu,
                resident_bytes_cuda=cuda,
                storage_digest=[e["storage_id"] for e in inventory.entries])
        return audit

    def _unit_audit(self, units):
        return {uid: self._refresh_unit(uid) for uid in units}

    def inventory(self):
        """全量账目（GET / status 用；不做 audit 副作用）。"""
        out = {"units": {}, "uninspectable": [], "storage_digest": []}
        if self.model is None or self.adapter is None:
            return out
        for uid in self.catalog.unit_ids():
            inventory = W.storage_inventory(
                self.adapter.resident_tensors(self.model, uid), owner=uid)
            cpu, cuda = inventory.total_cpu, inventory.total_cuda
            if cpu and cuda:
                device = "mixed"
            elif cuda:
                device = "cuda:0"
            elif cpu:
                device = "cpu"
            else:
                device = "absent"
            out["units"][uid] = {"device": device, "cpu": cpu, "cuda": cuda}
            out["uninspectable"] += list(inventory.uninspectable)
            out["storage_digest"] += [e["storage_id"]
                                      for e in inventory.entries]
        return out

    def _release_model(self):
        refs = []
        self.runtime.unet = None
        self.runtime.model_shifted = None
        self.runtime.diff_model = None
        self._patcher_pair = None
        if self.model is not None:
            refs.append(self._release_ref(self.model))
        self.model = None
        gc.collect()
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        alive = [i for i, ref in enumerate(refs) if ref() is not None]
        if alive:
            raise OperationError(
                "PURGE_INCOMPLETE",
                "DiT 骨架仍被其它引用持有，拒绝谎报 absent", 500, True,
                {"proof": {"weakrefs_alive": alive}})
