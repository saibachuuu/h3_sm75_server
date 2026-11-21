"""整组模型的显式权重后端：四动作 + 物理账 + purge 释放证明。

TE / VAE / VEDA 都是“能整体构造、整体搬运”的模型，区别只在构造与搬运方式。
本模块只保留通用状态机，具体模型在各自 backend 里实现：

* ``build_units(cfg)``：unit 列表（整组模型可以只有 1 个 unit）；
* ``construct_cpu(operation)``：从磁盘构造到 CPU，设置 ``self.obj``；
* ``move_to(device)``：只搬设备，不推理；
* ``drop_caches()``：清掉推理期产生的设备副本（如 ClipProj 的 GPU 缓存）；
* ``release()``：释放全部对象并返回 weakref 列表（实测释放证明）；
* ``inventory()``：返回 per-unit 的真实 storage 账。

四动作语义（plan 02 §4）：
* disk2ram 只构造到 CPU，不在 GPU 留任何权重；
* ram2vram 只搬主 unit，绝不触发额外加载；
* vram2ram 搬回并清设备副本；
* purgefromram 拒绝仍有 CUDA 驻留的 unit，释放后按 weakref 证明置 absent。
"""
import gc
import os
import threading
import time
import traceback
import weakref

from dit_svc import api_validation
from dit_svc import weights as W
from dit_svc.operations import OperationError

WEIGHT_ACTIONS = ("disk2ram", "ram2vram", "vram2ram", "purgefromram")


def suggestions(model_id, model_state):
    if model_state in ("absent",):
        return [{"action": "disk2ram", "model": model_id, "mode": "all"},
                {"action": "ram2vram", "model": model_id, "mode": "all"}]
    if model_state in ("partial", "ram"):
        return [{"action": "ram2vram", "model": model_id, "mode": "all"}]
    return [{"action": "purgefromram", "model": model_id, "mode": "all"}]


class GroupBackend:
    MODEL_ID = ""
    ADAPTER_VERSION = "group-v0"
    VALID_KINDS = ("checkpoint",)
    PRIMARY_UNIT = ""

    def __init__(self, runtime, registry):
        self.runtime = runtime
        self.registry = registry
        self._lock = threading.RLock()
        self.config = None
        self.catalog = None
        self.config_error = None
        self.obj = None
        self.load_s = 0.0
        self.last_error = None
        self._prepared = False

    # ---------------------------------------------------------- 子类钩子
    def build_units(self, cfg):
        raise NotImplementedError

    def default_kind(self, cfg):
        return "checkpoint"

    def construct_cpu(self, operation=None):
        raise NotImplementedError

    def move_to(self, device):
        raise NotImplementedError

    def drop_caches(self):
        pass

    def release(self):
        raise NotImplementedError

    def inventory(self):
        raise NotImplementedError

    def extra_state(self):
        return {}

    def validate_config(self, cfg):
        return cfg

    # ---------------------------------------------------------- recipe/catalog
    def prepare(self):
        """登记 recipe/catalog；只 stat 文件，不读权重、不构造模型。幂等。"""
        with self._lock:
            if self._prepared:
                return self.configured()
            cfg = dict((getattr(self.runtime.args, "model_recipes", {}) or {})
                       .get(self.MODEL_ID) or {})
            if not cfg:
                self._prepared = True
                return False
            try:
                self.config = self.validate_config(self._clean_config(cfg))
                self.registry.register_recipe(self._make_recipe(self.config))
                self.catalog = W.UnitCatalog(self.MODEL_ID,
                                             self.build_units(self.config))
                self.registry.attach_catalog(self.catalog)
                self.config_error = None
            except Exception as exc:
                self.config_error = "%s: %s" % (type(exc).__name__, exc)
                self.catalog = None
                self.config = None
                print("[%s] 配置无效: %s" % (self.MODEL_ID, self.config_error),
                      flush=True)
                self._prepared = True
                return False
            self._prepared = True
            print("[%s] recipe: %s (%s)"
                  % (self.MODEL_ID, self.config["kind"],
                     os.path.basename(self.config["path"])), flush=True)
            return True

    def _clean_config(self, cfg):
        allowed = {"kind", "path", "proj", "mmproj", "options"}
        unknown = sorted(set(cfg) - allowed)
        if unknown:
            raise ValueError("%s recipe 含未知字段: %s"
                             % (self.MODEL_ID, ", ".join(unknown)))
        path = cfg.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("%s.path 必须是非空字符串" % self.MODEL_ID)
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        proj = cfg.get("proj")
        mmproj = cfg.get("mmproj")
        for label, value in (("proj", proj), ("mmproj", mmproj)):
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError("%s.%s 必须是非空字符串"
                                 % (self.MODEL_ID, label))
            if os.path.isabs(value):
                resolved = os.path.abspath(value)
                if not os.path.isfile(resolved):
                    raise FileNotFoundError(resolved)
                value = resolved
        options = cfg.get("options") or {}
        if not isinstance(options, dict):
            raise ValueError("%s.options 必须是 JSON 对象" % self.MODEL_ID)
        kind = cfg.get("kind")
        if kind is None:
            kind = self.default_kind(cfg)
        if kind not in self.VALID_KINDS:
            raise ValueError("%s.kind 只能是 %s"
                             % (self.MODEL_ID, ", ".join(self.VALID_KINDS)))
        return {"kind": kind, "path": path, "proj": proj, "mmproj": mmproj,
                "options": dict(options)}

    def _make_recipe(self, cfg):
        def stat(path):
            values = os.stat(path)
            return {"path": os.path.abspath(path), "size": values.st_size,
                    "mtime_ns": values.st_mtime_ns}

        extras = [p for p in (cfg.get("proj"), cfg.get("mmproj"))
                  if p and os.path.isabs(p)]
        fingerprint = stat(cfg["path"])
        if extras:
            fingerprint["extras"] = [stat(p) for p in extras]
        return W.ModelRecipe(
            model_id=self.MODEL_ID, kind=cfg["kind"], path=cfg["path"],
            extra_paths=tuple(extras),
            options={"proj": cfg.get("proj"), "mmproj": cfg.get("mmproj"),
                     **cfg["options"]},
            fingerprint=fingerprint, adapter_version=self.ADAPTER_VERSION)

    # ---------------------------------------------------------- 查询
    def configured(self):
        return self.catalog is not None and self.config_error is None

    def state(self):
        out = {
            "backend": type(self).__name__,
            "configured": self.configured(),
            "config_error": self.config_error,
            "kind": (self.config or {}).get("kind"),
            "path": (self.config or {}).get("path"),
            "loaded": self.obj is not None,
            "load_s": round(self.load_s, 3),
            "last_error": self.last_error,
        }
        out.update(self.extra_state())
        if self.configured():
            record = self.registry.model_state(self.MODEL_ID)
            out.update({"model_state": record["state"],
                        "generation": record["generation"],
                        "units": record["units"]})
        return out

    def _require_configured(self):
        if self.catalog is None:
            raise OperationError(
                "%s_NOT_CONFIGURED" % self.MODEL_ID.upper().replace(".", "_"),
                "%s 尚未配置: %s"
                % (self.MODEL_ID, self.config_error or "defaults-file 缺少该模型"),
                409, False,
                {"config_error": self.config_error,
                 "hint": "在 --defaults-file 的 models.%s 里给 path/kind"
                         % self.MODEL_ID})

    def unit_ids(self, request):
        self._require_configured()
        if request.get("mode") == "all":
            return self.catalog.unit_ids()
        return api_validation.expand_units(request.get("units") or (),
                                           self.catalog)

    def _require_group(self, units):
        """整组后端的 group 约束；子类覆盖（如 TE 的 proj 依赖）。"""

    # ---------------------------------------------------------- 四动作
    def run_action(self, action, request, operation=None):
        if action not in WEIGHT_ACTIONS:
            raise OperationError("INVALID_REQUEST", "未知动作 %r" % (action,),
                                 400, False)
        with self._lock:
            units = self.unit_ids(request)
            self._require_group(units)
            self.registry.check_transition(action, self.MODEL_ID, units)
            if request.get("dry_run"):
                return self._dry_run(action, units)
            self._stage("%s %s" % (self.MODEL_ID, action), operation)
            if action == "disk2ram":
                return self._disk2ram(units, operation)
            if action == "ram2vram":
                return self._ram2vram(units, operation)
            if action == "vram2ram":
                return self._vram2ram(units, operation)
            return self._purge(units, operation)

    def _dry_run(self, action, units):
        records = {u: self.registry.unit_record(self.MODEL_ID, u)
                   for u in units}
        if action == "disk2ram":
            would_change = [u for u, r in records.items()
                            if r["state"] == "absent"]
        elif action == "purgefromram":
            would_change = [u for u, r in records.items()
                            if r["state"] != "absent"]
        else:
            would_change = list(units)
        noop = not would_change
        estimate = {
            "action": action,
            "model": self.MODEL_ID,
            "units": units,
            "would_change": would_change,
            "noop": noop,
            "estimated_checkpoint_bytes": sum(
                self.catalog.units[u].expected_bytes for u in units),
            "current": records,
            "note": "dry_run 不读权重、不构造对象、不搬设备",
        }
        return self.registry.summarize_operation(
            action, self.MODEL_ID, units, dry_run=True,
            extra={"estimate": estimate})

    def _disk2ram(self, units, operation=None):
        if self.obj is not None:
            return self._summary("disk2ram", units, completed=list(units),
                                 extra={"noop": True,
                                        "reason": "already loaded"})
        try:
            self._stage("%s 构造到 CPU" % self.MODEL_ID, operation)
            t0 = time.perf_counter()
            self.construct_cpu(operation)
            self.load_s = time.perf_counter() - t0
            audit = self._refresh()
            primary = audit["units"].get(self.PRIMARY_UNIT, {})
            if primary.get("device") != "cpu":
                raise OperationError(
                    "INTERNAL_STATE",
                    "%s 构造后主 unit 未落在 CPU: %s"
                    % (self.MODEL_ID, primary.get("device")), 500, False, audit)
        except Exception:
            self._mark_error()
            self.obj = None
            gc.collect()
            raise
        print("[%s] disk2ram 完成 %.1fs | %s"
              % (self.MODEL_ID, self.load_s, audit["bytes"]), flush=True)
        return self._summary("disk2ram", units, completed=list(units),
                             extra={"load_s": round(self.load_s, 3),
                                    "audit": audit})

    def _ram2vram(self, units, operation=None):
        if self.obj is None:
            raise OperationError(
                "MISSING_UNITS", "%s 未在 RAM 中，先 disk2ram" % self.MODEL_ID,
                409, False,
                {"missing": list(units),
                 "suggested": suggestions(self.MODEL_ID, "absent")})
        import torch
        self._stage("%s 权重 RAM -> VRAM" % self.MODEL_ID, operation)
        t0 = time.perf_counter()
        before = self._cuda_bytes(torch)
        try:
            self.move_to("cuda:0")
        except Exception as exc:
            self._mark_error()
            raise OperationError(
                "MOVE_FAILED", "%s 上卡失败: %s: %s"
                % (self.MODEL_ID, type(exc).__name__, exc), 500, True,
                {"actual": self._safe_state()}) from exc
        audit = self._refresh()
        primary = audit["units"].get(self.PRIMARY_UNIT, {})
        if primary.get("device") != "cuda:0":
            self._mark_error()
            raise OperationError(
                "INTERNAL_STATE",
                "%s 上卡后主 unit 仍为 %s（actual mixed）"
                % (self.MODEL_ID, primary.get("device")), 500, False, audit)
        elapsed = time.perf_counter() - t0
        after = self._cuda_bytes(torch)
        print("[%s] ram2vram %.3fs | cuda %.2f -> %.2f GB"
              % (self.MODEL_ID, elapsed, before / 1e9, after / 1e9), flush=True)
        return self._summary("ram2vram", units, completed=list(units),
                             extra={"elapsed_s": round(elapsed, 3),
                                    "cuda_bytes_before": before,
                                    "cuda_bytes_after": after,
                                    "audit": audit})

    def _vram2ram(self, units, operation=None):
        if self.obj is None:
            return self._summary("vram2ram", units, completed=list(units),
                                 extra={"noop": True, "reason": "not loaded"})
        import torch
        self._stage("%s 权重 VRAM -> RAM" % self.MODEL_ID, operation)
        t0 = time.perf_counter()
        before = self._cuda_bytes(torch)
        try:
            self.move_to("cpu")
            self.drop_caches()
        except Exception as exc:
            self._mark_error()
            raise OperationError(
                "MOVE_FAILED", "%s 下卡失败: %s: %s"
                % (self.MODEL_ID, type(exc).__name__, exc), 500, True,
                {"actual": self._safe_state()}) from exc
        audit = self._refresh()
        primary = audit["units"].get(self.PRIMARY_UNIT, {})
        if primary.get("device") != "cpu" or audit["cuda_bytes"] > 0:
            self._mark_error()
            raise OperationError(
                "INTERNAL_STATE",
                "%s 下卡后仍有 CUDA 驻留（actual mixed）" % self.MODEL_ID,
                500, False, audit)
        elapsed = time.perf_counter() - t0
        after = self._cuda_bytes(torch)
        print("[%s] vram2ram %.3fs | cuda %.2f -> %.2f GB"
              % (self.MODEL_ID, elapsed, before / 1e9, after / 1e9), flush=True)
        return self._summary("vram2ram", units, completed=list(units),
                             extra={"elapsed_s": round(elapsed, 3),
                                    "cuda_bytes_before": before,
                                    "cuda_bytes_after": after,
                                    "audit": audit})

    def _purge(self, units, operation=None):
        records = {u: self.registry.unit_record(self.MODEL_ID, u)
                   for u in units}
        if self.obj is None and all(
                r["state"] == "absent" and r["resident_bytes_cpu"] == 0
                and r["resident_bytes_cuda"] == 0
                for r in records.values()):
            return self._summary("purgefromram", units,
                                 completed=list(units), extra={"noop": True})
        import torch
        self._stage("%s 释放" % self.MODEL_ID, operation)
        t0 = time.perf_counter()
        before = self._cuda_bytes(torch)
        refs = self.release()
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        alive = [i for i, ref in enumerate(refs) if ref() is not None]
        after = self._cuda_bytes(torch)
        proof = {"weakrefs_alive": alive,
                 "cuda_bytes_before": before, "cuda_bytes_after": after}
        if alive:
            self._mark_error()
            raise OperationError(
                "PURGE_INCOMPLETE",
                "%s 对象仍被其它引用持有，拒绝谎报 absent" % self.MODEL_ID,
                500, True, {"actual": self._safe_state(), "proof": proof})
        for unit in units:
            self.registry.set_unit_absent(self.MODEL_ID, unit)
        elapsed = time.perf_counter() - t0
        print("[%s] purgefromram %.3fs | cuda %.2f -> %.2f GB"
              % (self.MODEL_ID, elapsed, before / 1e9, after / 1e9), flush=True)
        return self._summary("purgefromram", units, completed=list(units),
                             extra={"elapsed_s": round(elapsed, 3),
                                    "proof": proof})

    # ---------------------------------------------------------- 物理账
    def _refresh(self):
        inv = self.inventory()
        units = inv.get("units") or {}
        issues = list(inv.get("uninspectable") or [])
        audit = {
            "units": {uid: {"device": item.get("device"),
                            "cpu": int(item.get("cpu", 0)),
                            "cuda": int(item.get("cuda", 0))}
                      for uid, item in units.items()},
            "cpu_bytes": sum(int(i.get("cpu", 0)) for i in units.values()),
            "cuda_bytes": sum(int(i.get("cuda", 0)) for i in units.values()),
            "uninspectable": issues,
            "storage_digest": list(inv.get("storage_digest") or []),
        }
        audit["bytes"] = {uid: {"cpu": item["cpu"], "cuda": item["cuda"]}
                          for uid, item in audit["units"].items()}
        with self.registry.locked():
            if issues:
                self.registry.mark_error(
                    self.MODEL_ID, self.PRIMARY_UNIT,
                    "uninspectable tensors: %s" % issues[:3],
                    recovery=["检查量化包装/storage 展开后重新 disk2ram"])
                raise OperationError(
                    "UNINSPECTABLE_WEIGHTS",
                    "有无法审计的张量，拒绝宣称已加载", 500, False, audit)
            for uid, item in audit["units"].items():
                if item["device"] not in ("cpu", "cuda:0"):
                    self.registry.mark_error(
                        self.MODEL_ID, uid,
                        "参数跨设备: %s" % item["device"],
                        recovery=["先 vram2ram 或 purgefromram 修复"])
                    raise OperationError(
                        "MIXED_DEVICE", "%s.%s 参数跨设备（actual %s）"
                        % (self.MODEL_ID, uid, item["device"]),
                        500, False, audit)
                self.registry.set_unit_loaded(
                    self.MODEL_ID, uid, item["device"],
                    resident_bytes_cpu=item["cpu"],
                    resident_bytes_cuda=item["cuda"],
                    storage_digest=audit["storage_digest"])
        return audit

    @staticmethod
    def dominant_device(module):
        devices = set()
        for parameter in module.parameters():
            devices.add(str(parameter.device))
        if not devices:
            return "cpu"
        if all(d.startswith("cuda") for d in devices):
            return "cuda:0"
        if all(d == "cpu" for d in devices):
            return "cpu"
        return "mixed"

    @staticmethod
    def _cuda_bytes(torch):
        try:
            if torch.cuda.is_initialized():
                return int(torch.cuda.memory_allocated())
        except Exception:
            pass
        return 0

    # ---------------------------------------------------------- 杂项
    def _stage(self, message, operation=None):
        try:
            if operation is not None:
                operation.set_stage(message)
        except Exception:
            pass
        self.runtime.set_stage(message)

    def _summary(self, action, units, completed=(), extra=None):
        body = self.registry.summarize_operation(
            action, self.MODEL_ID, units, completed=completed, extra=extra)
        body["backend_state"] = self.state()
        return body

    def _mark_error(self):
        self.last_error = traceback.format_exc(limit=1).strip() or "unknown"
        try:
            with self.registry.locked():
                for unit in (self.catalog.unit_ids() if self.catalog else ()):
                    self.registry.mark_error(
                        self.MODEL_ID, unit, self.last_error,
                        recovery=["重新 disk2ram；已上卡则先 vram2ram"])
        except Exception:
            pass

    def _safe_state(self):
        try:
            return self.registry.model_state(self.MODEL_ID)
        except Exception as exc:
            return {"error": "%s: %s" % (type(exc).__name__, exc)}

    def _release_ref(self, value):
        """weakref.ref 对模块/普通对象都可用；失败返回一个永远存活的哨兵。"""
        try:
            return weakref.ref(value)
        except TypeError:
            return lambda: None
