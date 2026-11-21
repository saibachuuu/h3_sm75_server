"""artifact 存储：安全路径解析、原子落盘、引用保护、磁盘预算。

本模块分两层：
- **存储层（P0，纯标准库）**：``ArtifactStore`` 负责 ``<out_dir>/stage`` 下的
  安全路径解析、临时文件→原子改名、manifest、active 引用与磁盘预算。不 import
  torch，可在无 GPU 的解释器里单测。
- **序列化层（P3，lazy import torch/safetensors）**：cond / avlatent 的
  tensor 拆装；只有真正读写时才加载依赖。

安全模型（03 §7）：
- 只接受相对 ``stage`` 根的文件名；拒绝绝对路径、盘符、UNC（``\\\\host``）、
  反斜杠、``..``、NUL、以及百分号编码后的穿越（``%2e%2e`` / ``%2f`` / ``%5c``）；
- 用 ``realpath`` 后的包含关系判断，不做字符串前缀比较；
- 已存在路径上的 symlink / reparse point 一律拒绝（Windows junction 同样拦）。

错误统一 :class:`ArtifactError`，带 ``code`` / ``http_status``，主代理可映射到
``dit_svc/operations.py`` 的统一错误。
"""
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
import uuid

# ------------------------------------------------------------------ 常量

STAGE_DIRNAME = "stage"
MANIFEST_SUFFIX = ".json"
MAX_FILENAME_LEN = 200
DEFAULT_STAGE_QUOTA_BYTES = 32 * 1024 ** 3      # stage 目录总量上限 32 GiB
DISK_RESERVE_BYTES = 2 * 1024 ** 3              # 预留给系统/日志的余量 2 GiB

ARTIFACT_KINDS = ("cond", "latent", "video", "audio", "image")
ARTIFACT_ID_RE = re.compile(r"^(cond|latent)_[0-9a-f]{32}$")

_WINDOWS_RESERVED = frozenset(
    ["con", "prn", "aux", "nul"] +
    ["com%d" % i for i in range(1, 10)] +
    ["lpt%d" % i for i in range(1, 10)])


class ArtifactError(Exception):
    """artifact 层错误（路径/不存在/预算/结构）。"""

    def __init__(self, message, *, code="INVALID_REQUEST", http_status=400,
                 details=None):
        super(ArtifactError, self).__init__(message)
        self.code = code
        self.http_status = int(http_status)
        self.details = details

    def to_error(self):
        err = {"code": self.code, "message": str(self),
               "http_status": self.http_status, "retryable": False}
        if self.details is not None:
            err["details"] = self.details
        return {"ok": False, "error": err}


def _bad_path(message, path=None):
    details = {"path": path} if path is not None else None
    return ArtifactError(message, code="INVALID_PATH", http_status=400,
                         details=details)


# ------------------------------------------------------------------ 名称校验

def _percent_decode_fully(name, rounds=3):
    """反复解码到稳定，捕获多重编码的穿越。"""
    out = name
    for _ in range(rounds):
        try:
            nxt = urllib.parse.unquote(out, errors="strict")
        except (UnicodeDecodeError, ValueError):
            raise _bad_path("文件名百分号编码无法解码", out)
        if nxt == out:
            return out
        out = nxt
    return out


def validate_stage_name(name):
    """校验一个 stage 相对文件名，返回规范化后的名字。

    只允许单层文件名（不含目录分隔）；禁止绝对路径/盘符/UNC/反斜杠/
    ``..``/NUL/Windows 保留设备名/结尾点或空格。
    """
    if not isinstance(name, str) or not name:
        raise _bad_path("文件名必须是非空字符串")
    if len(name) > MAX_FILENAME_LEN:
        raise _bad_path("文件名过长（上限 %d）" % MAX_FILENAME_LEN, name)
    if "\x00" in name:
        raise _bad_path("文件名不能包含 NUL", name)

    decoded = _percent_decode_fully(name)
    if decoded != name:
        # 编码与原文不同本身就是歧义来源；直接要求客户端给未编码名
        raise _bad_path("文件名不能包含百分号编码（解码后为 %r）" % decoded, name)

    if "\\" in name:
        raise _bad_path("文件名不能包含反斜杠", name)
    if "/" in name:
        raise _bad_path("文件名不能包含目录分隔符", name)
    if name.startswith("~"):
        raise _bad_path("文件名不能以 ~ 开头", name)
    if ":" in name:
        raise _bad_path("文件名不能包含冒号（盘符/备用流）", name)
    if name in (".", "..") :
        raise _bad_path("文件名不能是 . 或 ..", name)
    if name.endswith(".") or name.endswith(" "):
        raise _bad_path("文件名不能以点或空格结尾", name)
    stem = name.split(".", 1)[0].lower()
    if stem in _WINDOWS_RESERVED:
        raise _bad_path("文件名使用了 Windows 保留设备名: %s" % stem, name)
    return name


def _assert_no_reparse(path, root):
    """逐段检查已存在部分不含 symlink / reparse point。"""
    rel = os.path.relpath(path, root)
    if rel == ".":
        return
    cur = root
    for part in rel.split(os.sep):
        cur = os.path.join(cur, part)
        if not os.path.lexists(cur):
            break
        if os.path.islink(cur):
            raise _bad_path("路径包含符号链接", cur)
        try:
            st = os.lstat(cur)
        except OSError:
            break
        attrs = getattr(st, "st_file_attributes", 0)
        if attrs & 0x400:  # FILE_ATTRIBUTE_REPARSE_POINT
            raise _bad_path("路径包含 reparse point", cur)


def safe_join(root, name):
    """把相对文件名安全解析到 ``root`` 下；一切逃逸都拒绝。"""
    validate_stage_name(name)
    root_real = os.path.realpath(root)
    candidate = os.path.join(root_real, name)
    _assert_no_reparse(candidate, root_real)
    resolved = os.path.realpath(candidate)
    try:
        common = os.path.commonpath([root_real, resolved])
    except ValueError:      # 不同盘符
        raise _bad_path("路径逃逸出 stage 根", name)
    if common != root_real:
        raise _bad_path("路径逃逸出 stage 根", name)
    return resolved


# ------------------------------------------------------------------ 原子写

def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return              # Windows 不支持目录 fsync；尽力而为
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path, data, *, mode="wb"):
    """同目录唯一临时文件 → fsync → ``os.replace`` 原子替换。"""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".part",
                               dir=directory)
    try:
        with os.fdopen(fd, mode) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(directory)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ ArtifactStore

class ArtifactStore(object):
    """``<out_dir>/stage`` 下的 artifact 仓库。

    并发约定：所有 manifest 写入与引用计数在 ``self._lock`` 内完成；文件本体
    用原子改名，读者要么看到旧文件要么看到完整新文件。
    """

    def __init__(self, out_dir, *, quota_bytes=DEFAULT_STAGE_QUOTA_BYTES,
                 disk_reserve_bytes=DISK_RESERVE_BYTES, ttl_s=24 * 3600):
        self.out_dir = os.path.realpath(out_dir)
        self.stage_dir = os.path.join(self.out_dir, STAGE_DIRNAME)
        os.makedirs(self.stage_dir, exist_ok=True)
        self.quota_bytes = int(quota_bytes)
        self.disk_reserve_bytes = int(disk_reserve_bytes)
        self.ttl_s = float(ttl_s)
        self._lock = threading.RLock()
        self._refs = {}          # artifact_id -> active 引用计数
        self._index = {}         # artifact_id -> manifest dict
        self._load_manifests()

    # -------------------------------------------------- 路径

    def path_for(self, name):
        return safe_join(self.stage_dir, name)

    def url_for(self, name):
        return "/outputs/%s/%s" % (STAGE_DIRNAME, validate_stage_name(name))

    def resolve(self, name, *, must_exist=True):
        """安全解析 stage 文件名；``must_exist`` 时校验普通文件。"""
        path = self.path_for(name)
        if must_exist:
            if not os.path.isfile(path):
                raise ArtifactError("artifact 文件不存在: %s" % name,
                                    code="ARTIFACT_NOT_FOUND", http_status=404)
            if os.path.islink(path):
                raise _bad_path("artifact 不能是符号链接", name)
        return path

    def resolve_any(self, ref, *, base_dir=None):
        """解析用户给的 file 引用。

        - 只允许 stage 内的相对名（base_dir 仅作提示，不给任意绝对路径）；
        - 绝对路径/盘符/UNC/反斜杠/穿越一律拒绝。
        """
        if not isinstance(ref, str) or not ref.strip():
            raise _bad_path("文件引用必须是非空字符串")
        ref = ref.strip()
        if os.path.isabs(ref) or re.match(r"^[A-Za-z]:", ref):
            raise _bad_path("不接受绝对路径，请用 stage 内的相对文件名", ref)
        if ref.startswith("\\\\") or ref.startswith("//"):
            raise _bad_path("不接受 UNC 路径", ref)
        if "\\" in ref:
            raise _bad_path("不接受反斜杠路径", ref)
        if ref.startswith("stage/"):
            ref = ref[len("stage/"):]
        if "/" in ref:
            raise _bad_path("只接受单层文件名（不带目录）", ref)
        return self.resolve(ref)

    # -------------------------------------------------- manifest / index

    def _manifest_path(self, artifact_id):
        return os.path.join(self.stage_dir, artifact_id + MANIFEST_SUFFIX)

    def _load_manifests(self):
        try:
            entries = os.listdir(self.stage_dir)
        except OSError:
            return
        for entry in entries:
            if not entry.endswith(MANIFEST_SUFFIX):
                continue
            artifact_id = entry[:-len(MANIFEST_SUFFIX)]
            if not ARTIFACT_ID_RE.match(artifact_id):
                continue
            try:
                with open(os.path.join(self.stage_dir, entry), "r",
                          encoding="utf-8") as f:
                    manifest = json.load(f)
            except (OSError, ValueError):
                continue
            if isinstance(manifest, dict) and manifest.get("id") == artifact_id:
                self._index[artifact_id] = manifest

    def new_id(self, kind):
        if kind not in ("cond", "latent"):
            raise ArtifactError("未知 artifact kind: %r" % (kind,),
                                code="INVALID_REQUEST", http_status=400)
        return "%s_%s" % (kind, uuid.uuid4().hex)

    def get(self, artifact_id):
        with self._lock:
            manifest = self._index.get(artifact_id)
        if manifest is None:
            raise ArtifactError("artifact 不存在: %s" % artifact_id,
                                code="ARTIFACT_NOT_FOUND", http_status=404)
        return dict(manifest)

    def public(self, artifact_id):
        manifest = self.get(artifact_id)
        return {
            "id": manifest["id"],
            "kind": manifest.get("kind"),
            "format": manifest.get("format"),
            "bytes": manifest.get("bytes"),
            "sha256": manifest.get("sha256"),
            "created_at": manifest.get("created_at"),
            "filename": manifest.get("filename"),
            "url": self.url_for(manifest["filename"]),
            "metadata": manifest.get("metadata") or {},
        }

    def list(self, *, limit=50):
        with self._lock:
            items = sorted(self._index.values(),
                           key=lambda m: m.get("created_at") or 0,
                           reverse=True)[:int(limit)]
        return [self.public(m["id"]) for m in items]

    def _write_manifest(self, manifest):
        data = json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8")
        atomic_write_bytes(self._manifest_path(manifest["id"]), data)
        with self._lock:
            self._index[manifest["id"]] = manifest

    # -------------------------------------------------- 预算

    def free_bytes(self):
        usage = shutil.disk_usage(self.stage_dir)
        return int(usage.free)

    def stage_usage_bytes(self):
        total = 0
        try:
            for entry in os.scandir(self.stage_dir):
                if entry.is_file(follow_symlinks=False):
                    try:
                        total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            return 0
        return total

    def budget(self, expected_bytes):
        """检查新写入 ``expected_bytes`` 是否有空间。

        返回 dict（fits / reasons / numbers），不抛错；调用方决定拒绝还是
        dry-run 报告。
        """
        expected = max(0, int(expected_bytes))
        free = self.free_bytes()
        usage = self.stage_usage_bytes()
        quota_left = max(0, self.quota_bytes - usage)
        usable = max(0, free - self.disk_reserve_bytes)
        fits = expected <= quota_left and expected <= usable
        reasons = []
        if expected > quota_left:
            reasons.append("stage 配额不足（需要 %d，剩余 %d）"
                           % (expected, quota_left))
        if expected > usable:
            reasons.append("磁盘余量不足（需要 %d，可用 %d，预留 %d）"
                           % (expected, usable, self.disk_reserve_bytes))
        return {
            "expected_bytes": expected,
            "stage_usage_bytes": usage,
            "stage_quota_bytes": self.quota_bytes,
            "stage_quota_remaining_bytes": quota_left,
            "disk_free_bytes": free,
            "disk_reserve_bytes": self.disk_reserve_bytes,
            "disk_usable_bytes": usable,
            "fits": fits,
            "reasons": reasons,
            "estimate_quality": "exact_expected_bytes",
        }

    def ensure_budget(self, expected_bytes):
        info = self.budget(expected_bytes)
        if not info["fits"]:
            raise ArtifactError("; ".join(info["reasons"]),
                                code="RESOURCE_EXHAUSTED", http_status=503,
                                details=info)
        return info

    # -------------------------------------------------- 生命周期

    def acquire(self, artifact_id):
        """consumer 取引用；期间清理不得删除该文件。"""
        self.get(artifact_id)      # 不存在直接 404
        with self._lock:
            self._refs[artifact_id] = self._refs.get(artifact_id, 0) + 1
        return artifact_id

    def release(self, artifact_id):
        with self._lock:
            n = self._refs.get(artifact_id, 0)
            if n <= 1:
                self._refs.pop(artifact_id, None)
            else:
                self._refs[artifact_id] = n - 1

    def refcount(self, artifact_id):
        with self._lock:
            return self._refs.get(artifact_id, 0)

    def delete(self, artifact_id, *, force=False):
        """删除 artifact；仍被引用且非 force 时拒绝。"""
        manifest = self.get(artifact_id)
        if not force and self.refcount(artifact_id):
            raise ArtifactError("artifact 正被引用，暂不能删除: %s" % artifact_id,
                                code="ACTIVE_BINDING", http_status=409)
        filename = manifest["filename"]
        path = self.path_for(filename)
        for p in (path, self._manifest_path(artifact_id)):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
        with self._lock:
            self._index.pop(artifact_id, None)
        return True

    def cleanup(self, *, now=None, limit=None):
        """清理过期且未被引用的 artifact。返回已删 ID 列表。"""
        now = time.time() if now is None else float(now)
        removed = []
        with self._lock:
            candidates = list(self._index.values())
        for manifest in sorted(candidates, key=lambda m: m.get("created_at") or 0):
            aid = manifest["id"]
            created = float(manifest.get("created_at") or 0)
            if now - created < self.ttl_s:
                continue
            if self.refcount(aid):
                continue
            try:
                self.delete(aid)
                removed.append(aid)
            except ArtifactError:
                continue
            if limit is not None and len(removed) >= int(limit):
                break
        return removed

    # -------------------------------------------------- 落盘入口

    def commit(self, kind, filename, data, *, metadata=None, fmt=None):
        """写入一个 artifact（唯一文件名 + 原子改名 + manifest）。

        预算在写之前按 ``len(data)`` 检查；失败不留半写文件。
        """
        return self.commit_stream(
            kind, filename, lambda path: atomic_write_bytes(path, data),
            metadata=metadata, fmt=fmt, expected_bytes=len(data))

    def commit_stream(self, kind, filename, writer, *, metadata=None, fmt=None,
                      expected_bytes=None):
        """用 ``writer(tmp_path)`` 落盘大文件（如 safetensors），再原子改名。

        writer 必须把完整内容写到给定路径；任何异常都不留半写产物。
        ``expected_bytes`` 用于预算预检（None 时按 0 预检，写前再复核）。
        """
        validate_stage_name(filename)
        self.ensure_budget(expected_bytes or 0)
        artifact_id = self.new_id(kind)
        path = self.path_for(filename)
        if os.path.lexists(path):
            raise ArtifactError("文件名已被占用: %s" % filename,
                                code="INVALID_REQUEST", http_status=409)
        directory = os.path.dirname(path) or "."
        fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".part", dir=directory)
        os.close(fd)
        try:
            writer(tmp)
            with open(tmp, "r+b") as f:
                os.fsync(f.fileno())
            size = os.path.getsize(tmp)
            if expected_bytes and size > expected_bytes * 2 + (1 << 20):
                raise ArtifactError(
                    "写入结果 %d 字节远超预估 %d，拒绝提交（疑似异常膨胀）"
                    % (size, expected_bytes),
                    code="RESOURCE_EXHAUSTED", http_status=503)
            os.replace(tmp, path)
            _fsync_dir(directory)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        manifest = {
            "id": artifact_id,
            "kind": kind,
            "format": fmt,
            "filename": filename,
            "bytes": size,
            "sha256": _sha256_file(path),
            "created_at": time.time(),
            "metadata": dict(metadata or {}),
        }
        self._write_manifest(manifest)
        return self.public(artifact_id)


# ================================================================== 序列化层（P3）
#
# torch / safetensors 全部 lazy import：本文件在无 GPU、无 torch 的解释器里也能
# 导入并跑路径层单测。
#
# 真实 ComfyUI 容器（remote comfy/nested_tensor.py）：
#     class NestedTensor:
#         def __init__(self, tensors): self.tensors = list(tensors); self.is_nested = True
#         def unbind(self): return self.tensors
# 且 comfy_extras/nodes_minimax_h3.py 里 `_av_latent` 固定是
#     NestedTensor((video[B,24,T,H//16,W//16], audio[B,32,2,T40]))
# 所以这里按 `.tensors` 列表 + shape 判别来处理，不使用 torch.nested。

FORMAT_COND = "h3.cond.v1"
FORMAT_AVLATENT = "h3.avlatent.v1"

COND_REQUIRED_KEYS = ("cond", "minimax_token_tags")
COND_OPTIONAL_KEYS = ("attention_mask", "pooled_output")
AVLATENT_CORE_KEYS = ("video", "audio")
AVLATENT_OPTIONAL_KEYS = ("noise_mask", "noise_mask_video", "noise_mask_audio")
AVLATENT_REQUIRED_META = ("format", "requested_length", "native_fps")
NATIVE_FPS = 24

# video: [B,24,T,H,W] ；audio: [B,32,2,T]
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_STREAMS = 2

_PROBE_CHUNK_ELEMS = 1 << 22


def _torch():
    import torch
    return torch


def _safetensors_mod():
    try:
        import safetensors                            # noqa: F401
    except ImportError as e:                          # pragma: no cover
        raise ArtifactError("safetensors 未安装：%s" % e,
                            code="ENV_NOT_READY", http_status=503)
    return safetensors


def _to_cpu_contiguous(tensor, *, label):
    """原 dtype 的 CPU contiguous 副本；不做 dtype 统一。"""
    if tensor is None:
        raise ArtifactError("%s 是 None" % label, code="INVALID_ARTIFACT",
                            http_status=422)
    if not hasattr(tensor, "detach"):
        raise ArtifactError("%s 不是 Tensor: %r" % (label, type(tensor)),
                            code="INVALID_ARTIFACT", http_status=422)
    out = tensor.detach().to(device="cpu", copy=True).contiguous()
    if out.layout != _torch().strided:
        raise ArtifactError("%s 不是 strided tensor（layout=%s）"
                            % (label, out.layout),
                            code="INVALID_ARTIFACT", http_status=422)
    return out


# ------------------------------------------------------------- 容器适配

def _classify_mask(tensor):
    """noise_mask 的 video/audio 块：只按维度区分（5D=video，4D=audio）。"""
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) not in (4, 5):
        raise ArtifactError("noise_mask tensor 维度不支持: %r"
                            % (tuple(shape) if shape is not None else tensor,),
                            code="INVALID_ARTIFACT", http_status=422)
    return "video" if len(shape) == 5 else "audio"


def split_noise_mask(noise_mask, *, split_container=None):
    """把 noise_mask 容器拆成 ``(video, audio)``；不套用 latent 通道判别。"""
    split = split_container or default_split_container
    video, audio, order = _split_pair(noise_mask, split, _classify_mask,
                                      label="noise_mask")
    return video, audio, order


def _split_pair(container, split, classify, *, label):
    """通用：从容器取出两个 tensor 并按 ``classify`` 定位 video/audio。"""
    if hasattr(container, "tensors") and isinstance(
            getattr(container, "tensors"), (list, tuple)):
        items = list(container.tensors)
        order_hint = None
    elif isinstance(container, dict):
        if set(container) != {"video", "audio"}:
            raise ArtifactError(
                "%s 映射必须且只能包含 video/audio，实际: %s"
                % (label, ", ".join(sorted(container))),
                code="INVALID_ARTIFACT", http_status=422)
        return container["video"], container["audio"], "video,audio"
    elif isinstance(container, (list, tuple)):
        items = list(container)
        order_hint = None
    else:
        raise ArtifactError("%s 未知容器类型: %s（需要 .tensors / "
                            "{video,audio} / 二元序列）" % (label, type(container)),
                            code="INVALID_ARTIFACT", http_status=422)
    if len(items) != 2:
        raise ArtifactError("%s 容器必须恰好 2 个 tensor，实际 %d 个"
                            % (label, len(items)),
                            code="INVALID_ARTIFACT", http_status=422)
    kinds = [classify(t) for t in items]
    if sorted(kinds) != ["audio", "video"]:
        raise ArtifactError("%s 两个 tensor 无法区分为 video/audio：%s"
                            % (label, kinds), code="INVALID_ARTIFACT",
                            http_status=422)
    video = items[kinds.index("video")]
    audio = items[kinds.index("audio")]
    return video, audio, ",".join(kinds)


def default_split_container(container):
    """把 latent 容器拆成 ``(video, audio, trailer)``。

    支持三种真实可能形状，且都只按“结构”识别，不猜：
    1. Comfy NestedTensor：``.tensors`` 为长度 2 的 list；
    2. 映射：``{"video": T, "audio": T}``；
    3. 二元 tuple / list。

    返回 ``(video, audio, order)``，order 记录容器内原始顺序（"video,audio" /
    "audio,video"），重建时按原序放回，避免隐式交换。
    """
    items = None
    if hasattr(container, "tensors") and isinstance(
            getattr(container, "tensors"), (list, tuple)):
        items = list(container.tensors)
    elif isinstance(container, dict):
        if set(container) == {"video", "audio"}:
            return container["video"], container["audio"], "video,audio"
        raise ArtifactError(
            "latent 映射必须且只能包含 video/audio，实际: %s"
            % ", ".join(sorted(container)),
            code="INVALID_ARTIFACT", http_status=422)
    elif isinstance(container, (list, tuple)):
        items = list(container)

    if items is None:
        raise ArtifactError("未知 latent 容器类型: %s（需要 .tensors / "
                            "{video,audio} / 二元序列）" % type(container),
                            code="INVALID_ARTIFACT", http_status=422)
    if len(items) != 2:
        raise ArtifactError("latent 容器必须恰好 2 个 tensor，实际 %d 个"
                            % len(items),
                            code="INVALID_ARTIFACT", http_status=422)

    kinds = [_classify_av(t) for t in items]
    if sorted(kinds) != ["audio", "video"]:
        raise ArtifactError("latent 两个 tensor 无法按 shape 区分为 video/audio：%s"
                            % (kinds,), code="INVALID_ARTIFACT",
                            http_status=422)
    video = items[kinds.index("video")]
    audio = items[kinds.index("audio")]
    return video, audio, ",".join(kinds)


def _classify_av(tensor):
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) not in (4, 5):
        raise ArtifactError("latent tensor 维度不支持: %r"
                            % (tuple(shape) if shape is not None else tensor,),
                            code="INVALID_ARTIFACT", http_status=422)
    dims = [int(d) for d in shape]
    if len(dims) == 5 and dims[1] == VIDEO_CHANNELS:
        return "video"
    if len(dims) == 4 and dims[1] == AUDIO_CHANNELS and dims[2] == AUDIO_STREAMS:
        return "audio"
    raise ArtifactError(
        "latent tensor shape %s 既不是 video[B,24,T,H,W] 也不是 audio[B,32,2,T]"
        % (tuple(dims),), code="INVALID_ARTIFACT", http_status=422)


def default_build_container(video, audio, order="video,audio"):
    """按记录的容器顺序重建（与原容器同类则可直接复用其类）。"""
    pair = {"video": video, "audio": audio}
    tensors = [pair[name] for name in order.split(",")]
    for cls in _container_classes():
        try:
            obj = cls(tuple(tensors))
            if list(getattr(obj, "tensors", [])) == tensors:
                return obj
        except Exception:            # noqa: BLE001 — 只是候选
            continue
    return tuple(tensors)


def _container_classes():
    """优先用真实 Comfy NestedTensor，其次尝试 comfy.nested_tensor。"""
    found = []
    import sys
    mod = sys.modules.get("comfy_extras.nested_tensor")
    if mod is not None and hasattr(mod, "NestedTensor"):
        found.append(mod.NestedTensor)
    mod = sys.modules.get("comfy.nested_tensor")
    if mod is not None and hasattr(mod, "NestedTensor"):
        found.append(mod.NestedTensor)
    return found


# ------------------------------------------------------------- latent 拆装

def split_av_latent(latent, *, split_container=None):
    """把采样器输出的 latent 拆成 ``({tensor keys}, order, extra_meta)``。

    ``latent`` 是 Comfy latent 字典（含 ``samples``，可能有 ``noise_mask``）。
    未知键直接报错，不静默丢弃。
    """
    if not isinstance(latent, dict):
        raise ArtifactError("latent 必须是 dict（含 samples）",
                            code="INVALID_ARTIFACT", http_status=422)
    if "samples" not in latent:
        raise ArtifactError("latent 缺少 samples 键",
                            code="INVALID_ARTIFACT", http_status=422)
    unsupported = sorted(k for k in latent
                         if k != "samples" and k not in ("noise_mask", "batch_index"))
    if unsupported:
        raise ArtifactError(
            "latent 含未支持的字段，拒绝保存（避免静默丢数据）: %s"
            % ", ".join(unsupported),
            code="INVALID_ARTIFACT", http_status=422,
            details={"unsupported_fields": unsupported})

    split = split_container or default_split_container
    video, audio, order = split(latent["samples"])
    out = {
        "video": _to_cpu_contiguous(video, label="video"),
        "audio": _to_cpu_contiguous(audio, label="audio"),
    }

    noise_mask = latent.get("noise_mask")
    if noise_mask is not None:
        if hasattr(noise_mask, "tensors") or isinstance(noise_mask, (list, tuple)):
            vmask, amask, _ = split_noise_mask(noise_mask, split_container=split_container)
            out["noise_mask_video"] = _to_cpu_contiguous(vmask, label="noise_mask_video")
            out["noise_mask_audio"] = _to_cpu_contiguous(amask, label="noise_mask_audio")
        else:
            out["noise_mask"] = _to_cpu_contiguous(noise_mask, label="noise_mask")

    extra = {"container_order": order,
             "container_type": type(latent["samples"]).__name__}
    if latent.get("batch_index") is not None:
        extra["batch_index"] = str(latent["batch_index"])[:200]
    return out, extra


def _metadata_strings(meta):
    out = {}
    for key, value in (meta or {}).items():
        if value is None:
            continue
        if isinstance(value, str):
            out[str(key)] = value
        elif isinstance(value, bool):
            out[str(key)] = "true" if value else "false"
        elif isinstance(value, (int, float)):
            out[str(key)] = str(value)
        else:
            out[str(key)] = json.dumps(value, ensure_ascii=False)
    return out


def save_av_latent(store, samples, metadata, *, filename=None,
                   split_container=None):
    """把 latent 存成 ``h3.avlatent.v1`` safetensors 并登记 artifact。

    ``metadata`` 必须包含 format=FORMAT_AVLATENT、requested_length、native_fps。
    """
    if not isinstance(metadata, dict):
        raise ArtifactError("metadata 必须是 dict", code="INVALID_REQUEST",
                            http_status=400)
    missing = [k for k in AVLATENT_REQUIRED_META if k not in metadata]
    if missing:
        raise ArtifactError("latent metadata 缺少必需字段: %s" % ", ".join(missing),
                            code="INVALID_REQUEST", http_status=400)
    if metadata["format"] != FORMAT_AVLATENT:
        raise ArtifactError("latent metadata.format 必须是 %s" % FORMAT_AVLATENT,
                            code="INVALID_REQUEST", http_status=400)
    try:
        fps = float(metadata["native_fps"])
        req_len = int(metadata["requested_length"])
    except (TypeError, ValueError):
        raise ArtifactError("native_fps / requested_length 必须是数字",
                            code="INVALID_REQUEST", http_status=400)
    if fps != NATIVE_FPS:
        raise ArtifactError("首版 latent 只支持 native_fps=24，收到 %s" % fps,
                            code="INVALID_REQUEST", http_status=400)
    if req_len < 1:
        raise ArtifactError("requested_length 必须 ≥1", code="INVALID_REQUEST",
                            http_status=400)

    tensors, extra = split_av_latent(samples, split_container=split_container)
    meta = dict(metadata)
    meta.update(extra)
    meta["video_shape"] = json.dumps([int(d) for d in tensors["video"].shape])
    meta["audio_shape"] = json.dumps([int(d) for d in tensors["audio"].shape])
    meta["dtypes"] = json.dumps({k: str(v.dtype) for k, v in tensors.items()})
    meta.setdefault("native_fps", NATIVE_FPS)

    expected = sum(t.numel() * t.element_size() for t in tensors.values())
    name = filename or "%s.safetensors" % uuid.uuid4().hex

    def _writer(tmp):
        from safetensors.torch import save_file
        save_file(tensors, tmp, metadata=_metadata_strings(meta))

    return store.commit_stream(
        "latent", name, _writer, metadata=_metadata_strings(meta),
        fmt=FORMAT_AVLATENT, expected_bytes=expected)


def load_av_latent(store, name, *, build_container=None):
    """读回 ``h3.avlatent.v1``：返回 ``(latent_dict, metadata)``。

    重建容器时按保存的 ``container_order`` 放回原序；不自动补未知字段。
    """
    path = store.resolve(name)
    from safetensors import safe_open
    tensors = {}
    md = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        md = dict(f.metadata() or {})
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    if md.get("format") != FORMAT_AVLATENT:
        raise ArtifactError("不是 %s 文件（format=%r）"
                            % (FORMAT_AVLATENT, md.get("format")),
                            code="INVALID_ARTIFACT", http_status=422)
    missing = [k for k in AVLATENT_CORE_KEYS if k not in tensors]
    if missing:
        raise ArtifactError("latent 文件缺少 tensor: %s" % ", ".join(missing),
                            code="INVALID_ARTIFACT", http_status=422)
    unknown = sorted(k for k in tensors
                     if k not in AVLATENT_CORE_KEYS + AVLATENT_OPTIONAL_KEYS)
    if unknown:
        raise ArtifactError("latent 文件含未支持 tensor 字段: %s"
                            % ", ".join(unknown), code="INVALID_ARTIFACT",
                            http_status=422)
    order = md.get("container_order") or "video,audio"
    if order not in ("video,audio", "audio,video"):
        raise ArtifactError("container_order 非法: %r" % order,
                            code="INVALID_ARTIFACT", http_status=422)
    build = build_container or default_build_container
    latent = {"samples": build(tensors["video"], tensors["audio"], order)}
    if "noise_mask_video" in tensors:
        latent["noise_mask"] = build(tensors["noise_mask_video"],
                                     tensors["noise_mask_audio"], order)
    elif "noise_mask" in tensors:
        latent["noise_mask"] = tensors["noise_mask"]
    md["requested_length"] = int(md.get("requested_length") or 0)
    md["native_fps"] = float(md.get("native_fps") or NATIVE_FPS)
    return latent, md


# ------------------------------------------------------------- cond 拆装

def split_cond(cond_data, *, extensions=()):
    """整理 cond 负载；``minimax_token_tags=None`` 必须显式决策，不静默丢弃。"""
    if not isinstance(cond_data, dict):
        raise ArtifactError("cond 数据必须是 dict", code="INVALID_ARTIFACT",
                            http_status=422)
    allowed = set(COND_REQUIRED_KEYS) | set(COND_OPTIONAL_KEYS) | set(extensions)
    unknown = sorted(k for k in cond_data if k not in allowed)
    if unknown:
        raise ArtifactError("cond 含未支持字段，拒绝保存: %s" % ", ".join(unknown),
                            code="INVALID_ARTIFACT", http_status=422,
                            details={"unsupported_fields": unknown})
    out = {}
    for key in COND_REQUIRED_KEYS:
        if key not in cond_data:
            raise ArtifactError("cond 缺少必需字段: %s" % key,
                                code="INVALID_ARTIFACT", http_status=422)
    out["cond"] = _to_cpu_contiguous(cond_data["cond"], label="cond")
    tags = cond_data["minimax_token_tags"]
    if tags is None:
        raise ArtifactError(
            "minimax_token_tags 为 None：不能写出缺少必需 key 的“成功”cond 文件",
            code="INVALID_ARTIFACT", http_status=422,
            details={"field": "minimax_token_tags", "reason": "none"})
    out["minimax_token_tags"] = _to_cpu_contiguous(tags, label="minimax_token_tags")
    # cond 通常是 [B, T, C]，tags 是 [B, T]；按 tags 的维度比较前缀
    tag_shape = tuple(out["minimax_token_tags"].shape)
    cond_shape = tuple(out["cond"].shape)
    if cond_shape[:len(tag_shape)] != tag_shape:
        raise ArtifactError(
            "minimax_token_tags shape %s 与 cond shape %s 的 token 维度不一致"
            % (tag_shape, cond_shape),
            code="INVALID_ARTIFACT", http_status=422)
    for key in COND_OPTIONAL_KEYS:
        if cond_data.get(key) is not None:
            out[key] = _to_cpu_contiguous(cond_data[key], label=key)
    return out


def save_cond(store, cond_data, metadata, *, filename=None, extensions=(),
              synthesize_tags=None):
    """保存 ``h3.cond.v1``。

    ``synthesize_tags``：只有已被验证的适配器才能提供的等价 tags 生成器；给了
    才允许把 tags=None 转成等价量，并在 metadata 里标明来源。默认拒绝。
    """
    if not isinstance(metadata, dict):
        raise ArtifactError("metadata 必须是 dict", code="INVALID_REQUEST",
                            http_status=400)
    md = dict(metadata)
    md["format"] = md.get("format") or FORMAT_COND
    if md["format"] != FORMAT_COND:
        raise ArtifactError("cond metadata.format 必须是 %s" % FORMAT_COND,
                            code="INVALID_REQUEST", http_status=400)

    data = dict(cond_data or {})
    if data.get("minimax_token_tags", "missing") is None and synthesize_tags is not None:
        tags = synthesize_tags(data.get("cond"))
        data["minimax_token_tags"] = tags
        md["token_tags_source"] = "synthesized"

    tensors = split_cond(data, extensions=extensions)
    if "token_tags_source" not in md:
        md["token_tags_source"] = "encoder"

    expected = sum(t.numel() * t.element_size() for t in tensors.values())
    name = filename or "%s.safetensors" % uuid.uuid4().hex

    def _writer(tmp):
        from safetensors.torch import save_file
        save_file(tensors, tmp, metadata=_metadata_strings(md))

    return store.commit_stream(
        "cond", name, _writer, metadata=_metadata_strings(md),
        fmt=FORMAT_COND, expected_bytes=expected)


def load_cond(store, name, *, extensions=()):
    """读回 cond safetensors：返回 ``(tensors dict, metadata)``。

    两个必需 key 缺失即报错；legacy 文件（无 h3.cond.v1 metadata）仍然接受，
    只是 ``metadata["format"]`` 为 None。
    """
    path = store.resolve(name)
    from safetensors import safe_open
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        md = dict(f.metadata() or {})
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    allowed = set(COND_REQUIRED_KEYS) | set(COND_OPTIONAL_KEYS) | set(extensions)
    unknown = sorted(k for k in tensors if k not in allowed)
    if unknown:
        raise ArtifactError("cond 文件含未支持 tensor: %s" % ", ".join(unknown),
                            code="INVALID_ARTIFACT", http_status=422)
    missing = [k for k in COND_REQUIRED_KEYS if k not in tensors]
    if missing:
        raise ArtifactError("cond 文件缺少必需 tensor: %s" % ", ".join(missing),
                            code="INVALID_ARTIFACT", http_status=422)
    return tensors, md


# ------------------------------------------------------------- 数值探针

def probe_tensor_chunked(tensor, *, chunk_elems=_PROBE_CHUNK_ELEMS):
    """有界分块的 NaN/Inf/min/max 探针，不新增整块 fp32 副本。"""
    torch = _torch()
    t = tensor.detach()
    numel = int(t.numel())
    if numel == 0:
        return {"min": None, "max": None, "nan": 0, "inf": 0, "elements": 0}
    if t.is_complex():
        t = t.real
    elif not t.is_floating_point():
        t = t.float()
    flat = t.reshape(-1)
    nan = 0
    inf = 0
    vmin = None
    vmax = None
    for start in range(0, numel, max(1, int(chunk_elems))):
        block = flat[start:start + chunk_elems]
        nan += int(torch.isnan(block).sum())
        inf += int(torch.isinf(block).sum())
        finite = block[torch.isfinite(block)]
        if finite.numel():
            bmin = float(finite.min())
            bmax = float(finite.max())
            vmin = bmin if vmin is None else min(vmin, bmin)
            vmax = bmax if vmax is None else max(vmax, bmax)
    return {"min": vmin, "max": vmax, "nan": nan, "inf": inf,
            "elements": numel}


def probe_av_latent(tensors):
    """对每个 tensor 分块探针；任一 NaN/Inf 整体标记 ok=False。"""
    out = {}
    total_nan = 0
    total_inf = 0
    for key in sorted(tensors):
        info = probe_tensor_chunked(tensors[key])
        out[key] = info
        total_nan += info["nan"]
        total_inf += info["inf"]
    out["ok"] = not (total_nan or total_inf)
    return out


# ------------------------------------------------------------- 时间轴（04 §6）

# 与 comfy_extras/nodes_minimax_h3.py 的常量一致（纯数值，不 import 该模块）
H3_FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_GRID = 17
FRAME_GRID_OFFSET = 5


def align_frame_count(length):
    """把请求帧数 snap 到 17k+5 网格（与 H3 节点的 align_frame_count 一致）。"""
    n = max(5, int(length))
    while n % FRAME_GRID != FRAME_GRID_OFFSET:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // FRAME_GRID) * 5 + 2


def plan_temporal(length):
    """返回 ``{requested_length, frame_count, latent_t, audio_t, seconds}``。

    ``requested_length`` 是用户要的帧数；``frame_count`` 是网格对齐后的真实
    解码帧数；``latent_t`` 只是 latent 的时间维，**不能**用来自行推断片长。
    """
    requested = int(length)
    if requested < 1:
        raise ArtifactError("requested_length 必须 ≥1", code="INVALID_REQUEST",
                            http_status=400)
    frame_count = align_frame_count(requested)
    return {
        "requested_length": requested,
        "frame_count": frame_count,
        "latent_t": video_latent_t(frame_count),
        "audio_t": int(round((frame_count / float(H3_FPS)) * AUDIO_LATENT_FPS)),
        "seconds": frame_count / float(H3_FPS),
    }


def decode_trim(decoded_frames, metadata):
    """按 artifact metadata 计算视频裁帧长度。

    ``decoded_frames`` 是 VAE 解码后的实际帧数（int 或带 shape[0] 的张量）。
    规则：以 metadata.requested_length 为准；解码结果不足即报错（不静默变短），
    多于 requested_length 时裁掉尾部 padding。绝不从 latent_t 猜片长。
    """
    if not isinstance(metadata, dict):
        raise ArtifactError("metadata 必须是 dict", code="INVALID_ARTIFACT",
                            http_status=422)
    if "requested_length" not in metadata:
        raise ArtifactError("latent metadata 缺少 requested_length，无法确定片长",
                            code="INVALID_ARTIFACT", http_status=422)
    try:
        requested = int(metadata["requested_length"])
    except (TypeError, ValueError):
        raise ArtifactError("metadata.requested_length 非法",
                            code="INVALID_ARTIFACT", http_status=422)
    if requested < 1:
        raise ArtifactError("metadata.requested_length 必须 ≥1",
                            code="INVALID_ARTIFACT", http_status=422)
    total = int(decoded_frames.shape[0]) if hasattr(decoded_frames, "shape") \
        else int(decoded_frames)
    frame_count = align_frame_count(requested)
    if total < requested:
        raise ArtifactError(
            "解码输出只有 %d 帧，少于 requested_length=%d（不做静默截断）"
            % (total, requested), code="EXECUTION_FAILED", http_status=500)
    return {"requested_length": requested, "frame_count": frame_count,
            "decoded_frames": total, "keep": requested,
            "trim_tail": total - requested}

