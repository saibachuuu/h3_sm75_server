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
