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

_ALLOWED_MANIFEST_KEYS = frozenset(
    ["id", "kind", "format", "filename", "bytes", "sha256", "created_at",
     "metadata"])
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _is_reparse_or_link(path):
    """路径自身是 symlink / junction / 其它 reparse point。"""
    try:
        if os.path.islink(path):
            return True
        st = os.lstat(path)
    except OSError:
        return False
    return bool(getattr(st, "st_file_attributes", 0)
                & _FILE_ATTRIBUTE_REPARSE_POINT)


class ArtifactStore(object):
    """``<out_dir>/stage`` 下的 artifact 仓库。

    并发约定：manifest 索引、引用计数、配额预留都在 ``self._lock`` 内完成；
    文件本体用唯一临时文件 + 原子改名，读者要么看到旧文件要么看到完整新文件。
    """

    def __init__(self, out_dir, *, quota_bytes=DEFAULT_STAGE_QUOTA_BYTES,
                 disk_reserve_bytes=DISK_RESERVE_BYTES, ttl_s=24 * 3600):
        self.out_dir = os.path.realpath(out_dir)
        self.stage_dir = os.path.join(self.out_dir, STAGE_DIRNAME)
        os.makedirs(self.stage_dir, exist_ok=True)
        self._validate_stage_root()
        self.quota_bytes = int(quota_bytes)
        self.disk_reserve_bytes = int(disk_reserve_bytes)
        self.ttl_s = float(ttl_s)
        self._lock = threading.RLock()
        self._refs = {}          # artifact_id -> active 引用计数
        self._index = {}         # artifact_id -> manifest dict
        self._pending = {}       # reservation token -> 预留字节
        self._load_manifests()

    def _validate_stage_root(self):
        """stage 根自身不能是 symlink/junction，且 realpath 必须留在 out_dir 内。

        否则 ``realpath(stage)`` 会落到 out_dir 之外，而 ``safe_join`` 以它为新
        根，包含关系检查反而全部“通过”。
        """
        if _is_reparse_or_link(self.stage_dir):
            raise ArtifactError(
                "stage 目录不能是符号链接/junction/reparse point: %s"
                % self.stage_dir, code="INVALID_PATH", http_status=400)
        real = os.path.realpath(self.stage_dir)
        try:
            common = os.path.commonpath([self.out_dir, real])
        except ValueError:
            common = None
        if common != self.out_dir:
            raise ArtifactError(
                "stage 目录 realpath 逃逸出 out_dir（%s -> %s）"
                % (self.stage_dir, real), code="INVALID_PATH", http_status=400)
        _assert_no_reparse(real, self.out_dir)

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
        """只把通过 schema/文件一致性校验的 manifest 收入索引。

        防任意 JSON 进入索引：校验 ID 与文件名一致、schema 白名单、文件存在且
        不是 reparse、大小与 manifest 相符；不跟随 manifest 自身的 symlink。
        """
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
            mpath = os.path.join(self.stage_dir, entry)
            if _is_reparse_or_link(mpath) or not os.path.isfile(mpath):
                continue
            try:
                if os.path.getsize(mpath) > 1024 * 1024:
                    continue
                with open(mpath, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except (OSError, ValueError):
                continue
            if self._manifest_is_valid(manifest, artifact_id):
                self._index[artifact_id] = manifest

    def _manifest_is_valid(self, manifest, artifact_id):
        if not isinstance(manifest, dict):
            return False
        if set(manifest) - _ALLOWED_MANIFEST_KEYS:
            return False
        if manifest.get("id") != artifact_id:
            return False
        if not ARTIFACT_ID_RE.match(artifact_id):
            return False
        kind = manifest.get("kind")
        if kind != artifact_id.split("_", 1)[0] or kind not in ("cond", "latent"):
            return False
        filename = manifest.get("filename")
        if not isinstance(filename, str):
            return False
        try:
            validate_stage_name(filename)
        except ArtifactError:
            return False
        size = manifest.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            return False
        sha = manifest.get("sha256")
        if not isinstance(sha, str) or not _SHA256_RE.match(sha):
            return False
        created = manifest.get("created_at")
        if isinstance(created, bool) or not isinstance(created, (int, float)):
            return False
        if not isinstance(manifest.get("metadata") or {}, dict):
            return False
        if "format" in manifest and manifest["format"] is not None \
                and not isinstance(manifest["format"], str):
            return False
        # 数据文件必须存在、不是 reparse、大小与 manifest 一致
        dpath = os.path.join(self.stage_dir, filename)
        if _is_reparse_or_link(dpath) or not os.path.isfile(dpath):
            return False
        try:
            if os.path.getsize(dpath) != size:
                return False
        except OSError:
            return False
        return True

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

    # -------------------------------------------------- 配额预留

    def _pending_bytes(self):
        return sum(self._pending.values())

    def reserve(self, expected_bytes, *, token=None):
        """在锁内预留配额，避免并发提交各自通过预检后合计超额。"""
        expected = max(0, int(expected_bytes))
        token = token or uuid.uuid4().hex
        with self._lock:
            info = self.budget(expected + self._pending_bytes())
            if not info["fits"]:
                raise ArtifactError("; ".join(info["reasons"]),
                                    code="RESOURCE_EXHAUSTED", http_status=503,
                                    details=info)
            self._pending[token] = expected
        return token

    def release_reservation(self, token):
        with self._lock:
            self._pending.pop(token, None)

    def _check_actual_locked(self, actual_bytes, expected):
        """实际字节复核：临时文件已计入 usage，因此可用量要扣掉预留本身。

        ``_pending`` 里包含本次预留，临时文件也已被 ``stage_usage_bytes`` 统计，
        两者会双算；这里按实际值重新算，且把本次预留从 pending 中剔除后再算，
        避免同一份数据被计两次。
        """
        actual = int(actual_bytes)
        free = self.free_bytes()
        usage = self.stage_usage_bytes()
        quota_left = self.quota_bytes - usage
        usable = free - self.disk_reserve_bytes
        # pending 中除本次以外的预留仍需占位
        others = self._pending_bytes() - int(expected or 0)
        ok = actual <= quota_left - others and actual <= usable
        if ok:
            return
        raise ArtifactError(
            "实际写入 %d 字节超出预算（stage 剩余 %d，磁盘可用 %d，"
            "其它预留 %d）" % (actual, quota_left, usable, others),
            code="RESOURCE_EXHAUSTED", http_status=503)

    # -------------------------------------------------- 生命周期

    def acquire(self, artifact_id):
        """consumer 取引用；get 与加引用在同一把锁内完成。"""
        with self._lock:
            if artifact_id not in self._index:
                raise ArtifactError("artifact 不存在: %s" % artifact_id,
                                    code="ARTIFACT_NOT_FOUND", http_status=404)
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
        """删除 artifact；引用检查与 unlink 在同一把锁内完成。"""
        with self._lock:
            manifest = self._index.get(artifact_id)
            if manifest is None:
                raise ArtifactError("artifact 不存在: %s" % artifact_id,
                                    code="ARTIFACT_NOT_FOUND", http_status=404)
            if not force and self._refs.get(artifact_id):
                raise ArtifactError("artifact 正被引用，暂不能删除: %s"
                                    % artifact_id,
                                    code="ACTIVE_BINDING", http_status=409)
            filename = manifest["filename"]
            path = self.path_for(filename)
            for p in (path, self._manifest_path(artifact_id)):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
            self._index.pop(artifact_id, None)
            self._refs.pop(artifact_id, None)
        return True

    def _delete_expired_locked(self, now, limit, removed):
        for manifest in sorted(list(self._index.values()),
                               key=lambda m: m.get("created_at") or 0):
            if limit is not None and len(removed) >= int(limit):
                break
            aid = manifest["id"]
            created = float(manifest.get("created_at") or 0)
            if now - created < self.ttl_s:
                continue
            if self._refs.get(aid):
                continue
            filename = manifest["filename"]
            for p in (self.path_for(filename), self._manifest_path(aid)):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass
                except OSError:
                    continue
            self._index.pop(aid, None)
            removed.append(aid)

    def cleanup(self, *, now=None, limit=None):
        """清理过期且未被引用的 artifact（引用检查与删除在同一把锁内）。"""
        now = time.time() if now is None else float(now)
        removed = []
        with self._lock:
            self._delete_expired_locked(now, limit, removed)
        return removed

    # -------------------------------------------------- 落盘入口

    def _validate_metadata(self, metadata):
        """写入前把 metadata 规范成 JSON 字符串字典（safetensors 只收 str）。

        非 dict / 键含控制字符 / 值不是 str|int|float|bool|None / 不可 JSON
        序列化 → 400；不引入额外依赖，也不把错误留到 manifest 落盘后才暴露。
        """
        if metadata is None:
            return {}
        if not isinstance(metadata, dict):
            raise ArtifactError("metadata 必须是 dict", code="INVALID_REQUEST",
                                http_status=400)
        out = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or not key:
                raise ArtifactError("metadata 键必须是非空字符串",
                                    code="INVALID_REQUEST", http_status=400)
            if any(ord(c) < 0x20 for c in key):
                raise ArtifactError("metadata 键不能包含控制字符: %r" % key,
                                    code="INVALID_REQUEST", http_status=400)
            if value is None:
                continue
            if isinstance(value, bool):
                out[key] = "true" if value else "false"
            elif isinstance(value, str):
                out[key] = value
            elif isinstance(value, int):
                out[key] = str(value)
            elif isinstance(value, float):
                if value != value or value in (float("inf"), float("-inf")):
                    raise ArtifactError("metadata.%s 不是有限数字" % key,
                                        code="INVALID_REQUEST", http_status=400)
                out[key] = repr(value)
            else:
                try:
                    out[key] = json.dumps(value, ensure_ascii=False,
                                          allow_nan=False)
                except (TypeError, ValueError):
                    raise ArtifactError(
                        "metadata.%s 不是 JSON 可序列化值: %r"
                        % (key, type(value).__name__),
                        code="INVALID_REQUEST", http_status=400)
        return out

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
        流程：校验 → 配额预留 → 写临时文件 → **实际字节复核** → 确认目标名未被
        占用 → 原子改名 → 写 manifest。manifest 失败只回滚本次数据文件。
        """
        validate_stage_name(filename)
        if kind not in ("cond", "latent"):
            raise ArtifactError("未知 artifact kind: %r" % (kind,),
                                code="INVALID_REQUEST", http_status=400)
        meta = self._validate_metadata(metadata)
        expected = max(0, int(expected_bytes or 0))
        token = self.reserve(expected)
        artifact_id = self.new_id(kind)
        path = self.path_for(filename)
        data_written = False
        try:
            if os.path.lexists(path):
                raise ArtifactError("文件名已被占用: %s" % filename,
                                    code="INVALID_REQUEST", http_status=409)
            directory = os.path.dirname(path) or "."
            fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".part",
                                       dir=directory)
            os.close(fd)
            try:
                writer(tmp)
                with open(tmp, "r+b") as f:
                    os.fsync(f.fileno())
                size = os.path.getsize(tmp)
                with self._lock:
                    self._check_actual_locked(size, expected)
                # 目标名必须仍然空着（防并发抢占）
                if os.path.lexists(path):
                    raise ArtifactError("文件名已被占用: %s" % filename,
                                        code="INVALID_REQUEST", http_status=409)
                os.replace(tmp, path)
                data_written = True
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
                "metadata": meta,
            }
            try:
                self._write_manifest(manifest)
            except BaseException:
                # manifest 未落地 → 数据文件成为无索引孤儿，只回滚本次写入
                try:
                    os.unlink(path)
                except OSError:
                    pass
                raise
            return self.public(artifact_id)
        finally:
            self.release_reservation(token)


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

# 单文件字节上限（读取前按 header 预算，不先读大 tensor）
MAX_AVLATENT_BYTES = 8 * 1024 ** 3
MAX_COND_FILE_BYTES = 96 * 1024 * 1024

# tags 的合法形状族（真实 ClipProj 返回 [seq]，也可能带 batch 维）
# 见 clipproj_nodes.tags_from_embeds_info: torch.ones(seq_len, dtype=long)


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
    """把 noise_mask 容器拆成 ``(video, audio, order)``。

    ``split_container`` 若提供就真正被使用（不再被忽略）；返回值必须按
    video,audio 排列，随后再用 mask 维度复核，不匹配即 422。
    """
    split = split_container or _structural_mask_split
    return _split_pair(noise_mask, split, _classify_mask, label="noise_mask")


def _structural_mask_split(container):
    """mask 的默认拆分：按 5D/4D 定位 video/audio，不套 latent 通道规则。"""
    first, second, _ = _structural_split(container, label="noise_mask")
    if _classify_mask(first) == "video":
        return first, second, "video,audio"
    return second, first, "audio,video"


def _structural_split(container, *, label):
    """按结构（.tensors / {video,audio} / 二元序列）取两个元素，不看语义。"""
    if hasattr(container, "tensors") and isinstance(
            getattr(container, "tensors"), (list, tuple)):
        items = list(container.tensors)
    elif isinstance(container, dict):
        if set(container) != {"video", "audio"}:
            raise ArtifactError(
                "%s 映射必须且只能包含 video/audio，实际: %s"
                % (label, ", ".join(sorted(map(str, container)))),
                code="INVALID_ARTIFACT", http_status=422)
        return container["video"], container["audio"], "video,audio"
    elif isinstance(container, (list, tuple)):
        items = list(container)
    else:
        raise ArtifactError("%s 未知容器类型: %s（需要 .tensors / "
                            "{video,audio} / 二元序列）" % (label, type(container)),
                            code="INVALID_ARTIFACT", http_status=422)
    if len(items) != 2:
        raise ArtifactError("%s 容器必须恰好 2 个元素，实际 %d 个"
                            % (label, len(items)),
                            code="INVALID_ARTIFACT", http_status=422)
    return items[0], items[1], "video,audio"


def _split_pair(container, split, classify, *, label):
    """用 ``split`` 拆容器，再用 ``classify`` 复核 video/audio 位置。

    ``split`` 统一返回 ``(video, audio, order)``；order 只用于记录原始顺序，
    语义校验以 ``classify`` 为准，因此顺序写错会变成明确的 422 而不是
    后续 KeyError/静默错位。
    """
    video, audio, order = split(container)
    if order not in ("video,audio", "audio,video"):
        raise ArtifactError("%s 拆分返回的 order 非法: %r" % (label, order),
                            code="INVALID_ARTIFACT", http_status=422)
    kv = classify(video)
    ka = classify(audio)
    if kv != "video" or ka != "audio":
        raise ArtifactError(
            "%s 拆分的两个元素与 video/audio 不符（得到 %s/%s）"
            % (label, kv, ka), code="INVALID_ARTIFACT", http_status=422,
            details={"label": label, "kinds": [kv, ka]})
    return video, audio, order


def default_split_container(container):
    """把 latent 容器拆成 ``(video, audio, order)``。

    支持三种真实可能形状，且都只按“结构”识别，不猜：
    1. Comfy NestedTensor：``.tensors`` 为长度 2 的 list；
    2. 映射：``{"video": T, "audio": T}``；
    3. 二元 tuple / list。

    返回的 order 记录容器内原始顺序（``video,audio`` / ``audio,video``），
    重建时按原序放回，避免隐式交换。
    """
    first, second, _ = _structural_split(container, label="latent")
    kinds = [_classify_av(first), _classify_av(second)]
    if kinds[0] == "video":
        return first, second, "video,audio"
    return second, first, "audio,video"


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
    """按记录的容器顺序重建（与原容器同类则可直接复用其类）。

    校验用 ``is``（身份比较），不用 ``==``，避免 tensor 逐元素布尔歧义。
    """
    pair = {"video": video, "audio": audio}
    tensors = [pair[name] for name in order.split(",")]
    for cls in _container_classes():
        try:
            obj = cls(list(tensors))
        except Exception:            # noqa: BLE001 — 只是候选
            continue
        got = getattr(obj, "tensors", None)
        if isinstance(got, (list, tuple)) and len(got) == len(tensors) \
                and all(a is b for a, b in zip(got, tensors)):
            return obj
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

def _batch_index_meta(batch_index):
    """batch_index 只接受 JSON 整数列表（可无损 roundtrip）；其它形态一律拒绝。"""
    if batch_index is None:
        return None
    if isinstance(batch_index, bool):
        raise ArtifactError("batch_index 不支持 bool", code="INVALID_ARTIFACT",
                            http_status=422)
    if hasattr(batch_index, "tolist"):
        batch_index = batch_index.tolist()
    if isinstance(batch_index, int):
        batch_index = [batch_index]
    if not isinstance(batch_index, (list, tuple)) or not batch_index:
        raise ArtifactError(
            "batch_index 必须是 JSON 整数列表（实际 %s），无法无损保存"
            % type(batch_index).__name__, code="INVALID_ARTIFACT",
            http_status=422)
    out = []
    for i, item in enumerate(batch_index):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ArtifactError("batch_index[%d] 不是整数: %r" % (i, item),
                                code="INVALID_ARTIFACT", http_status=422)
        out.append(int(item))
    return out


def split_av_latent(latent, *, split_container=None):
    """把采样器输出的 latent 拆成 ``({tensor keys}, extra_meta)``。

    ``latent`` 是 Comfy latent 字典（含 ``samples``，可能有 ``noise_mask``）。
    未知键直接报错，不静默丢弃。
    """
    if not isinstance(latent, dict):
        raise ArtifactError("latent 必须是 dict（含 samples）",
                            code="INVALID_ARTIFACT", http_status=422)
    if "samples" not in latent:
        raise ArtifactError("latent 缺少 samples 键",
                            code="INVALID_ARTIFACT", http_status=422)
    unsupported = sorted(
        k for k in latent
        if k != "samples" and k not in (
            "noise_mask", "noise_mask_video", "noise_mask_audio",
            "batch_index"))
    if unsupported:
        raise ArtifactError(
            "latent 含未支持的字段，拒绝保存（避免静默丢数据）: %s"
            % ", ".join(unsupported),
            code="INVALID_ARTIFACT", http_status=422,
            details={"unsupported_fields": unsupported})

    split = split_container or default_split_container
    video, audio, order = _split_pair(latent["samples"], split, _classify_av,
                                      label="latent")
    out = {
        "video": _to_cpu_contiguous(video, label="video"),
        "audio": _to_cpu_contiguous(audio, label="audio"),
    }

    # noise_mask：统一单块与拆块不能同时存在；拆块必须 video/audio 成对
    has_unified = latent.get("noise_mask") is not None
    has_split = any(latent.get(k) is not None
                    for k in ("noise_mask_video", "noise_mask_audio"))
    if has_unified and has_split:
        raise ArtifactError(
            "noise_mask 与 noise_mask_video/noise_mask_audio 不能同时提供",
            code="INVALID_ARTIFACT", http_status=422)
    if has_split:
        vm, am = latent.get("noise_mask_video"), latent.get("noise_mask_audio")
        if vm is None or am is None:
            raise ArtifactError(
                "noise_mask_video / noise_mask_audio 必须成对提供",
                code="INVALID_ARTIFACT", http_status=422)
        out["noise_mask_video"] = _to_cpu_contiguous(vm, label="noise_mask_video")
        out["noise_mask_audio"] = _to_cpu_contiguous(am, label="noise_mask_audio")
    elif has_unified:
        mask = latent["noise_mask"]
        if hasattr(mask, "tensors") or isinstance(mask, (list, tuple)):
            # mask 的结构与 latent 容器无必然关系（可能是 5D/4D 对），因此不使用
            # 调用方给 latent 的 split_container，统一按 mask 自身维度拆分。
            vmask, amask, _ = split_noise_mask(mask)
            out["noise_mask_video"] = _to_cpu_contiguous(vmask, label="noise_mask_video")
            out["noise_mask_audio"] = _to_cpu_contiguous(amask, label="noise_mask_audio")
        else:
            mask_t = _to_cpu_contiguous(mask, label="noise_mask")
            if mask_t.ndim == 5:
                # 单块 5D mask 视为整个 video；audio mask 缺失必须显式声明
                raise ArtifactError(
                    "单个 5D noise_mask 缺少 audio 部分；请分别提供 "
                    "noise_mask_video / noise_mask_audio", code="INVALID_ARTIFACT",
                    http_status=422)
            out["noise_mask"] = mask_t

    extra = {"container_order": order,
             "container_type": type(latent["samples"]).__name__}
    bi = _batch_index_meta(latent.get("batch_index"))
    if bi is not None:
        extra["batch_index"] = bi
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

    **先在 header 层校验**（metadata/keys/shape/dtype/字节上限），通过后才读取
    tensor，避免伪造大文件把显存/内存打满。重建容器时按保存的
    ``container_order`` 放回原序；不自动补未知字段。
    """
    path = store.resolve(name)
    size = os.path.getsize(path)
    if size > MAX_AVLATENT_BYTES:
        raise ArtifactError(
            "latent 文件 %d 字节超过上限 %d" % (size, MAX_AVLATENT_BYTES),
            code="PAYLOAD_TOO_LARGE", http_status=413)
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        md = dict(f.metadata() or {})
        keys = list(f.keys())
        shapes = {k: list(f.get_slice(k).get_shape()) for k in keys}
        dtypes = {k: str(f.get_slice(k).get_dtype()) for k in keys}

        # 1) 元数据必填与取值
        if md.get("format") != FORMAT_AVLATENT:
            raise ArtifactError("不是 %s 文件（format=%r）"
                                % (FORMAT_AVLATENT, md.get("format")),
                                code="INVALID_ARTIFACT", http_status=422)
        for req in AVLATENT_REQUIRED_META:
            if req == "format":
                continue
            if req not in md:
                raise ArtifactError("latent 文件缺少 metadata.%s" % req,
                                    code="INVALID_ARTIFACT", http_status=422)
        try:
            req_len = int(md["requested_length"])
            native_fps = float(md["native_fps"])
        except (TypeError, ValueError):
            raise ArtifactError("requested_length / native_fps 不是数字",
                                code="INVALID_ARTIFACT", http_status=422)
        if req_len < 1:
            raise ArtifactError("requested_length 必须 ≥1",
                                code="INVALID_ARTIFACT", http_status=422)
        if native_fps != NATIVE_FPS:
            raise ArtifactError("native_fps 必须是 24，文件里是 %s" % native_fps,
                                code="INVALID_ARTIFACT", http_status=422)

        # 2) keys 白名单与成对约束
        missing = [k for k in AVLATENT_CORE_KEYS if k not in keys]
        if missing:
            raise ArtifactError("latent 文件缺少 tensor: %s" % ", ".join(missing),
                                code="INVALID_ARTIFACT", http_status=422)
        unknown = sorted(k for k in keys
                         if k not in AVLATENT_CORE_KEYS + AVLATENT_OPTIONAL_KEYS)
        if unknown:
            raise ArtifactError("latent 文件含未支持 tensor 字段: %s"
                                % ", ".join(unknown),
                                code="INVALID_ARTIFACT", http_status=422)
        if ("noise_mask_video" in keys) != ("noise_mask_audio" in keys):
            raise ArtifactError(
                "noise_mask_video / noise_mask_audio 必须成对出现",
                code="INVALID_ARTIFACT", http_status=422)
        if "noise_mask" in keys and "noise_mask_video" in keys:
            raise ArtifactError(
                "noise_mask 与 noise_mask_video/audio 不能同时存在",
                code="INVALID_ARTIFACT", http_status=422)

        # 3) shape/dtype 校验（不读数据）
        _check_av_shapes(shapes, dtypes)
        order = md.get("container_order") or "video,audio"
        if order not in ("video,audio", "audio,video"):
            raise ArtifactError("container_order 非法: %r" % order,
                                code="INVALID_ARTIFACT", http_status=422)
        bi = None
        if md.get("batch_index") is not None:
            try:
                bi = json.loads(md["batch_index"])
            except ValueError:
                raise ArtifactError("metadata.batch_index 不是合法 JSON",
                                    code="INVALID_ARTIFACT", http_status=422)
            bi = _batch_index_meta(bi)

        # 4) 通过校验后才真正读 tensor
        tensors = {k: f.get_tensor(k) for k in keys}

    build = build_container or default_build_container
    latent = {"samples": _build_checked(build, tensors["video"],
                                        tensors["audio"], order,
                                        label="samples")}
    if bi is not None:
        latent["batch_index"] = bi
    if "noise_mask_video" in tensors:
        latent["noise_mask"] = _build_checked(
            build, tensors["noise_mask_video"], tensors["noise_mask_audio"],
            order, label="noise_mask")
    elif "noise_mask" in tensors:
        latent["noise_mask"] = tensors["noise_mask"]
    md["requested_length"] = req_len
    md["native_fps"] = native_fps
    if bi is not None:
        md["batch_index"] = bi
    return latent, md


def _check_av_shapes(shapes, dtypes):
    """video 必须是 5D 且 C=24；audio 必须是 4D 且 C=32/streams=2。"""
    v = shapes.get("video")
    a = shapes.get("audio")
    if not v or len(v) != 5 or v[1] != VIDEO_CHANNELS:
        raise ArtifactError("video shape %s 不是 [B,24,T,H,W]" % (v,),
                            code="INVALID_ARTIFACT", http_status=422)
    if not a or len(a) != 4 or a[1] != AUDIO_CHANNELS or a[2] != AUDIO_STREAMS:
        raise ArtifactError("audio shape %s 不是 [B,32,2,T]" % (a,),
                            code="INVALID_ARTIFACT", http_status=422)
    if v[0] != a[0]:
        raise ArtifactError("video/audio batch 不一致: %s vs %s" % (v[0], a[0]),
                            code="INVALID_ARTIFACT", http_status=422)
    for key, shape in shapes.items():
        if not all(isinstance(d, int) and d >= 0 for d in shape):
            raise ArtifactError("%s shape 含非法维度 %s" % (key, shape),
                                code="INVALID_ARTIFACT", http_status=422)
    for key in ("video", "audio"):
        dt = dtypes.get(key) or ""
        if not any(dt.endswith(s) for s in ("F32", "F16", "BF16", "F64")):
            raise ArtifactError("%s dtype %s 不是浮点" % (key, dt),
                                code="INVALID_ARTIFACT", http_status=422)
    for key in ("noise_mask", "noise_mask_video", "noise_mask_audio"):
        if key in shapes and not shapes[key]:
            raise ArtifactError("%s shape 为空" % key,
                                code="INVALID_ARTIFACT", http_status=422)


def _build_checked(build, video, audio, order, *, label):
    """重建容器后再做一次结构/顺序校验，防止注入的 build 调换顺序。

    同时接受映射（``{video,audio}``）、带 ``.tensors`` 的类与二元序列；
    用 ``is`` 比较身份，不用 ``==``（避免 tensor 逐元素布尔歧义）。
    """
    obj = build(video, audio, order)
    pair = [video, audio] if order == "video,audio" else [audio, video]
    if isinstance(obj, dict):
        got = [obj.get("video"), obj.get("audio")]
        if order == "audio,video":
            got = [obj.get("audio"), obj.get("video")]
    else:
        got = getattr(obj, "tensors", None)
        if got is None and isinstance(obj, (list, tuple)):
            got = list(obj)
    if got is not None:
        if not isinstance(got, (list, tuple)) or len(got) != 2 \
                or not all(a is b for a, b in zip(got, pair)):
            raise ArtifactError("%s 重建后的容器顺序/内容与 source 不符" % label,
                                code="INVALID_ARTIFACT", http_status=422)
    return obj


# ------------------------------------------------------------- cond 拆装

# cond 的 tags 形状族：真实 ClipProj `tags_from_embeds_info` 返回 [seq]，
# 某些适配器会带 batch 维 → 接受 [T] 或 [B,T]，但必须与 cond 的 token 维对得上。


def _tag_shape_ok(tag_shape, cond_shape):
    """返回 (ok, reason)。"""
    if len(cond_shape) < 2:
        return False, "cond shape %s 至少要有 [T,C] 两维" % (tuple(cond_shape),)
    if len(tag_shape) == 1:
        ok = tag_shape[0] == cond_shape[-2]
        return ok, "1D tags 长度应等于 cond 的 token 维 %d" % cond_shape[-2]
    if len(tag_shape) == 2:
        ok = (tag_shape[0] == cond_shape[0] and tag_shape[1] == cond_shape[-2])
        return ok, "2D tags 应为 [B=%d, T=%d]" % (cond_shape[0], cond_shape[-2])
    return False, "tags 维度只支持 1D/2D，实际 %dD" % len(tag_shape)


def _check_tags_dtype(tag_tensor, *, label):
    if str(tag_tensor.dtype) not in ("torch.int64", "torch.int32",
                                     "torch.int16", "torch.uint8"):
        raise ArtifactError("%s dtype %s 不是整型 tag" % (label, tag_tensor.dtype),
                            code="INVALID_ARTIFACT", http_status=422)


def split_cond(cond_data, *, extensions=()):
    """整理 cond 负载；``minimax_token_tags=None`` 必须显式决策，不静默丢弃。

    ``extensions`` 声明的字段会**被真正保存**（不做“声明了却丢弃”），
    调用方 read 时也必须传入同样的 extensions 才能读回。
    """
    if not isinstance(cond_data, dict):
        raise ArtifactError("cond 数据必须是 dict", code="INVALID_ARTIFACT",
                            http_status=422)
    allowed = set(COND_REQUIRED_KEYS) | set(COND_OPTIONAL_KEYS) | set(extensions)
    unknown = sorted(k for k in cond_data if k not in allowed)
    if unknown:
        raise ArtifactError("cond 含未支持字段，拒绝保存: %s" % ", ".join(unknown),
                            code="INVALID_ARTIFACT", http_status=422,
                            details={"unsupported_fields": unknown})
    for key in COND_REQUIRED_KEYS:
        if key not in cond_data:
            raise ArtifactError("cond 缺少必需字段: %s" % key,
                                code="INVALID_ARTIFACT", http_status=422)
    out = {}
    out["cond"] = _to_cpu_contiguous(cond_data["cond"], label="cond")
    tags = cond_data["minimax_token_tags"]
    if tags is None:
        raise ArtifactError(
            "minimax_token_tags 为 None：不能写出缺少必需 key 的“成功”cond 文件",
            code="INVALID_ARTIFACT", http_status=422,
            details={"field": "minimax_token_tags", "reason": "none"})
    out["minimax_token_tags"] = _to_cpu_contiguous(tags, label="minimax_token_tags")
    _check_tags_dtype(out["minimax_token_tags"], label="minimax_token_tags")
    ok, reason = _tag_shape_ok(tuple(out["minimax_token_tags"].shape),
                               tuple(out["cond"].shape))
    if not ok:
        raise ArtifactError(
            "minimax_token_tags shape %s 与 cond shape %s 不匹配：%s"
            % (tuple(out["minimax_token_tags"].shape), tuple(out["cond"].shape),
               reason), code="INVALID_ARTIFACT", http_status=422)
    for key in list(COND_OPTIONAL_KEYS) + list(extensions):
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
    if extensions:
        md["extensions"] = sorted(str(e) for e in extensions)

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

    先在 header 层（metadata/keys/shape/dtype/字节上限）校验，通过后才读 tensor。
    两个必需 key 缺失即报错；legacy 文件（无 h3.cond.v1 metadata）仍然接受，
    但明确写着别的 format 时拒绝。
    """
    path = store.resolve(name)
    size = os.path.getsize(path)
    if size > MAX_COND_FILE_BYTES:
        raise ArtifactError(
            "cond 文件 %d 字节超过上限 %d" % (size, MAX_COND_FILE_BYTES),
            code="PAYLOAD_TOO_LARGE", http_status=413)
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        md = dict(f.metadata() or {})
        keys = list(f.keys())
        shapes = {k: list(f.get_slice(k).get_shape()) for k in keys}
        dtypes = {k: str(f.get_slice(k).get_dtype()) for k in keys}

        declared = md.get("format")
        if declared is not None and declared != FORMAT_COND:
            raise ArtifactError(
                "cond 文件 format=%r，既不是 %s 也不是 legacy（无 metadata）"
                % (declared, FORMAT_COND),
                code="INVALID_ARTIFACT", http_status=422)
        allowed = set(COND_REQUIRED_KEYS) | set(COND_OPTIONAL_KEYS) | set(extensions)
        unknown = sorted(k for k in keys if k not in allowed)
        if unknown:
            raise ArtifactError("cond 文件含未支持 tensor: %s" % ", ".join(unknown),
                                code="INVALID_ARTIFACT", http_status=422)
        missing = [k for k in COND_REQUIRED_KEYS if k not in keys]
        if missing:
            raise ArtifactError("cond 文件缺少必需 tensor: %s" % ", ".join(missing),
                                code="INVALID_ARTIFACT", http_status=422)
        _check_cond_shapes(shapes, dtypes)

        tensors = {k: f.get_tensor(k) for k in keys}
    return tensors, md


def _check_cond_shapes(shapes, dtypes):
    """在 header 层校验 cond/tags 形状与 dtype（不读数据）。"""
    cond = shapes.get("cond")
    tags = shapes.get("minimax_token_tags")
    if not cond or len(cond) < 2:
        raise ArtifactError("cond shape %s 至少要有 [T,C] 两维" % (cond,),
                            code="INVALID_ARTIFACT", http_status=422)
    dt = (dtypes.get("cond") or "")
    if not any(dt.endswith(s) for s in ("F32", "F16", "BF16", "F64")):
        raise ArtifactError("cond dtype %s 不是浮点" % dt,
                            code="INVALID_ARTIFACT", http_status=422)
    if not tags or len(tags) not in (1, 2):
        raise ArtifactError("minimax_token_tags shape %s 不是 1D/2D" % (tags,),
                            code="INVALID_ARTIFACT", http_status=422)
    ok, reason = _tag_shape_ok(tuple(tags), tuple(cond))
    if not ok:
        raise ArtifactError("minimax_token_tags %s 与 cond %s 不匹配：%s"
                            % (tuple(tags), tuple(cond), reason),
                            code="INVALID_ARTIFACT", http_status=422)
    tdt = dtypes.get("minimax_token_tags") or ""
    if not any(tdt.endswith(s) for s in ("I64", "I32", "I16", "U8")):
        raise ArtifactError("minimax_token_tags dtype %s 不是整型" % tdt,
                            code="INVALID_ARTIFACT", http_status=422)


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

