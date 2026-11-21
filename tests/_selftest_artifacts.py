"""自检：dit_svc/artifacts.py —— 路径/预算/manifest 层纯标准库可跑。

序列化层（torch/safetensors）另见 tests/_selftest_artifacts_serial.py，
需要 torch+safetensors 的隔离环境。
"""
import json
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dit_svc.artifacts import (  # noqa: E402
    ArtifactError, ArtifactStore, atomic_write_bytes, safe_join,
    validate_stage_name,
)

fails = []
TMP = tempfile.mkdtemp(prefix="h3art_")


def check(name, fn):
    try:
        value = fn()
        if value is False:
            raise AssertionError("返回 False")
        print("  OK   %-50s %s" % (name, value))
    except Exception as e:  # noqa: BLE001
        print("  FAIL %-50s %s: %s" % (name, type(e).__name__, e))
        fails.append(name)


def expect(name, fn, *, code=None, contains=None):
    def run():
        try:
            fn()
        except ArtifactError as e:
            if code is not None and e.code != code:
                raise AssertionError("code=%s 期望 %s" % (e.code, code))
            if contains is not None and contains not in str(e):
                raise AssertionError("message %r 不含 %r" % (str(e), contains))
            assert e.to_error()["ok"] is False
            return "%s: %s" % (e.code, str(e)[:48])
        raise AssertionError("没有抛出 ArtifactError")
    check(name, run)


print("== 文件名校验 ==")
for ok_name in ("abc.safetensors", "cond_%s.safetensors" % ("a" * 8), "x.mp4"):
    check("接受 %s" % ok_name, lambda n=ok_name: validate_stage_name(n) == n)
for bad, why in (
        ("../x", ".."), ("a/b", "目录分隔"), ("a\\b", "反斜杠"),
        ("C:\\x", "冒号"), ("\\\\host\\share", "反斜杠"),
        ("%2e%2e%2fx", "百分号编码"), ("a%00b", "百分号编码"),
        ("x.", "以点"), ("x ", "以空格"), ("con", "保留设备名"),
        ("", "非空"), (".", ".."), ("~x", "~")):
    expect("拒绝 %r（%s）" % (bad, why), lambda b=bad: validate_stage_name(b),
           code="INVALID_PATH")
expect("拒绝超长名字", lambda: validate_stage_name("a" * 300), code="INVALID_PATH")
expect("拒绝非字符串", lambda: validate_stage_name(123), code="INVALID_PATH")

print("== safe_join ==")
root = os.path.join(TMP, "stage")
os.makedirs(root, exist_ok=True)
check("正常 join 在根内",
      lambda: safe_join(root, "a.bin").startswith(os.path.realpath(root)))
expect("join 穿越拒绝", lambda: safe_join(root, "..\\a"), code="INVALID_PATH")


def _make_link():
    """尝试创建 symlink（Windows 需开发者模式/管理员）；不支持则跳过。"""
    target = os.path.join(root, "real.bin")
    with open(target, "wb") as f:
        f.write(b"x")
    link = os.path.join(root, "link.bin")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        return None
    return link


link = _make_link()
if link is None:
    print("  SKIP symlink 逃逸（本机无创建权限）")
else:
    expect("symlink 拒绝", lambda: safe_join(root, "link.bin"),
           code="INVALID_PATH", contains="符号链接")

print("== 原子写 ==")


def _atomic():
    p = os.path.join(root, "atomic.bin")
    atomic_write_bytes(p, b"hello")
    leftovers = [n for n in os.listdir(root) if n.startswith(".tmp_")]
    with open(p, "rb") as f:
        data = f.read()
    assert data == b"hello", data
    assert not leftovers, leftovers
    return "ok"


check("原子写无临时残留", _atomic)


def _atomic_cleanup():
    p = os.path.join(root, "boom.bin")

    class Boom(BaseException):
        pass

    try:
        atomic_write_bytes(p, object())
    except BaseException:
        pass
    leftovers = [n for n in os.listdir(root) if n.startswith(".tmp_")]
    assert not os.path.exists(p)
    assert not leftovers, leftovers
    return "ok"


check("写入异常不留半写文件/临时文件", _atomic_cleanup)


def _unique_tmp():
    seen = set()
    for _ in range(20):
        p = os.path.join(root, "u%d.bin" % len(seen))
        before = set(os.listdir(root))
        atomic_write_bytes(p, b"x")
    return "ok"


check("批量原子写", _unique_tmp)

print("== ArtifactStore ==")
out_dir = os.path.join(TMP, "out")
store = ArtifactStore(out_dir, quota_bytes=1 << 20, disk_reserve_bytes=0)
check("stage 目录创建", lambda: os.path.isdir(store.stage_dir))

a = store.commit("cond", "c1.safetensors", b"payload-1", fmt="h3.cond.v1",
                 metadata={"format": "h3.cond.v1"})
check("commit 返回 id/url/bytes",
      lambda: a["id"].startswith("cond_") and a["url"] == "/outputs/stage/c1.safetensors"
      and a["bytes"] == 9)
check("get 摘要一致", lambda: store.get(a["id"])["sha256"] == a["sha256"])
check("public 不含原字节",
      lambda: "data" not in store.public(a["id"]))
expect("未知 artifact 404", lambda: store.get("cond_" + "f" * 32),
       code="ARTIFACT_NOT_FOUND")
expect("重复文件名 409",
       lambda: store.commit("cond", "c1.safetensors", b"x"), code="INVALID_REQUEST")
expect("kind 非法", lambda: store.commit("nope", "z.bin", b"x"),
       code="INVALID_REQUEST")

print("-- 预算 --")
b = store.budget(1 << 19)
check("预算字段齐全",
      lambda: b["fits"] is True and b["estimate_quality"] == "exact_expected_bytes")
big = store.budget(1 << 21)
check("超配额 fits=False",
      lambda: big["fits"] is False and big["reasons"])
expect("ensure_budget 超限 503",
       lambda: store.ensure_budget(1 << 21), code="RESOURCE_EXHAUSTED")

print("-- 引用与清理 --")
store.acquire(a["id"])
check("引用计数", lambda: store.refcount(a["id"]) == 1)
expect("被引用删除 409", lambda: store.delete(a["id"]), code="ACTIVE_BINDING")
store.release(a["id"])
check("释放后归零", lambda: store.refcount(a["id"]) == 0)
check("未过期不清理", lambda: store.cleanup(now=time.time()) == [])
removed = store.cleanup(now=time.time() + 48 * 3600)
check("过期清理", lambda: a["id"] in removed)
check("清理后 manifest 移除", lambda: store.get(a["id"]) if False else
      (store.list() == []))

print("-- 重启加载 manifest --")
c = store.commit("latent", "l1.safetensors", b"latent-bytes", fmt="h3.avlatent.v1")
store2 = ArtifactStore(out_dir, quota_bytes=1 << 20, disk_reserve_bytes=0)
check("新实例可查旧 manifest", lambda: store2.get(c["id"])["bytes"] == 12)
check("新实例引用计数从 0 开始", lambda: store2.refcount(c["id"]) == 0)

print("-- resolve_any --")
check("stage/ 前缀剥离",
      lambda: store2.resolve_any("stage/l1.safetensors").endswith("l1.safetensors"))
expect("绝对路径拒绝", lambda: store2.resolve_any("C:\\out\\stage\\l1.safetensors"),
       code="INVALID_PATH", contains="绝对路径")
expect("UNC 拒绝", lambda: store2.resolve_any("\\\\host\\share\\x"), code="INVALID_PATH")
expect("反斜杠拒绝", lambda: store2.resolve_any("stage\\x"), code="INVALID_PATH")
expect("子目录拒绝", lambda: store2.resolve_any("sub/x"), code="INVALID_PATH")
expect("不存在 404", lambda: store2.resolve_any("nope.safetensors"),
       code="ARTIFACT_NOT_FOUND")

print("-- commit_stream --")


def _stream():
    def writer(tmp):
        with open(tmp, "wb") as f:
            f.write(b"streamed!")
    info = store2.commit_stream("latent", "s1.safetensors", writer,
                                fmt="h3.avlatent.v1", expected_bytes=9)
    assert info["bytes"] == 9
    leftovers = [n for n in os.listdir(store2.stage_dir) if n.startswith(".tmp_")]
    assert not leftovers, leftovers
    return "ok"


check("commit_stream 原子落盘", _stream)


def _stream_fail():
    def writer(tmp):
        with open(tmp, "wb") as f:
            f.write(b"partial")
        raise RuntimeError("boom")
    try:
        store2.commit_stream("latent", "s2.safetensors", writer, expected_bytes=7)
    except RuntimeError:
        pass
    leftovers = [n for n in os.listdir(store2.stage_dir)
                 if n.startswith(".tmp_") or n == "s2.safetensors"]
    assert not leftovers, leftovers
    return "ok"


check("commit_stream 失败不留产物", _stream_fail)

print("-- list --")
check("list 按时间倒序且有上限", lambda: len(store2.list(limit=1)) == 1)

print()
shutil.rmtree(TMP, ignore_errors=True)
if fails:
    print("FAILED %d: %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("ALL OK")
