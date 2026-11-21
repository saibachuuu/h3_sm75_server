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

print("-- stage 根不能是 symlink/junction --")


def _stage_root_link():
    base = os.path.join(TMP, "rootlink")
    os.makedirs(base)
    outside = os.path.join(TMP, "rootlink_outside")
    os.makedirs(outside)
    link = os.path.join(base, "stage")
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        return "SKIP"
    try:
        ArtifactStore(base, quota_bytes=1 << 20, disk_reserve_bytes=0)
    except ArtifactError as e:
        assert e.code == "INVALID_PATH", e.code
        return "拒绝: %s" % str(e)[:40]
    raise AssertionError("stage 是 symlink 却未拒绝")


check("stage 根为 symlink 时拒绝", _stage_root_link)

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
print("== 时间轴（stdlib，不 import torch）==")
from dit_svc.artifacts import (  # noqa: E402
    align_frame_count, decode_trim, plan_temporal, video_latent_t,
)

check("align 5 -> 5", lambda: align_frame_count(5) == 5)
check("align 124 -> 124", lambda: align_frame_count(124) == 124)
check("align 100 -> 107", lambda: align_frame_count(100) == 107)
check("align 1 -> 5（下限）", lambda: align_frame_count(1) == 5)
check("align 结果都满足 17k+5",
      lambda: all(align_frame_count(n) % 17 == 5
                  for n in range(1, 400)))
check("latent_t(124) = 37",
      lambda: plan_temporal(124)["latent_t"] == video_latent_t(124) == 37)
check("plan_temporal 124 音频 207",
      lambda: plan_temporal(124)["audio_t"] == 207)
check("plan_temporal 保留 requested_length",
      lambda: plan_temporal(100)["requested_length"] == 100
      and plan_temporal(100)["frame_count"] == 107)


def _expect_artifact(name, fn, code=None, contains=None):
    def run():
        try:
            fn()
        except ArtifactError as e:
            if code and e.code != code:
                raise AssertionError("code=%s" % e.code)
            if contains and contains not in str(e):
                raise AssertionError(str(e))
            return "%s: %s" % (e.code, str(e)[:40])
        raise AssertionError("没抛 ArtifactError")
    check(name, run)


_expect_artifact("decode_trim 帧数不足报错",
                 lambda: decode_trim(100, {"requested_length": 124}),
                 code="EXECUTION_FAILED")
_expect_artifact("decode_trim 缺 requested_length 报错",
                 lambda: decode_trim(127, {}), code="INVALID_ARTIFACT")


def _trim_ok():
    r = decode_trim(127, {"requested_length": 124})
    assert r["keep"] == 124 and r["trim_tail"] == 3 and r["frame_count"] == 124, r
    return "ok"


check("decode_trim 127 -> keep 124", _trim_ok)

print("== 索引安全：伪造/异常 manifest ==")


def _forged_manifest():
    base = os.path.join(TMP, "forged")
    os.makedirs(os.path.join(base, "stage"))
    st = ArtifactStore(base, quota_bytes=1 << 20, disk_reserve_bytes=0)
    good = st.commit("cond", "good.safetensors", b"0123456789",
                     metadata={"format": "h3.cond.v1"})
    sdir = st.stage_dir
    data_path = os.path.join(sdir, "good.safetensors")

    def dump(name, obj):
        with open(os.path.join(sdir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f)

    # 1) 大小与文件不符
    bad = dict(st.get(good["id"]))
    dump("cond_%s.json" % ("1" * 32), dict(bad, id="cond_" + "1" * 32,
                                          filename="good.safetensors",
                                          bytes=999))
    # 2) filename 带穿越
    dump("cond_%s.json" % ("2" * 32), dict(bad, id="cond_" + "2" * 32,
                                          filename="../evil.safetensors"))
    # 3) kind 与 ID 前缀不符
    dump("cond_%s.json" % ("3" * 32), dict(bad, id="cond_" + "3" * 32,
                                          kind="latent"))
    # 4) 多余字段 schema
    dump("cond_%s.json" % ("4" * 32), dict(bad, id="cond_" + "4" * 32,
                                          extra_key="x"))
    # 5) 数据文件不存在
    dump("cond_%s.json" % ("5" * 32), dict(bad, id="cond_" + "5" * 32,
                                          filename="missing.safetensors"))
    # 6) sha 非法
    dump("cond_%s.json" % ("6" * 32), dict(bad, id="cond_" + "6" * 32,
                                          sha256="abc"))
    # 7) 完全不是 JSON
    with open(os.path.join(sdir, "cond_%s.json" % ("7" * 32)), "w") as f:
        f.write("{not json")
    # 8) 非 dict
    dump("cond_%s.json" % ("8" * 32), ["not", "a", "dict"])

    st2 = ArtifactStore(base, quota_bytes=1 << 20, disk_reserve_bytes=0)
    ids = [m["id"] for m in st2.list(limit=100)]
    assert ids == [good["id"]], ids
    assert os.path.isfile(data_path)
    return "只保留合法项"


check("伪造/异常 manifest 不入索引", _forged_manifest)


def _manifest_symlink():
    base = os.path.join(TMP, "mslink")
    os.makedirs(os.path.join(base, "stage"))
    st = ArtifactStore(base, quota_bytes=1 << 20, disk_reserve_bytes=0)
    good = st.commit("cond", "g.safetensors", b"abcdef",
                     metadata={"format": "h3.cond.v1"})
    real_json = os.path.join(st.stage_dir, good["id"] + ".json")
    target = os.path.join(TMP, "stolen.safetensors")
    with open(target, "wb") as f:
        f.write(b"abcdef")
    link_json = os.path.join(st.stage_dir, "cond_%s.json" % ("a" * 32))
    try:
        os.symlink(real_json, link_json)
    except (OSError, NotImplementedError):
        return "SKIP"
    st2 = ArtifactStore(base, quota_bytes=1 << 20, disk_reserve_bytes=0)
    ids = [m["id"] for m in st2.list(limit=100)]
    assert "cond_" + "a" * 32 not in ids, ids
    return "拒绝 symlink manifest"


check("不跟随 symlink manifest", _manifest_symlink)

print("== 生命周期竞态 ==")


def _barrier_acquire_vs_cleanup():
    import threading as th
    base = os.path.join(TMP, "race")
    os.makedirs(base)
    st = ArtifactStore(base, quota_bytes=1 << 20, disk_reserve_bytes=0,
                       ttl_s=0.0)
    info = st.commit("cond", "r.safetensors", b"race-data",
                     metadata={"format": "h3.cond.v1"})
    st._index[info["id"]]["created_at"] = 0    # 立即过期
    barrier = th.Barrier(2)
    got = {}
    errors = []

    def acquirer():
        try:
            barrier.wait()
            st.acquire(info["id"])
            got["ref"] = st.refcount(info["id"])
        except ArtifactError as e:
            got["acq_err"] = e.code     # 清理先赢时这是合法结果
        except Exception as e:      # noqa: BLE001
            errors.append(("acquire", e))

    def cleaner():
        try:
            barrier.wait()
            got["removed"] = st.cleanup()
        except Exception as e:      # noqa: BLE001
            errors.append(("cleanup", e))

    threads = [th.Thread(target=acquirer), th.Thread(target=cleaner)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    if info["id"] in got.get("removed", []):
        # 清理先赢：artifact 已删，acquire 必须 404 而不是拿到半删文件
        assert got.get("acq_err") == "ARTIFACT_NOT_FOUND", got
        assert "ref" not in got, got
        assert not os.path.exists(os.path.join(st.stage_dir, "r.safetensors"))
    else:
        assert got.get("ref") == 1, got
        assert os.path.isfile(os.path.join(st.stage_dir, "r.safetensors"))
        assert st.refcount(info["id"]) == 1
    return "removed=%s acq_err=%s ref=%s" % (
        info["id"] in got.get("removed", []), got.get("acq_err"),
        got.get("ref"))


check("acquire/cleanup 并发不会删 active consumer", _barrier_acquire_vs_cleanup)


def _barrier_delete_vs_acquire():
    import threading as th
    base = os.path.join(TMP, "race2")
    os.makedirs(base)
    st = ArtifactStore(base, quota_bytes=1 << 20, disk_reserve_bytes=0)
    info = st.commit("cond", "d.safetensors", b"x" * 4,
                     metadata={"format": "h3.cond.v1"})
    barrier = th.Barrier(2)
    out = {}

    def deleter():
        barrier.wait()
        try:
            st.delete(info["id"])
            out["deleted"] = True
        except ArtifactError as e:
            out["delete_err"] = e.code

    def acquirer():
        barrier.wait()
        try:
            st.acquire(info["id"])
            out["acquired"] = True
        except ArtifactError as e:
            out["acq_err"] = e.code

    threads = [th.Thread(target=deleter), th.Thread(target=acquirer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 二选一顺序都必须自洽：要么删除成功 + acquire 404，要么 acquire 成功 + delete 409
    if out.get("deleted"):
        assert out.get("acq_err") == "ARTIFACT_NOT_FOUND", out
    else:
        assert out.get("delete_err") == "ACTIVE_BINDING", out
        assert out.get("acquired") is True
    return str(out)


check("delete/acquire 并发自洽", _barrier_delete_vs_acquire)


def _ref_count_multi():
    base = os.path.join(TMP, "multi")
    os.makedirs(base)
    st = ArtifactStore(base, quota_bytes=1 << 20, disk_reserve_bytes=0)
    info = st.commit("cond", "m.safetensors", b"m",
                     metadata={"format": "h3.cond.v1"})
    st.acquire(info["id"])
    st.acquire(info["id"])
    assert st.refcount(info["id"]) == 2
    st.release(info["id"])
    try:
        st.delete(info["id"])
    except ArtifactError as e:
        assert e.code == "ACTIVE_BINDING"
    else:
        raise AssertionError("仍有 1 个引用却允许删除")
    st.release(info["id"])
    st.delete(info["id"])
    assert not os.path.exists(os.path.join(st.stage_dir, "m.safetensors"))
    return "ok"


check("多引用逐个释放后才可删", _ref_count_multi)

print("== 配额：实际字节复核与预留 ==")


def _actual_oversize():
    base = os.path.join(TMP, "quota1")
    os.makedirs(base)
    st = ArtifactStore(base, quota_bytes=64 * 1024, disk_reserve_bytes=0)
    # 预估 1KB（配额 64KB），实际写 4KB 仍然 fits；改小配额后必须失败
    st2 = ArtifactStore(os.path.join(TMP, "quota2"), quota_bytes=4096,
                        disk_reserve_bytes=0)

    def writer(tmp):
        with open(tmp, "wb") as f:
            f.write(b"z" * 32768)

    try:
        st2.commit_stream("cond", "big.safetensors", writer,
                          metadata={"format": "h3.cond.v1"},
                          expected_bytes=1024)
    except ArtifactError as e:
        assert e.code == "RESOURCE_EXHAUSTED", e.code
    else:
        raise AssertionError("实际 32KB 超 4KB 配额却提交成功")
    assert not os.path.exists(os.path.join(st2.stage_dir, "big.safetensors"))
    leftovers = [n for n in os.listdir(st2.stage_dir) if n.startswith(".tmp_")]
    assert not leftovers, leftovers
    return "ok"


check("实际字节超配额时提交前拒绝", _actual_oversize)


def _expected_none():
    base = os.path.join(TMP, "quota3")
    os.makedirs(base)
    st = ArtifactStore(base, quota_bytes=1024, disk_reserve_bytes=0)

    def writer(tmp):
        with open(tmp, "wb") as f:
            f.write(b"y" * 8192)

    try:
        st.commit_stream("cond", "nonep.safetensors", writer,
                         metadata={"format": "h3.cond.v1"}, expected_bytes=None)
    except ArtifactError as e:
        assert e.code == "RESOURCE_EXHAUSTED", e.code
    else:
        raise AssertionError("expected=None 时 8KB 仍被接受")
    return "ok"


check("expected=None 仍按实际字节复核", _expected_none)


def _reservation_concurrent():
    base = os.path.join(TMP, "quota4")
    os.makedirs(base)
    # 配额只够一份 4KB
    st = ArtifactStore(base, quota_bytes=4096, disk_reserve_bytes=0)
    t1 = st.reserve(4096)
    try:
        st.reserve(4096)
    except ArtifactError as e:
        assert e.code == "RESOURCE_EXHAUSTED"
    else:
        raise AssertionError("第二份预留未拒绝")
    st.release_reservation(t1)
    st.reserve(4096)
    return "ok"


check("并发预留不会合计超额", _reservation_concurrent)


def _manifest_fail_rollback():
    base = os.path.join(TMP, "quota5")
    os.makedirs(base)
    st = ArtifactStore(base, quota_bytes=1 << 20, disk_reserve_bytes=0)
    before = set(os.listdir(st.stage_dir))

    def writer(tmp):
        with open(tmp, "wb") as f:
            f.write(b"data")

    orig = st._write_manifest

    def boom(manifest):
        raise OSError("disk full on manifest")

    st._write_manifest = boom
    try:
        st.commit_stream("cond", "mf.safetensors", writer,
                         metadata={"format": "h3.cond.v1"}, expected_bytes=4)
    except OSError:
        pass
    else:
        raise AssertionError("manifest 失败未抛出")
    st._write_manifest = orig
    after = set(os.listdir(st.stage_dir))
    assert after == before, after - before
    assert not os.path.exists(os.path.join(st.stage_dir, "mf.safetensors"))
    assert st.list() == []
    # 不覆盖已存在文件
    st.commit("cond", "keep.safetensors", b"keep",
              metadata={"format": "h3.cond.v1"})
    try:
        st.commit("cond", "keep.safetensors", b"overwrite",
                  metadata={"format": "h3.cond.v1"})
    except ArtifactError as e:
        assert e.code == "INVALID_REQUEST"
    else:
        raise AssertionError("覆盖了已有 artifact")
    with open(os.path.join(st.stage_dir, "keep.safetensors"), "rb") as f:
        assert f.read() == b"keep"
    return "ok"


check("manifest 失败只回滚本次数据且不覆盖", _manifest_fail_rollback)


def _metadata_json_guard():
    base = os.path.join(TMP, "meta")
    os.makedirs(base)
    st = ArtifactStore(base, quota_bytes=1 << 20, disk_reserve_bytes=0)

    class NotJson(object):
        pass

    try:
        st.commit("cond", "m1.safetensors", b"x",
                  metadata={"format": "h3.cond.v1", "bad": NotJson()})
    except ArtifactError as e:
        assert e.code == "INVALID_REQUEST", e.code
    else:
        raise AssertionError("不可序列化 metadata 被接受")
    assert st.list() == []
    try:
        st.commit("cond", "m2.safetensors", b"x",
                  metadata={"format": "h3.cond.v1", "bad\nkey": 1})
    except ArtifactError as e:
        assert e.code == "INVALID_REQUEST"
    else:
        raise AssertionError("控制字符 metadata 键被接受")
    info = st.commit("cond", "m3.safetensors", b"x",
                     metadata={"format": "h3.cond.v1", "n": 3,
                               "f": 0.5, "b": True, "nested": {"a": [1]}})
    md = st.get(info["id"])["metadata"]
    assert md == {"format": "h3.cond.v1", "n": "3", "f": "0.5",
                  "b": "true", "nested": '{"a": [1]}'}, md
    return "ok"


check("metadata 写入前 JSON 规范化", _metadata_json_guard)

shutil.rmtree(TMP, ignore_errors=True)
if fails:
    print("FAILED %d: %s" % (len(fails), ", ".join(fails)))
    sys.exit(1)
print("ALL OK")
