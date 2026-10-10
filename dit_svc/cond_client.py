"""经 ssh -R 隧道回调本机 cond server（方案 A）。"""
import json
import urllib.error
import urllib.request


def health(cond_url, timeout=3):
    """返回 cond server /health 的 JSON；不可达返回 None。"""
    try:
        with urllib.request.urlopen(cond_url.rstrip("/") + "/health", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def fetch(cond_url, payload, timeout=1800):
    """POST /encode（同步编码）+ 下载 cond safetensors。返回 (bytes, stats dict)。

    隧道未建立时会抛 URLError —— 调用方给出明确提示（先开 -R 隧道）。
    """
    base = cond_url.rstrip("/")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base + "/encode", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            "连不上本机 cond server（%s）——远程回调依赖 ssh -R 隧道（见 tunnel.ps1），"
            "请开隧道后再提交任务（原始错误: %s）" % (base, e)
        ) from e
    if not resp.get("ok"):
        raise RuntimeError("cond server 编码失败: %s" % resp.get("error", "未知错误"))

    file_url = resp.get("file", "")
    if file_url.startswith("/"):
        file_url = base + file_url
    try:
        with urllib.request.urlopen(file_url, timeout=300) as r:
            data = r.read()
    except urllib.error.URLError as e:
        raise RuntimeError("下载 cond 文件失败（%s）: %s" % (file_url, e)) from e
    return data, resp.get("stats", {})
