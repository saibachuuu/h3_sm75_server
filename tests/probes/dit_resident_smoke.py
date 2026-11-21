"""远端 resident DiT 推理冒烟：显式权重上卡后跑一个最小 video job。

流程（全部走 HTTP API，不碰进程内对象）：
  te disk2ram -> ram2vram -> infer.encode -> vram2ram -> purge
  dit disk2ram(all) -> ram2vram(all)
  vae.video / vae.audio disk2ram -> ram2vram
  POST /api/jobs/cond (cond_file = TE artifact, length=22, steps=1)
  轮询到 done，报告 timings / files / latent probe
只停止自己启动的服务进程。

用法：
  python tests/probes/dit_resident_smoke.py --server-root . \
      --defaults-file C:\\Users\\sheepserver\\Desktop\\h3_explicit_defaults.json \
      --out C:\\Users\\sheepserver\\Desktop\\_dit_resident.json
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_remote_smoke import Client, free_port, tiny_image, wait_server  # noqa: E402


def action(client, model, name, body, timeout, report):
    started = time.perf_counter()
    status, receipt = client.request(
        "POST", "/api/weights/%s/%s" % (model, name), body)
    if status != 202:
        raise RuntimeError("%s %s -> %s: %s" % (model, name, status, receipt))
    operation = client.wait_operation(receipt["operation_id"], timeout)
    entry = {"model": model, "action": name, "status": operation["status"],
             "elapsed_s": round(time.perf_counter() - started, 2),
             "error": operation["error"]}
    report.setdefault("actions", []).append(entry)
    print("[smoke] %s %s -> %s (%.1fs)"
          % (model, name, operation["status"], entry["elapsed_s"]), flush=True)
    if operation["status"] != "done":
        raise RuntimeError("%s %s failed: %s"
                           % (model, name, operation["error"]))
    return operation


def weights_of(client, model):
    status, body = client.request("GET", "/api/weights/" + model)
    if status != 200:
        raise RuntimeError("weights %s -> %s: %s" % (model, status, body))
    return body["model"]


def wait_job(client, job_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, body = client.request("GET", "/api/jobs/" + job_id)
        if status != 200:
            raise RuntimeError("job GET %s: %s" % (status, body))
        job = body
        if job.get("status") in ("done", "error", "failed", "cancelled"):
            return job
        time.sleep(1.0)
    raise TimeoutError("job %s exceeded %.0fs" % (job_id, timeout))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--defaults-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--length", type=int, default=22)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.server_root).resolve()
    report_path = Path(args.out).resolve()
    log_path = report_path.with_suffix(".log")
    port = free_port()
    report = {"probe": "dit_resident_smoke", "port": port,
              "length": args.length, "steps": args.steps, "verdict": "failed"}
    client = Client(port, timeout=60)
    process = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [sys.executable, "h3_dit_server.py",
                 "--host", "127.0.0.1", "--port", str(port),
                 "--defaults-file", args.defaults_file,
                 "--out-dir", str(root / "outputs")],
                cwd=str(root), stdout=log, stderr=subprocess.STDOUT)
            report["created_pid"] = process.pid
            boot = wait_server(client, time.monotonic() + 240, process)
            report["boot_weights"] = {k: v["state"] for k, v in boot["weights"].items()}

            # 1) TE 先出一份 cond，然后立刻下卡释放显存
            action(client, "te", "disk2ram", {"mode": "all"}, 900, report)
            action(client, "te", "ram2vram", {"mode": "all"}, 300, report)
            status, receipt = client.request("POST", "/api/infer/encode", {
                "prompt": "a red cube on a wooden table",
                "image_b64s": [tiny_image()],
                "mode": "ref2va", "width": 640, "height": 960,
                "ref_image_size": "match"})
            if status != 202:
                raise RuntimeError("encode -> %s: %s" % (status, receipt))
            operation = client.wait_operation(receipt["operation_id"], 600)
            if operation["status"] != "done":
                raise RuntimeError("encode failed: %s" % (operation["error"],))
            artifact = operation["result"]["artifact"]
            report["cond"] = {"filename": artifact["filename"],
                              "bytes": artifact["bytes"],
                              "stats": operation["result"]["stats"]}
            action(client, "te", "vram2ram", {"mode": "all"}, 300, report)
            action(client, "te", "purgefromram", {"mode": "all"}, 300, report)

            # 2) DiT 全量上卡（14GB）
            action(client, "dit", "disk2ram", {"mode": "all"}, 1800, report)
            model = weights_of(client, "dit")
            report["dit_after_disk2ram"] = model["counts"]
            action(client, "dit", "ram2vram", {"mode": "all"}, 900, report)
            model = weights_of(client, "dit")
            report["dit_after_ram2vram"] = {
                "state": model["state"], "counts": model["counts"],
                "bytes": model["bytes"]}
            assert model["state"] == "vram", model["state"]

            # 3) VAE 只到 RAM：采样不占显存；decode 阶段由 pipeline 显式上卡
            for vae in ("vae.video", "vae.audio"):
                action(client, vae, "disk2ram", {"mode": "all"}, 900, report)
                model = weights_of(client, vae)
                assert model["state"] == "ram", (vae, model["state"])

            # 4) 最小 video job（cond 文件来自 TE artifact）
            payload = {
                "cond_file": "stage/" + artifact["filename"],
                "prompt": "a red cube on a wooden table",
                "mode": "ref2va", "ref_image_size": "match",
                "width": 640, "height": 960,
                "length": args.length, "steps": args.steps,
                "refine_steps": 0, "seed": 7,
                "sampler": "euler", "scheduler": "simple",
            }
            started = time.perf_counter()
            status, receipt = client.request("POST", "/api/jobs/cond", payload)
            if status != 201:
                raise RuntimeError("jobs/cond -> %s: %s" % (status, receipt))
            job = wait_job(client, receipt["job_id"], 1800)
            report["job"] = {
                "id": receipt["job_id"],
                "status": job.get("status"),
                "elapsed_s": round(time.perf_counter() - started, 2),
                "timings": job.get("timings"),
                "files": job.get("files"),
                "latent_probe": job.get("latent_probe"),
                "cond_stats": job.get("cond_stats"),
                "error": job.get("error") or job.get("message"),
                "log_tail": (job.get("log_tail") or [])[-20:],
            }
            print("[smoke] job -> %s (%.1fs)"
                  % (job.get("status"), report["job"]["elapsed_s"]), flush=True)
            if job.get("status") != "done":
                raise RuntimeError("job failed: %s" % (report["job"],))
            report["verdict"] = "ok"
    except Exception as error:
        import traceback
        traceback.print_exc()
        report["error"] = "%s: %s" % (type(error).__name__, error)
        try:
            report["actual_weights"] = {
                name: weights_of(client, name)["state"]
                for name in ("te", "dit", "vae.video", "vae.audio")}
        except Exception:
            pass
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            report["created_process_stopped"] = True
        report["log_path"] = str(log_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(json.dumps({"verdict": report["verdict"],
                          "error": report.get("error"),
                          "report": str(report_path)}, ensure_ascii=False,
                         indent=2), flush=True)
    return 0 if report["verdict"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
