"""远端权重 API 冒烟：DiT block 级、VAE 整组、VEDA bundle。

与 te_remote_smoke.py 共用一个自启服务；只操作自己创建的进程。
默认跑：
  dit  dry_run(all) -> disk2ram(dit.blocks.0) -> ram2vram -> vram2ram -> purge
  vae.video  disk2ram(all) -> purge
  vae.audio  disk2ram(all) -> purge
  veda.bundle disk2ram(all) -> purge
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from te_remote_smoke import Client, free_port, wait_server  # noqa: E402


def action(client, model, name, body, timeout, report):
    started = time.perf_counter()
    status, receipt = client.request(
        "POST", "/api/weights/%s/%s" % (model, name), body)
    if status != 202:
        raise RuntimeError("%s %s -> %s: %s" % (model, name, status, receipt))
    operation = client.wait_operation(receipt["operation_id"], timeout)
    entry = {"model": model, "action": name, "status": operation["status"],
             "elapsed_s": round(time.perf_counter() - started, 2),
             "result": operation["result"], "error": operation["error"]}
    report.setdefault("actions", []).append(entry)
    print("[smoke] %s %s -> %s (%.1fs)"
          % (model, name, operation["status"], entry["elapsed_s"]), flush=True)
    if operation["status"] != "done":
        raise RuntimeError("%s %s failed: %s" % (model, name, operation["error"]))
    return operation


def weights_of(client, model):
    status, body = client.request("GET", "/api/weights/" + model)
    if status != 200:
        raise RuntimeError("weights %s -> %s: %s" % (model, status, body))
    return body["model"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--defaults-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default="dit,vae.video,vae.audio,veda.bundle")
    args = parser.parse_args()
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]

    root = Path(args.server_root).resolve()
    report_path = Path(args.out).resolve()
    log_path = report_path.with_suffix(".log")
    port = free_port()
    report = {"probe": "weights_remote_smoke", "port": port,
              "models": wanted, "verdict": "failed"}
    client = Client(port)
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
            report["boot_weights"] = boot["weights"]

            if "dit" in wanted:
                operation = action(client, "dit", "disk2ram",
                                   {"dry_run": True}, 120, report)
                estimate = operation["result"]["estimate"]
                assert len(estimate["would_change"]) >= 50, estimate
                report["dit_units"] = len(estimate["would_change"])
                action(client, "dit", "disk2ram",
                       {"mode": "units", "units": ["dit.blocks.0"]}, 600,
                       report)
                model = weights_of(client, "dit")
                units = {u["id"]: u for u in model["units"]}
                assert units["dit.blocks.0"]["state"] == "ram", units["dit.blocks.0"]
                assert model["counts"]["ram"] == 1, model["counts"]
                assert model["state"] == "partial", model
                assert model["bytes"]["resident_cpu"] >= 380 * 1024 ** 2, \
                    model["bytes"]
                action(client, "dit", "ram2vram",
                       {"mode": "units", "units": ["dit.blocks.0"]}, 240,
                       report)
                model = weights_of(client, "dit")
                units = {u["id"]: u for u in model["units"]}
                assert units["dit.blocks.0"]["state"] == "vram", units["dit.blocks.0"]
                action(client, "dit", "vram2ram",
                       {"mode": "units", "units": ["dit.blocks.0"]}, 240,
                       report)
                action(client, "dit", "purgefromram",
                       {"mode": "units", "units": ["dit.blocks.0"]}, 240,
                       report)
                model = weights_of(client, "dit")
                units = {u["id"]: u for u in model["units"]}
                assert units["dit.blocks.0"]["state"] == "absent", units["dit.blocks.0"]
                assert model["bytes"]["resident_cpu"] == 0, model["bytes"]
                report["dit_after_purge"] = model["state"]

            for model_id in ("vae.video", "vae.audio", "veda.bundle"):
                if model_id not in wanted:
                    continue
                action(client, model_id, "disk2ram", {"mode": "all"}, 600,
                       report)
                model = weights_of(client, model_id)
                assert model["bytes"]["resident_cpu"] > 0, model
                report[model_id + "_ram"] = model["bytes"]
                action(client, model_id, "ram2vram", {"mode": "all"}, 300,
                       report)
                model = weights_of(client, model_id)
                assert model["state"] == "vram", model
                action(client, model_id, "vram2ram", {"mode": "all"}, 300,
                       report)
                action(client, model_id, "purgefromram", {"mode": "all"}, 300,
                       report)
                model = weights_of(client, model_id)
                assert model["state"] == "absent", model
            report["verdict"] = "ok"
    except Exception as error:
        import traceback
        traceback.print_exc()
        report["error"] = "%s: %s" % (type(error).__name__, error)
        try:
            report["actual"] = {name: weights_of(client, name)
                                for name in wanted}
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
