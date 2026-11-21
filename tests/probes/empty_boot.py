"""Bounded real empty-boot HTTP probe; stops only the process it creates.

Run with the actual server interpreter. No model, attention or preload action
is requested. Does not modify installed dependencies or existing services.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-root', required=True)
    parser.add_argument('--runner-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--timeout', type=float, default=45)
    args = parser.parse_args()
    root = Path(args.server_root).resolve()
    report_path = Path(args.out).resolve()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    log_path = report_path.with_suffix('.log')
    process = None
    report = {'probe': 'real_empty_boot', 'port': port,
              'model_actions_requested': [], 'verdict': 'failed'}
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    try:
        with log_path.open('wb') as log:
            process = subprocess.Popen(
                [sys.executable, str(root / 'h3_dit_server.py'),
                 '--host', '127.0.0.1', '--port', str(port),
                 '--runner-dir', str(Path(args.runner_dir).resolve()),
                 '--out-dir', str(root / 'probe_outputs')],
                cwd=str(root), stdout=log, stderr=subprocess.STDOUT, env=env)
            report['created_pid'] = process.pid
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('test server exited %s' % process.returncode)
                try:
                    with urllib.request.urlopen('http://127.0.0.1:%d/api/status' % port,
                                                timeout=3) as response:
                        status = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise TimeoutError('empty HTTP boot exceeded deadline')
            report['status'] = status
            assert status['api_version'] == 2
            assert status['env_ready'] is True, status.get('error')
            assert status['ready'] is False and status['compute_ready'] is False
            assert status['attn'] == 'none'
            assert status['executor']['state'] == 'idle'
            assert len(status['weights']) == 5
            assert all(item['state'] == 'absent' for item in status['weights'].values())
            with urllib.request.urlopen('http://127.0.0.1:%d/api/weights' % port,
                                        timeout=3) as response:
                report['weights'] = json.load(response)
            assert all(not item['catalog_ready'] and not item['units']
                       for item in report['weights']['models'].values())
            report['verdict'] = 'ok'
    except Exception as error:
        report['error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        if process is not None and process.poll() is None:
            # This PID/handle comes exclusively from our Popen above.
            process.terminate()
            process.wait(timeout=10)
            report['created_process_stopped'] = True
        report['log_path'] = str(log_path)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['verdict'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
