"""Self-test for tests/run_selftests.py (stdlib only, no torch)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import run_selftests as runner


def _write_script(path, source):
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


class RunnerTests(unittest.TestCase):
    def test_phase_plan_and_review_glob(self):
        self.assertEqual([s.label for s in runner.P0_SPECS],
                         ["environment_probe", "weights_logic", "storage",
                          "partial", "h3", "real_catalog", "weights_review"])
        self.assertEqual([s.label for s in runner.P1_SPECS],
                         ["empty_startup", "control_http", "operation_admission",
                          "execution_boundary", "check_call_sites"])
        review = runner.P0_SPECS[-1]
        self.assertEqual(review.filename, "_selftest_registry_review.py")
        self.assertIn("_selftest_*weights*review*.py", review.globs)
        self.assertIn("_selftest_*registry*review*.py", review.globs)

        with tempfile.TemporaryDirectory() as directory:
            tests = Path(directory)
            actual = _write_script(
                tests / "_selftest_registry_review.py", "# actual\n")
            path, resolution = runner.resolve_script(review, tests)
            self.assertEqual(path, actual)
            self.assertEqual(resolution, "exact")

        with tempfile.TemporaryDirectory() as directory:
            tests = Path(directory)
            legacy = _write_script(tests / "_weights_review.py", "# legacy\n")
            path, resolution = runner.resolve_script(review, tests)
            self.assertEqual(path, legacy)
            self.assertEqual(resolution, "glob:_weights_review.py")

        # 旧 spec 精确名缺失时，实际 *_review 文件应被 glob 找到。
        legacy_spec = runner.TestSpec(
            "weights_review", "_weights_review.py",
            ("_selftest_*review*.py", "_weights_review.py"))
        with tempfile.TemporaryDirectory() as directory:
            tests = Path(directory)
            actual = _write_script(
                tests / "_selftest_registry_review.py", "# actual\n")
            path, resolution = runner.resolve_script(legacy_spec, tests)
            self.assertEqual(path, actual)
            self.assertEqual(resolution, "glob:_selftest_*review*.py")

    def test_weights_review_does_not_run_unrelated_artifact_review(self):
        review = next(spec for spec in runner.P0_SPECS if spec.label == "weights_review")
        with tempfile.TemporaryDirectory() as directory:
            tests = Path(directory)
            _write_script(tests / "_selftest_artifact_review.py", "# wrong subsystem\n")
            path, _ = runner.resolve_script(review, tests)
            self.assertIsNone(path)

    def test_execute_statuses_utf8_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            tests = Path(directory)
            ok = _write_script(
                tests / "_selftest_ok.py",
                """
                import sys
                print("喵 UTF-8 ok")
                sys.exit(0)
                """)
            fail = _write_script(
                tests / "_selftest_fail.py",
                """
                import sys
                sys.exit(7)
                """)
            slow = _write_script(
                tests / "_selftest_slow.py",
                """
                import time
                time.sleep(5)
                """)
            missing = runner.TestSpec("missing", "_does_not_exist.py")

            rec = runner.execute_spec(
                "p1", runner.TestSpec("ok", ok.name), sys.executable, 5.0,
                tests, tests)
            self.assertEqual(rec["status"], "passed", rec)
            self.assertIn("喵 UTF-8 ok", rec["output_tail"])
            self.assertEqual(rec["returncode"], 0)

            rec = runner.execute_spec(
                "p1", runner.TestSpec("fail", fail.name), sys.executable, 5.0,
                tests, tests)
            self.assertEqual(rec["status"], "failed", rec)
            self.assertEqual(rec["returncode"], 7)

            rec = runner.execute_spec(
                "p1", runner.TestSpec("slow", slow.name), sys.executable, 0.4,
                tests, tests)
            self.assertEqual(rec["status"], "timeout", rec)
            self.assertIn("超过", rec["reason"])

            rec = runner.execute_spec(
                "p1", missing, sys.executable, 5.0, tests, tests)
            self.assertEqual(rec["status"], "not-covered", rec)
            self.assertIn("不计为通过", rec["reason"])

    def test_summary_exit_semantics(self):
        def rec(status):
            return {"status": status}

        passed = runner.summarize([rec("passed"), rec("passed")])
        self.assertEqual(passed["overall"], "passed")
        self.assertEqual(passed["exit_code"], 0)

        partial = runner.summarize([rec("passed"), rec("not-covered")])
        self.assertEqual(partial["overall"], "partial")
        self.assertEqual(partial["exit_code"], 2)

        skipped = runner.summarize([rec("not-covered")])
        self.assertEqual(skipped["overall"], "not-covered")
        self.assertEqual(skipped["exit_code"], 2)

        failed = runner.summarize([rec("passed"), rec("failed"),
                                   rec("not-covered")])
        self.assertEqual(failed["overall"], "failed")
        self.assertEqual(failed["exit_code"], 1)

        timed_out = runner.summarize([rec("timeout")])
        self.assertEqual(timed_out["overall"], "failed")
        self.assertEqual(timed_out["exit_code"], 1)

    def test_write_report_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "report.json"
            written = runner.write_report(str(target), {"ok": True})
            self.assertEqual(written, target.resolve())
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")),
                             {"ok": True})

    def test_actual_worktree_p1_visible_subset_and_report(self):
        """在本 worktree 只跑可见 P1 脚本；缺失脚本必须显式 not-covered。"""
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "p1-report.json"
            command = [
                sys.executable, str(TESTS / "run_selftests.py"),
                "--phase", "p1", "--python", sys.executable,
                "--timeout", "180", "--report", str(report_path),
            ]
            completed = subprocess.run(
                command, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=300, check=False)
            output = completed.stdout.decode("utf-8", "replace")
            self.assertTrue(report_path.is_file(),
                            "missing report; output:\n%s" % output)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["phase"], "p1")
            self.assertEqual(report["exit_code"], completed.returncode)

            records = {item["label"]: item for item in report["tests"]}
            for label in ("operation_admission", "check_call_sites"):
                self.assertIn(label, records)
                self.assertEqual(records[label]["status"], "passed", output)

            missing = [item for item in records.values()
                       if item["status"] == "not-covered"]
            hard = [item for item in records.values()
                    if item["status"] in ("failed", "timeout", "error")]
            expected_exit = 1 if hard else (2 if missing else 0)
            self.assertEqual(completed.returncode, expected_exit, output)
            self.assertEqual(report["summary"]["not-covered"], len(missing))
            if missing:
                # 未覆盖绝不能伪装成 passed；报告和进程码都必须诚实。
                self.assertNotEqual(report["overall"], "passed")
                self.assertEqual(report["overall"],
                                 "failed" if hard else "partial")
            for item in records.values():
                if item["status"] == "not-covered":
                    self.assertIn("不计为通过", item["reason"])
                else:
                    self.assertTrue(item["resolved"], item)


if __name__ == "__main__":
    unittest.main(verbosity=2)
