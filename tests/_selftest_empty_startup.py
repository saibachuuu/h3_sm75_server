"""Verify process-only CLI and empty environment initialization with fake dependencies."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h3_dit_server as entry
from dit_svc.runtime import Runtime


def dependencies():
    modules = {}
    for name in ("comfy", "comfy.ldm", "comfy.ldm.modules", "comfy.ldm.minimax",
                 "comfy.options", "comfy.cli_args", "comfy.sd",
                 "comfy.model_management", "comfy.ldm.modules.attention",
                 "comfy.ldm.minimax.model", "comfy_kitchen", "torch"):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    modules["comfy.options"].enable_args_parsing = Mock()
    modules["comfy.cli_args"].args = SimpleNamespace(gpu_only=False)
    modules["torch"].device = lambda value: value
    original = object()
    modules["comfy.ldm.modules.attention"].optimized_attention = original
    modules["comfy.ldm.minimax.model"].optimized_attention = original
    return modules, original


class StartupTests(unittest.TestCase):
    def test_cli_has_no_model_or_attention_action(self):
        args = entry.parse_args([])
        self.assertEqual(args.attn, "none")
        self.assertEqual(args.model_recipes, {})
        self.assertFalse(hasattr(args, "unet"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.parse_args(["--attn", "int8"])

    def test_paths_are_pinned_before_runner_chdir(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                Path("paths.json").write_text(json.dumps({"models": {
                    "dit": {"path": "weight.safetensors"}}}), encoding="utf-8")
                args = entry.parse_args(["--out-dir", "results", "--runner-dir", "runner",
                                         "--defaults-file", "paths.json"])
                os.chdir(original)
                self.assertEqual(args.out_dir, str(Path(directory) / "results"))
                self.assertEqual(args.model_recipes["dit"]["path"],
                                 str(Path(directory) / "weight.safetensors"))
                self.assertEqual(args.runner_dir, str(Path(directory) / "runner"))
            finally:
                os.chdir(original)

    def test_configuration_cannot_request_preload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"models": {}, "preload": "all"}), encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                entry.parse_args(["--defaults-file", str(path)])

    def test_environment_keeps_weights_and_attention_empty(self):
        modules, original = dependencies()
        rt = Mock()
        runtime = Runtime(entry.parse_args([]), rt)
        with patch.dict(sys.modules, modules), patch("dit_svc.runtime.find_ffmpeg", return_value="ffmpeg"), redirect_stdout(io.StringIO()):
            runtime.ensure_env()
            runtime.ensure_env()
        self.assertTrue(runtime.env_ready)
        self.assertFalse(runtime.ready)
        self.assertFalse(runtime.compute_ready)
        for name in ("unet", "model_shifted", "vvae", "avae", "diff_model",
                     "text_encoder", "te_proj", "veda_patch", "attn_override"):
            self.assertIsNone(getattr(runtime, name))
        self.assertIs(modules["comfy.ldm.modules.attention"].optimized_attention, original)
        self.assertIs(modules["comfy.ldm.minimax.model"].optimized_attention, original)
        modules["comfy.options"].enable_args_parsing.assert_called_once_with(True)
        self.assertEqual(rt.mock_calls, [])

    def test_environment_failure_can_be_retried(self):
        modules, _ = dependencies()
        runtime = Runtime(entry.parse_args([]), Mock())
        with patch.dict(sys.modules, modules), patch("dit_svc.runtime.find_ffmpeg", return_value=None), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError):
                runtime.ensure_env()
        self.assertFalse(runtime.env_ready)
        self.assertFalse(runtime.ready)
        self.assertIn("ffmpeg", runtime.error)
        with patch.dict(sys.modules, modules), patch("dit_svc.runtime.find_ffmpeg", return_value="ffmpeg"), redirect_stdout(io.StringIO()):
            runtime.ensure_env()
        self.assertTrue(runtime.env_ready)
        self.assertIsNone(runtime.error)


if __name__ == "__main__":
    unittest.main()
