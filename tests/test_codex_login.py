# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Login-mode validation and LAB credential-home lifecycle regressions."""

import importlib.util
import json
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from harness2 import config, install
from harness2.substrate.codex import CodexSubstrate


class CodexLoginTests(unittest.TestCase):
    def test_codex_does_not_require_opencode_executor(self):
        with (
            patch.object(config, "SOLVER_MODEL_CX", "test-model"),
            patch.object(config, "SOLVER_MODEL_OC", ""),
            patch.object(config, "IMPROVER_MODEL", "fixture/improver"),
            patch.dict(os.environ, {"HARNESS2_SOLVER_MODEL": ""}),
        ):
            config.validate_models(substrate="cx")
            with self.assertRaises(config.ConfigError):
                config.validate_models(substrate="oc")

    def test_login_setup_does_not_overwrite_existing_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('model = "existing"\n')
            with (
                patch.object(config, "CODEX_AUTH_MODE", "chatgpt"),
                patch.object(config, "CODEX_HOME_DIR", Path(tmp)),
            ):
                install.configure_codex()
            self.assertEqual(path.read_text(), 'model = "existing"\n')

    def test_preflight_checks_login_without_endpoint_or_socket(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "codex").touch()
            with (
                patch.object(config, "CODEX_AUTH_MODE", "chatgpt"),
                patch.object(config, "CODEX_DIST", tmp),
                patch.object(config, "CODEX_BASE_URL", ""),
                patch("harness2.substrate.codex.socket.create_connection") as connect,
                patch("harness2.substrate.codex.subprocess.run") as run,
            ):
                run.return_value = subprocess.CompletedProcess([], 0, "", "Logged in using ChatGPT")
                CodexSubstrate().preflight()
                connect.assert_not_called()
                self.assertEqual(run.call_args.kwargs["env"]["CODEX_HOME"], str(config.CODEX_LOGIN_HOME))
                run.return_value = subprocess.CompletedProcess([], 1, "", "Not logged in")
                with self.assertRaisesRegex(config.ConfigError, "codex login"):
                    CodexSubstrate().preflight()
                run.side_effect = subprocess.TimeoutExpired("codex", 15)
                with self.assertRaisesRegex(config.ConfigError, "timed out"):
                    CodexSubstrate().preflight()

    def test_lab_login_home_survives_success_and_failure(self):
        source = Path(__file__).resolve().parents[1] / "third_party/lab_runner/adapt_runs_cx/run_codex_lab.py"
        fake = types.ModuleType("external.tts_lab")
        fake.config = types.SimpleNamespace()
        with patch.dict("sys.modules", {"external": types.ModuleType("external"), "external.tts_lab": fake}):
            spec = importlib.util.spec_from_file_location("lab_login_runner", source)
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
                root = Path(tmp)
                login = root / "login"
                login.mkdir()
                (login / "auth.json").write_text("fixture-login")
                (login / "config.toml").write_text("fixture-config")
                binary = root / "codex"
                binary.touch()
                reader = root / "read_doc.py"
                reader.touch()
                args = runner.parser.parse_args(["--task", "fixture", "--run-id", "run1", "--model", "fixture"])

                def execute(cmd, **kw):
                    self.assertEqual(kw["env"]["CODEX_HOME"], str(login))
                    self.assertIn('model_provider="openai"', cmd)
                    self.assertIn('forced_login_method="chatgpt"', cmd)
                    self.assertNotIn("OPENAI_API_KEY", kw["env"])
                    if fail:
                        raise OSError("fixture failure")
                    (Path(kw["cwd"]) / "output/answer.txt").write_text("done")
                    kw["stdout"].write(json.dumps({"type": "turn.completed", "usage": {}}) + "\n")
                    return subprocess.CompletedProcess(cmd, 0)

                with (
                    patch.multiple(runner, AUTH_MODE="chatgpt", LOGIN_HOME=login,
                                   SCRATCH_BASE=root / "scratch", RESULTS_ROOT=root / "results",
                                   CODEX_BIN=binary, READ_DOC=reader),
                    patch.object(runner, "C", types.SimpleNamespace(
                        task_json=lambda _: {"instructions": "fixture"},
                        documents_dir=lambda _: root / "no-docs")),
                    patch.object(runner.signal, "signal"),
                    patch.object(runner.subprocess, "run", side_effect=execute),
                ):
                    if fail:
                        with self.assertRaises(OSError):
                            runner.main(args)
                    else:
                        self.assertEqual(runner.main(args), 0)
                self.assertEqual((login / "auth.json").read_text(), "fixture-login")
                self.assertEqual((login / "config.toml").read_text(), "fixture-config")
                self.assertFalse(list((root / "scratch").rglob("auth.json")))
