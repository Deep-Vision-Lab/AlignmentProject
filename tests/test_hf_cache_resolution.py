"""Exercise startup cache discovery without importing CUDA/JAX training modules."""
import ast
import contextlib
import io
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODEL = "aubmindlab/bert-base-arabertv02"
SLUG = "models--aubmindlab--bert-base-arabertv02"


def startup_functions():
    tree = ast.parse((ROOT / "train.py").read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_cached_model_directory", "_resolve_hf_home"}]
    namespace = {"os": os, "Path": Path, "P": SimpleNamespace(arabic_text_model_name=MODEL)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "train.py", "exec"), namespace)
    return namespace


def encoder_cache_directory():
    tree = ast.parse((ROOT / "arabic_span_text_encoder_legacy.py").read_text())
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "cache_dir" for t in node.targets))
    return eval(compile(ast.Expression(assignment.value), "encoder-cache", "eval"), {"os": os})


class CacheResolutionTests(unittest.TestCase):
    def test_discovery_and_encoder_agree(self):
        for layout in ("flat", "hub"):
            for source in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "project", "XDG_CACHE_HOME"):
                with self.subTest(layout=layout, source=source), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    project = root / "project"
                    if source == "project":
                        home = project / ".hf_cache"
                        env = {}
                    elif source == "XDG_CACHE_HOME":
                        home = root / "xdg" / "huggingface"
                        env = {source: str(root / "xdg")}
                    else:
                        home = root / "custom"
                        env = {source: str(home)}
                    cache = home if layout == "flat" else home / "hub"
                    snapshot = cache / SLUG / "snapshots" / "revision"
                    snapshot.mkdir(parents=True)
                    (snapshot / "config.json").write_text("{}")
                    (snapshot / "model.safetensors").touch()
                    namespace = startup_functions()
                    namespace["PROJECT_DIR"] = project
                    with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(io.StringIO()):
                        namespace["_resolve_hf_home"]()
                        self.assertEqual(Path(encoder_cache_directory()), cache.resolve())
                        self.assertEqual(os.environ["HF_HUB_CACHE"], str(cache.resolve()))
                        self.assertNotIn("TRANSFORMERS_CACHE", os.environ)

    def test_config_without_weights_is_not_accepted(self):
        with TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / SLUG / "snapshots" / "revision"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            self.assertIsNone(startup_functions()["_cached_model_directory"](Path(tmp), MODEL))


if __name__ == "__main__":
    unittest.main()
