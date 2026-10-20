"""Guard against reintroducing dependencies on sibling forecasting models."""

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "forecast_backbones" / "m4_pinball"
PREFIX = "forecast_backbones.m4_pinball"


def test_imports_stay_within_this_model():
    for source in PACKAGE.rglob("*.py"):
        package = ".".join(
            (
                "forecast_backbones",
                "m4_pinball",
                *source.relative_to(PACKAGE).parts[:-1],
            )
        )
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = "." * node.level + (node.module or "")
                modules = [importlib.util.resolve_name(module, package)]
            for module in modules:
                if module.startswith("forecast_backbones."):
                    assert module == PREFIX or module.startswith(PREFIX + "."), (
                        source,
                        module,
                    )


@pytest.mark.parametrize("head", ["flatten", "step_cov"])
def test_training_and_inference_without_sibling_models(tmp_path, head):
    isolated = tmp_path / "isolated"
    shutil.copytree(
        PACKAGE,
        isolated / "forecast_backbones" / "m4_pinball",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for relative in (
        "interfaces/modeloutput.py",
        "pipelines/model_registry.py",
        "pipelines/transformer_dataset.py",
        "pipelines/pipeline_wrapper_inference.py",
    ):
        target = isolated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    assert not (isolated / "forecast_backbones" / "m4").exists()
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        class OwnModelOnly(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                prefix = 'forecast_backbones.m4_pinball'
                if fullname.startswith('forecast_backbones.') and not (
                    fullname == prefix or fullname.startswith(prefix + '.')
                ):
                    raise AssertionError('Sibling model dependency: ' + fullname)
                return None
        sys.meta_path.insert(0, OwnModelOnly())
        import torch
        from pathlib import Path
        from forecast_backbones.m4_pinball.config import load_config
        from forecast_backbones.m4_pinball.tests.test_pinball import make_frame, make_batch
        from forecast_backbones.m4_pinball.train import train
        from pipelines.model_registry import build_model
        from safetensors.torch import load_model
        torch.set_num_threads(1)
        cfg = load_config({
            'inference': {'input_len': 32, 'pred_len': 192},
            'model': {'prediction_head': 'HEAD_CHOICE',
                      'patch_len': 8, 'stride': 8, 'd_model': 16, 'n_heads': 2,
                      'e_layers': 1, 'd_ff': 32, 'embedding_dim': 16},
            'optimization': {'use_cpu': True, 'fp16': False, 'batch_size': 2,
                             'eval_batch_size': 2, 'train_epochs': 1},
        })
        out = Path('run')
        frame = make_frame(future=192)
        trainer = train(cfg, frame, frame, out, max_steps=1)
        restored = build_model('m4_pinball', cfg).eval()
        load_model(restored, str(out / 'model_result' / 'model.safetensors'))
        trainer.model.eval()
        with torch.no_grad():
            expected = trainer.model(make_batch(future=192)).logits
            actual = restored(make_batch(future=192)).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == (3, 192)
        assert trainer.state.best_metric is not None
        print('Independent train/save/reload/inference passed')
    """).replace("HEAD_CHOICE", head)
    env = {
        **os.environ,
        "PYTHONPATH": str(isolated),
        "HF_HOME": str(tmp_path / "hf"),
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=isolated,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Independent train/save/reload/inference passed" in result.stdout
