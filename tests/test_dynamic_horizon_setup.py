import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from dynamic_horizon.evaluate import command
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from siglip_dynamics.experiments.config import load_config
from siglip_dynamics.pace import PaceConfig, PaceSettings, calibrate


def test_video_dataset_auto_uses_lerobot_reader(tmp_path):
    import logging
    from siglip_dynamics.extract import _load_dataset
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"features": {"observation.images.image": {"dtype": "video"}}}))
    args = Namespace(dataset_path=str(tmp_path), dataset_repo_id="lerobot/libero", dataset_loader="auto")
    with patch("siglip_dynamics.extract.LeRobotDataset", autospec=True) as reader, patch("siglip_dynamics.extract.LocalParquetDataset") as parquet:
        ds, _ = _load_dataset(args, logging.getLogger(__name__))
        assert ds is reader.return_value
        parquet.assert_not_called()
    args.dataset_loader = "parquet"
    with pytest.raises(ValueError, match="Video-backed"):
        _load_dataset(args, logging.getLogger(__name__))


@pytest.mark.parametrize("horizon", [50, 40, 30, 20, 10])
@pytest.mark.parametrize("benchmark", ["metaworld", "libero", "robotwin"])
@pytest.mark.parametrize("method", ["baseline", "LACO_replan", "LACO_adjust", "corrector", "pace"])
def test_supported_execution_caps_preserve_prediction_horizon(horizon, benchmark, method):
    if method.startswith("LACO_") and benchmark == "robotwin":
        pytest.skip("LACO rollout supports MetaWorld and LIBERO")
    args = Namespace(benchmark=benchmark, method=method, policy=None, output=None,
                     horizon=horizon, seed=1000, episodes=20, no_ogg=False)
    cmd = command(args)
    assert any(f"_h{horizon}_seed1000" in option for option in cmd)
    if method.startswith("LACO_"):
        assert f"evaluation.action_horizon={horizon}" in cmd
    elif "siglip_dynamics.experiments.run" in cmd:
        assert f"evaluation.n_action_steps={horizon}" in cmd
    else:
        assert f"--policy.n_action_steps={horizon}" in cmd
        assert "--policy.chunk_size=50" in cmd


def test_typed_compatibility_view_loads_abstract_and_concrete(tmp_path):
    source, target = tmp_path / "source", tmp_path / "compat"
    source.mkdir()
    PI05Config(device="cpu").save_pretrained(source)
    config = json.loads((source / "config.json").read_text())
    config["use_relative_actions"] = False
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors").write_bytes(b"test asset")
    spec = importlib.util.spec_from_file_location("compat", "scripts/make_pi05_compat.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with patch("sys.argv", ["compat", "--source", str(source), "--output", str(target)]):
        mod.main()
    assert isinstance(PreTrainedConfig.from_pretrained(target), PI05Config)
    assert isinstance(PI05Config.from_pretrained(target), PI05Config)
    assert (target / "model.safetensors").is_symlink()
    assert "use_relative_actions" in json.loads((source / "config.json").read_text())


@pytest.mark.parametrize("benchmark", ["metaworld", "libero", "robotwin"])
def test_corrector_commands_have_matched_prediction_and_max_horizon(benchmark):
    cfg = load_config(f"configs/corrector/{benchmark}_original_pool64.yaml")
    assert cfg.model.image_token_grid_size == 8
    assert cfg.evaluation.chunk_size == cfg.evaluation.n_action_steps == 50
    assert cfg.evaluation.seed == 1000 and cfg.evaluation.episodes == 20
    args = Namespace(benchmark=benchmark, method="corrector", policy=None, output=None,
                     horizon=50, seed=1000, episodes=20, no_ogg=True)
    cmd = command(args)
    assert "evaluation.guidance=false" in cmd
    assert "evaluation.n_action_steps=50" in cmd


def test_libero_commands_preserve_checkpoint_camera_keys_and_suite_limits():
    import yaml
    from siglip_dynamics.experiments.run import eval_command
    from laco.utils import load_config as load_laco
    model = json.loads(Path("pretrained_weights/pi05_libero/config.json").read_text()) if Path("pretrained_weights/pi05_libero/config.json").exists() else None
    baseline = yaml.safe_load(Path("configs/baseline/libero_pi05.yaml").read_text())
    corrector = load_config("configs/corrector/libero_original_pool64.yaml")
    assert baseline["rename_map"] == corrector.evaluation.rename_map == {}
    assert not any(x.startswith("--env.episode_length=") for x in eval_command(corrector, "evaluate", "test.yaml"))
    for method in ("LACO_replan", "LACO_adjust"):
        config = load_laco(f"configs/laco/libero_{method}.yaml")
        assert config["evaluation"]["rename_map"] == {}
        assert config["evaluation"]["mode"] == method
        assert config["evaluation"]["episode_limit"] is None
        assert config["extract"]["normalize_actions"] is False
        assert config["extract"]["use_wrist_image"] is True
    if model:
        assert {"observation.images.image", "observation.images.image2"} <= set(model["input_features"])


def test_libero_extraction_wrapper_keeps_raw_actions(tmp_path):
    import os
    import subprocess
    # Intercept `python` at the shell boundary, without loading a VLA or touching a cache.
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    fake.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    result = subprocess.run(["bash", "scripts/extract_corrector_pool64.sh", "libero"], env=env,
                            text=True, capture_output=True, check=True)
    assert "--use-normalized-delta-action" not in result.stdout
    assert "--use-wrist-image" in result.stdout
    result = subprocess.run(["bash", "scripts/extract_corrector_pool64.sh", "metaworld"], env=env,
                            text=True, capture_output=True, check=True)
    assert "--use-normalized-delta-action" in result.stdout


def test_pace_calibration_reads_split_v3_parquet_and_task_labels(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = tmp_path / "data_source"
    (source / "meta").mkdir(parents=True)
    (source / "data").mkdir()
    pq.write_table(pa.table({"task_index": [0], "task": ["task"]}), source / "meta/tasks.parquet")
    actions = np.zeros((100, 4), dtype=np.float32)
    actions[:, 0] = 2 + np.cos(np.arange(100) * 2*np.pi/20)
    for file, lo, hi in [(0, 0, 45), (1, 45, 100)]:
        pq.write_table(pa.table({"episode_index": [0]*(hi-lo), "task_index": [0]*(hi-lo),
                                  "action": actions[lo:hi].tolist()}), source / f"data/file-{file:03}.parquet")
    artifact = calibrate(PaceConfig(source, tmp_path / "artifact.json", PaceSettings()))
    assert artifact["tasks"]["task"]["episode_count"] == 1
    assert artifact["tasks"]["task"]["threshold"] > 0
