import importlib.util
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from dynamic_horizon.assets import (
    ROBOTWIN_OFFICIAL_ARCHIVE, ROBOTWIN_OFFICIAL_DIRECTORY,
    ROBOTWIN_OFFICIAL_POLICY, ROBOTWIN_OFFICIAL_REPO,
    ROBOTWIN_OFFICIAL_REVISION, unpack_official_robotwin,
)


def downloader():
    spec = importlib.util.spec_from_file_location("download", "scripts/download_dynamic_horizon.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_download_selects_only_official_weights_and_clean_demonstrations():
    jobs = downloader().plan("all", ["*"])
    assert ("dataset", ROBOTWIN_OFFICIAL_REPO, ROBOTWIN_OFFICIAL_DIRECTORY, [ROBOTWIN_OFFICIAL_ARCHIVE]) in jobs
    assert ("dataset", ROBOTWIN_OFFICIAL_REPO, "datasets/robotwin_archives", ["dataset/*/aloha-agilex_clean_50.zip"]) in jobs
    assert not any("Myungkyu" in repo for _, repo, _, _ in jobs)
    assert len([job for job in jobs if job[0] == "model"]) == 2  # Only MetaWorld/LIBERO exports.


def test_official_download_is_pinned_and_never_calls_compatibility_converter():
    module = downloader()
    with patch("sys.argv", ["download", "--benchmark", "robotwin", "--execute"]), \
         patch("huggingface_hub.snapshot_download") as download, patch.object(module.subprocess, "run") as compat:
        module.main()
    assert download.call_count == 2
    assert download.call_args_list[0].kwargs["revision"] == ROBOTWIN_OFFICIAL_REVISION
    assert download.call_args_list[0].kwargs["repo_type"] == "dataset"
    assert download.call_args_list[0].kwargs["allow_patterns"] == [ROBOTWIN_OFFICIAL_ARCHIVE]
    compat.assert_not_called()


def _archive(path, member):
    with tarfile.open(path, "w:gz") as handle:
        value = b"original fixture contents"
        item = tarfile.TarInfo(member)
        item.size = len(value)
        handle.addfile(item, io.BytesIO(value))


def test_official_archive_unpacks_once_and_records_source_without_conversion(tmp_path):
    archive = tmp_path / "official.tar.gz"
    _archive(archive, "checkpoint/29999/params/test-data")
    output = tmp_path / "extracted"
    unpack_official_robotwin(archive, output)
    assert (output / "checkpoint/29999/params/test-data").read_bytes() == b"original fixture contents"
    source = json.loads((output / "source_archive.json").read_text())
    assert source["revision"] == ROBOTWIN_OFFICIAL_REVISION
    assert not (output / "config.json").exists()
    assert archive.exists()
    assert unpack_official_robotwin(archive, output) == output


def test_official_archive_rejects_traversal_and_preserves_existing_output(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    _archive(archive, "../outside")
    output = tmp_path / "extracted"
    with pytest.raises(tarfile.TarError):
        unpack_official_robotwin(archive, output)
    assert not (tmp_path / "outside").exists() and not output.exists()
    output.mkdir()
    (output / "keep").write_text("existing")
    with pytest.raises(FileExistsError):
        unpack_official_robotwin(archive, output)
    assert (output / "keep").read_text() == "existing"


def test_robotwin_recipes_all_require_the_official_export():
    import yaml
    from argparse import Namespace
    from dynamic_horizon.evaluate import command
    for name in ["baseline", "corrector"]:
        path = "configs/baseline/robotwin_pi05.yaml" if name == "baseline" else "configs/corrector/robotwin_original_pool64.yaml"
        assert yaml.safe_load(Path(path).read_text())["evaluation"]["policy"] == ROBOTWIN_OFFICIAL_POLICY
    for method in ["baseline", "corrector", "pace"]:
        cmd = command(Namespace(benchmark="robotwin", method=method, horizon=50, episodes=20,
                                seed=1000, policy=None, output=None, no_ogg=False))
        assert f"evaluation.policy={ROBOTWIN_OFFICIAL_POLICY}" in cmd


def test_clean_demo_unpack_resumes_only_complete_matching_archives(tmp_path, monkeypatch):
    import zipfile
    module = downloader()
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "archives"
    source = root / "dataset/adjust_bottle/aloha-agilex_clean_50.zip"
    source.parent.mkdir(parents=True)
    with zipfile.ZipFile(source, "w") as handle:
        handle.writestr("episodes/episode0.hdf5", b"small fixture")
    module.unpack_robotwin(root)
    module.unpack_robotwin(root)
    result = tmp_path / "datasets/robotwin/adjust_bottle"
    assert (result / "episodes/episode0.hdf5").read_bytes() == b"small fixture"
    assert source.exists()
    (result / "source_archive.json").unlink()
    with pytest.raises(FileExistsError):
        module.unpack_robotwin(root)


def test_robotwin_stops_before_rollout_when_official_export_is_missing(tmp_path, monkeypatch):
    from dynamic_horizon.evaluate import main
    monkeypatch.setattr("dynamic_horizon.assets.ROBOTWIN_OFFICIAL_POLICY", str(tmp_path / "not-converted"))
    with patch("sys.argv", ["evaluate", "--benchmark", "robotwin", "--method", "baseline"]), \
         patch("dynamic_horizon.evaluate.subprocess.run") as launch:
        with pytest.raises(FileNotFoundError, match="converted LeRobot export"):
            main()
        launch.assert_not_called()
