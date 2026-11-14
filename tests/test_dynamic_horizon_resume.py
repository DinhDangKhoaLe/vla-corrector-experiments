import csv
import json

import pytest

from dynamic_horizon.resume import prepare_output


def _completed(path):
    path.mkdir()
    (path / "metrics_summary.json").write_text(json.dumps({"tasks": 2, "episodes": 4}))
    with (path / "per_task_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_group", "task_id", "episodes"])
        writer.writeheader()
        writer.writerows([dict(task_group="easy", task_id=i, episodes=2) for i in range(2)])
    (path / "videos").mkdir()
    for i in range(4):
        (path / "videos" / f"ep{i}.mp4").write_bytes(b"video")


def test_resume_preserves_and_skips_completed_result(tmp_path):
    path = tmp_path / "baseline"
    _completed(path)
    assert prepare_output(path, {"episodes": 2}, 2, True)
    assert len(list((path / "videos").glob("*.mp4"))) == 4
    assert not list(tmp_path.glob("*.incomplete_*"))


def test_resume_archives_partial_output_without_deleting_it(tmp_path):
    path = tmp_path / "corrector"
    path.mkdir()
    (path / "resolved_experiment.yaml").write_text("partial run")
    assert not prepare_output(path, {"episodes": 2}, 2, True)
    assert not path.exists()
    archived, = tmp_path.glob("corrector.incomplete_*")
    assert (archived / "resolved_experiment.yaml").read_text() == "partial run"


def test_resume_does_not_skip_missing_video_or_changed_settings(tmp_path):
    path = tmp_path / "baseline"
    _completed(path)
    (path / "evaluation_request.json").write_text(json.dumps({"episodes": 2, "horizon": 50}))
    with pytest.raises(ValueError, match="Resume settings differ"):
        prepare_output(path, {"episodes": 2, "horizon": 10}, 2, True)
    assert path.exists()
    (path / "videos/ep0.mp4").unlink()
    assert not prepare_output(path, {"episodes": 2, "horizon": 50}, 2, True)
