import io
from contextlib import redirect_stderr

from dynamic_horizon.log_stream import collect
from lerobot.utils.episode_progress import episode_progress


def test_episode_progress_uses_one_line_and_keeps_logs_compact(monkeypatch):
    monkeypatch.setattr("lerobot.utils.episode_progress.os.get_terminal_size", lambda *a: (_ for _ in ()).throw(OSError()))
    monkeypatch.setenv("COLUMNS", "80")
    raw = io.StringIO()
    with redirect_stderr(raw):
        for episode in (1, 2):
            with episode_progress(50, f"easy:0 ep={episode}/20") as bar:
                for _ in range(50):
                    bar.set_postfix({"calls": 1, "ok": "75%"}, refresh=False)
                    bar.update()
                    bar.refresh()  # Force updates to exercise the piped log path.
    output = raw.getvalue()
    assert output.count("\n") == 2
    assert "\x1b[A" not in output
    assert all(len(line) <= 79 for line in output.replace("\r", "\n").splitlines())
    console, log = io.BytesIO(), io.BytesIO()
    collect(io.BytesIO(output.encode()), console, log)
    assert console.getvalue() == output.encode()
    assert len(log.getvalue().splitlines()) == 2
    assert b"50/50" in log.getvalue()


def test_log_collector_preserves_errors_and_windows_newlines():
    source = b"Loading checkpoint\r\n\r10%\r50%\r100%\nTraceback:\n  error detail\n"
    console, log = io.BytesIO(), io.BytesIO()
    collect(io.BytesIO(source), console, log)
    assert console.getvalue() == source
    assert log.getvalue() == b"Loading checkpoint\n100%\nTraceback:\n  error detail\n"
