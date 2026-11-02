"""Detach a remote job completely from Brev's SSH connection (standard library only)."""

from datetime import datetime
import fcntl
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    mode, *arguments = sys.argv[1:]
    state = root / "outputs/brev"
    state.mkdir(parents=True, exist_ok=True)
    lock_name = "gpu.lock"
    if mode == "setup":
        if arguments:
            raise SystemExit("Setup accepts no additional launch arguments")
        command = ["bash", "scripts/brev/remote_setup.sh"]
        name = "setup"
    elif mode == "train":
        if not arguments or arguments[0] not in {"original", "prefix", "uncertainty", "all"}:
            raise SystemExit("Select original, prefix, uncertainty, or all")
        if not (state / "setup.status").read_text().startswith("complete "):
            raise SystemExit("Setup and extraction must complete before training")
        command = ["bash", "scripts/brev/train.sh", "--local", *arguments]
        name = f"train-{arguments[0]}"
    elif mode == "uncertainty-workflow":
        command = ["bash", "scripts/brev/uncertainty.sh", "--local", *arguments]
        name = "uncertainty-workflow"
        lock_name = "uncertainty-workflow.lock"
    else:
        raise SystemExit("Select setup, train, or uncertainty-workflow")
    # The child acquires this same lock for its entire lifetime.
    with (state / lock_name).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            message = ("An uncertainty workflow is already queued or running" if mode == "uncertainty-workflow"
                       else "Setup or training is already running")
            raise SystemExit(message) from None
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    log = state / f"{name}-{timestamp}-{time.time_ns()}.log"
    with log.open("xb") as output:
        process = subprocess.Popen(
            command, cwd=root, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
        )
    time.sleep(1)
    if process.poll() is not None and process.returncode:
        raise SystemExit(f"Job exited with {process.returncode}; inspect {log}")
    print(f"Launched {name}: pid={process.pid}\nLog: {log}\nStatus: {state / (name + '.status')}")


if __name__ == "__main__":
    main()
