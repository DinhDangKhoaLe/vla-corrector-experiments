"""One terminal-width progress line per rollout, including piped evaluations."""
import os
import shutil

from tqdm import tqdm


def episode_progress(total: int, description: str, *, disable: bool = False):
    # stderr is piped to the log collector, but stdin may still be the terminal.
    columns = shutil.get_terminal_size((80, 24)).columns
    for fd in (0, 2, 1):
        try:
            columns = os.get_terminal_size(fd).columns
            break
        except OSError:
            pass
    return tqdm(total=total, desc=description, position=0, leave=True,
                ncols=max(20, columns - 1), mininterval=0.25, miniters=1,
                ascii=True, disable=disable,
                bar_format="{desc:.24} |{bar}| {n_fmt}/{total_fmt}{postfix:.20}")
