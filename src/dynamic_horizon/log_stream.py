"""Forward live console output but retain only final carriage-return lines in logs."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ANSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]")


def collect(source, console, log):
    pending = bytearray()
    carriage_return = False
    read = getattr(source, "read1", source.read)
    while chunk := read(4096):
        console.write(chunk)
        console.flush()
        for byte in chunk:
            if byte == 13:  # In-place progress refresh replaces the current line.
                carriage_return = True
            elif byte == 10:
                line = ANSI.sub(b"", bytes(pending)).rstrip()
                if line:
                    log.write(line + b"\n")
                pending.clear()
                carriage_return = False
            else:
                if carriage_return:
                    pending.clear()
                    carriage_return = False
                pending.append(byte)
        log.flush()
    if pending:
        line = ANSI.sub(b"", bytes(pending)).rstrip()
        if line:
            log.write(line + b"\n")
    log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    args = parser.parse_args()
    with args.log.open("ab") as log:
        collect(sys.stdin.buffer, sys.stdout.buffer, log)


if __name__ == "__main__":
    main()
