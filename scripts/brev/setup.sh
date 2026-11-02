#!/usr/bin/env bash
# Local entry point: copy source, install remotely, download MT50 and extract pool64.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
instance=primitive-pink-bovid
remote_dir=/data/vla-corrector-experiments
wait_seconds=900
foreground=false
while (($#)); do
    case "$1" in
        --instance) instance=${2:?}; shift 2 ;;
        --remote-dir) remote_dir=${2:?}; shift 2 ;;
        --wait-seconds) wait_seconds=${2:?}; shift 2 ;;
        --foreground) foreground=true; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/brev/setup.sh [--instance NAME] [--remote-dir /absolute/path] [--wait-seconds 900] [--foreground]'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$remote_dir" == /* && "$remote_dir" != / ]] || { echo '--remote-dir must be an absolute project path.' >&2; exit 2; }
[[ "$wait_seconds" =~ ^[0-9]+$ ]] || { echo '--wait-seconds must be an integer.' >&2; exit 2; }
for tool in brev git tar; do command -v "$tool" >/dev/null; done
brev ls
brev refresh
deadline=$((SECONDS + wait_seconds))
until brev exec "$instance" 'true'; do
    (( SECONDS < deadline )) || { echo "Timed out waiting for $instance." >&2; exit 1; }
    echo "Waiting for $instance SSH..."
    sleep 15
done
quoted_dir=$(brev_quote "$remote_dir")
brev exec "$instance" "mkdir -p $quoted_dir/outputs/brev && flock -n $quoted_dir/outputs/brev/gpu.lock true"
archive=$(mktemp /tmp/vla-corrector-source.XXXXXX.tar.gz)
trap 'rm -f -- "$archive"' EXIT
cd -- "$BREV_REPO_ROOT"
# Include local edits and new scripts, omit ignored datasets, weights and secrets.
git ls-files --cached --others --exclude-standard -z |
    tar --exclude='.env' --exclude='.env.*' --exclude='*.pem' --exclude='*.key' \
        --null -T - -czf "$archive"
remote_archive="/tmp/$(basename "$archive")"
brev copy "$archive" "$instance:$remote_archive"
brev exec "$instance" "tar -xzf $(brev_quote "$remote_archive") -C $quoted_dir && rm -f $(brev_quote "$remote_archive")"
if $foreground; then
    brev exec "$instance" "cd $quoted_dir && bash scripts/brev/remote_setup.sh"
else
    brev exec "$instance" "cd $quoted_dir && python3 scripts/brev/launch.py setup"
    echo "Status: brev exec $instance 'cat $remote_dir/outputs/brev/setup.status'"
fi
