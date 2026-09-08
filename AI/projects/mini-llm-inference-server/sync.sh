#!/usr/bin/env bash
# Sync this project with the Elice GPU box and run things there.
# Local git is the source of truth; the box is disposable.
#
#   ./sync.sh push              local -> box   (mirrors: --delete removes box-only files)
#   ./sync.sh pull              box -> local   (never deletes; skips newer local files)
#   ./sync.sh diff              preview what pull would change
#   ./sync.sh run <cmd...>      push, then run <cmd> on the box venv
#   ./sync.sh py <script.py>    push, then run a python script on the box
#   ./sync.sh gpu               nvidia-smi on the box
#   ./sync.sh shell             interactive ssh
#
# Override the connection if the Elice port changes:
#   ELICE_PORT=34224 ./sync.sh push
#
# Direction matters. `push` mirrors, so anything edited only on the box is
# destroyed. `pull` uses --update, so a file you just edited locally is not
# clobbered by an older copy on the box. Run `diff` first if unsure.

set -euo pipefail

KEY="${ELICE_KEY:-$HOME/elice-cloud-ondemand-73195e20-1640-4600-b8ee-953d2ca6b2c3.pem}"
HOST="${ELICE_HOST:-central-01.tcp.tunnel.elice.io}"
PORT="${ELICE_PORT:-34224}"   # the 2x A100 box
USER_="${ELICE_USER:-elicer}"

REMOTE_DIR="${ELICE_DIR:-/home/elicer/mini-llm}"
VENV="$REMOTE_DIR/.venv/bin"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SSH=(ssh -i "$KEY" -p "$PORT" -o ConnectTimeout=20)
RSH="ssh -i $KEY -p $PORT -o ConnectTimeout=20"
TARGET="$USER_@$HOST"

# Keep weights, venv, and caches out of the transfer.
EXCLUDES=(
  --exclude '.git/'
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude 'hf/'
  --exclude '.DS_Store'
)

push() {
  echo "==> push  local  ->  $TARGET:$REMOTE_DIR/"
  rsync -az --delete --itemize-changes "$@" "${EXCLUDES[@]}" \
    --exclude 'bench/results/' \
    -e "$RSH" "$LOCAL_DIR/" "$TARGET:$REMOTE_DIR/"
}

pull() {
  # --update: never replace a local file that is newer than the box's copy.
  # No --delete: pulling must not remove local files.
  echo "==> pull  $TARGET:$REMOTE_DIR/  ->  local"
  rsync -az --update --itemize-changes "$@" "${EXCLUDES[@]}" \
    -e "$RSH" "$TARGET:$REMOTE_DIR/" "$LOCAL_DIR/"
}

case "${1:-}" in
  push)  push ;;
  pull)  pull ;;
  diff)  echo "(preview only, nothing written)"; pull --dry-run ;;
  run)   shift; push >/dev/null; echo "==> run: $*"
         "${SSH[@]}" "$TARGET" "cd $REMOTE_DIR && export PATH=$VENV:\$PATH HF_HOME=$REMOTE_DIR/hf && $*" ;;
  py)    shift; push >/dev/null; echo "==> python $*"
         "${SSH[@]}" "$TARGET" "cd $REMOTE_DIR && HF_HOME=$REMOTE_DIR/hf $VENV/python $*" ;;
  gpu)   "${SSH[@]}" "$TARGET" 'nvidia-smi' ;;
  shell) exec "${SSH[@]}" "$TARGET" ;;
  *)     sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
