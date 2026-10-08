#!/bin/bash
# Continuous deployment: when main on GitHub has a new commit whose CI checks
# all passed, pull it and restart the bot. Run every minute as root by
# telbot-update.timer.
set -euo pipefail

APP_DIR=/opt/tel-bot
REPO_API=https://api.github.com/repos/shikubro90/Tel-bot

as_bot() { runuser -u telbot -- "$@"; }

# Wrapped in a function so bash reads the whole script before git replaces it.
main() {
  cd "$APP_DIR"
  as_bot git fetch -q origin main
  local current latest status
  current=$(as_bot git rev-parse HEAD)
  latest=$(as_bot git rev-parse origin/main)
  if [ "$current" = "$latest" ]; then
    exit 0
  fi
  if [ -f .deploy-skip ] && [ "$(cat .deploy-skip)" = "$latest" ]; then
    exit 0
  fi

  status=$(curl -fsS -m 20 -H "Accept: application/vnd.github+json" \
    "$REPO_API/commits/$latest/check-runs" | python3 -c '
import json, sys
runs = json.load(sys.stdin)["check_runs"]
if not runs or any(r["status"] != "completed" for r in runs):
    print("pending")
elif all(r["conclusion"] in ("success", "skipped", "neutral") for r in runs):
    print("passed")
else:
    print("failed")
') || status=pending

  case "$status" in
    pending)
      exit 0  # checks still running; try again next minute
      ;;
    failed)
      echo "CI failed for $latest, not deploying it"
      echo "$latest" > .deploy-skip
      exit 0
      ;;
  esac

  as_bot git merge -q --ff-only origin/main
  as_bot .venv/bin/pip install -q -r requirements.txt
  install -m 644 deploy/telbot.service deploy/telbot-update.service \
    deploy/telbot-update.timer /etc/systemd/system/
  systemctl daemon-reload
  systemctl restart telbot
  echo "Deployed $latest"
}

main "$@"
