#!/bin/bash
# One-time server setup for Debian or Ubuntu. Run as root:
#   TUNNEL_PUBKEY="ssh-ed25519 AAAA... name" bash setup-vm.sh
# TUNNEL_PUBKEY is the Mac's public SSH key; it may only open the Ollama tunnel.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/shikubro90/Tel-bot.git}"
APP_DIR=/opt/tel-bot

apt-get update -qq
apt-get install -y -qq git python3 python3-venv curl

# The bot runs as its own unprivileged user.
id -u telbot >/dev/null 2>&1 || useradd --system --home-dir "$APP_DIR" --shell /usr/sbin/nologin telbot
if [ ! -d "$APP_DIR/.git" ]; then
  git clone -q "$REPO_URL" "$APP_DIR"
fi
chown -R telbot:telbot "$APP_DIR"
runuser -u telbot -- python3 -m venv "$APP_DIR/.venv"
runuser -u telbot -- "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

# A second user that can do nothing except hold the tunnel to the Mac's Ollama.
if [ -n "${TUNNEL_PUBKEY:-}" ]; then
  id -u tunnel >/dev/null 2>&1 || useradd --create-home --shell /usr/sbin/nologin tunnel
  install -d -m 700 -o tunnel -g tunnel /home/tunnel/.ssh
  echo "restrict,port-forwarding,permitlisten=\"127.0.0.1:11434\" $TUNNEL_PUBKEY" \
    > /home/tunnel/.ssh/authorized_keys
  chown tunnel:tunnel /home/tunnel/.ssh/authorized_keys
  chmod 600 /home/tunnel/.ssh/authorized_keys

  # Drop the tunnel within 30 seconds of the Mac going silent, so the port is
  # free again when the Mac comes back.
  printf 'Match User tunnel\n    ClientAliveInterval 15\n    ClientAliveCountMax 2\n' \
    > /etc/ssh/sshd_config.d/70-telbot-tunnel.conf
  sshd -t
  systemctl reload ssh
fi

install -m 644 "$APP_DIR/deploy/telbot.service" "$APP_DIR/deploy/telbot-update.service" \
  "$APP_DIR/deploy/telbot-update.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable telbot
systemctl enable --now telbot-update.timer

echo "Setup done. Copy .env to $APP_DIR/.env (owner telbot, mode 600), then: systemctl start telbot"
