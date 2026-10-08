# Tel-bot

A private Telegram personal assistant. It answers only its owner, remembers
what it learns about them in a local SQLite file, and uses a local Ollama model
by default. Claude is an optional helper for fresh or expert information and
for photos.

## How it runs

- **Bot:** `bot.py`, on a small Linux server, as a systemd service.
- **Ollama:** on the owner's Mac. The Mac keeps a private SSH tunnel open to
  the server, so the bot reaches Ollama at `127.0.0.1:11434`.
- **Mac off or asleep:** the tunnel is down, so the bot says so and Claude
  gives a short answer instead.

## Settings

Copy these into a `.env` file next to `bot.py`. It is never committed.

| Setting | Meaning |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Token from @BotFather |
| `OWNER_ID` | Numeric Telegram ID of the only account allowed to use the bot (`whoami.py` finds it) |
| `ASSISTANT_NAME` | What the assistant calls itself |
| `OLLAMA_MODEL`, `OLLAMA_URL` | Local model and where to reach it |
| `HISTORY_LIMIT` | How many recent messages the model sees |
| `ANTHROPIC_API_KEY` | Optional. Leave empty to stay fully local |
| `CLAUDE_MODEL`, `CLAUDE_DAILY_LIMIT` | Claude model and the most requests per day |

## Run locally

```bash
./start.sh
```

## Deploy

1. On the server, as root: `TUNNEL_PUBKEY="<the Mac's public SSH key>" bash deploy/setup-vm.sh`
2. Copy `.env` to `/opt/tel-bot/.env` (owner `telbot`, mode 600) and run `systemctl start telbot`.
3. On the Mac, keep the tunnel open:
   `ssh -N -R 127.0.0.1:11434:127.0.0.1:11434 tunnel@<server>`

After that, deployment is automatic: every minute the server checks `main` on
GitHub and, once the CI checks for a new commit have passed, pulls it and
restarts the bot (`deploy/update.sh`).

## Commands

`/ask`, `/usage`, `/remember`, `/memory`, `/forget`, `/reset`

## Server logs

```bash
journalctl -u telbot -f          # the bot
journalctl -u telbot-update -n 20  # deployments
```
