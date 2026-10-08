"""Find your Telegram ID through your own bot and save it as OWNER_ID in .env.

Send any message to your bot in Telegram first, then run this.
"""

import re
import sys
from pathlib import Path

import httpx
from dotenv import dotenv_values

ENV_PATH = Path(__file__).parent / ".env"

token = (dotenv_values(ENV_PATH).get("TELEGRAM_BOT_TOKEN") or "").strip()
if not token:
    sys.exit("TELEGRAM_BOT_TOKEN is empty in .env. Paste your BotFather token there first.")

resp = httpx.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=30)
if resp.status_code == 401:
    sys.exit("Telegram rejected the token. Copy it again from BotFather into .env.")
resp.raise_for_status()

senders = [
    u["message"]["from"]
    for u in resp.json()["result"]
    if u.get("message", {}).get("chat", {}).get("type") == "private"
]
if not senders:
    sys.exit("No messages found. Open your bot in Telegram, send it 'hi', then run this again.")

me = senders[-1]
text = ENV_PATH.read_text()
text = re.sub(r"^OWNER_ID=.*$", f"OWNER_ID={me['id']}", text, flags=re.M)
ENV_PATH.write_text(text)

print(f"Saved OWNER_ID={me['id']} for {me.get('first_name', '')} (@{me.get('username', '-')}).")
print("If that is not you, clear OWNER_ID in .env and run this again after messaging the bot.")
