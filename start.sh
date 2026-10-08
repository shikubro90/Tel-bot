#!/bin/bash
# Sets up the Python environment when needed, then starts the bot.
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

# Reinstall libraries only when requirements.txt has changed.
if [ requirements.txt -nt .venv/.installed ]; then
  .venv/bin/pip install --quiet -r requirements.txt
  touch .venv/.installed
fi

exec .venv/bin/python bot.py
