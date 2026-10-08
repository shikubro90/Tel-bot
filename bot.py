"""Private Telegram assistant backed by a local Ollama model.

Only the owner (OWNER_ID) can talk to it. Chat history and remembered facts
are stored in a local SQLite file next to this script.

Ollama answers everything by default. Claude is an optional helper, used only
when the local model decides it needs fresh or expert information, when the
owner sends a photo, or on /ask. Claude never sees the chat history or the
remembered facts, only the single question or photo.

The bot can run on a server while Ollama stays on the owner's Mac. When the
Mac is off, the bot says so and Claude gives a short answer instead.
"""

import asyncio
import base64
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import anthropic
import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OWNER_ID = os.getenv("OWNER_ID", "").strip()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Buddy")
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_DAILY_LIMIT = int(os.getenv("CLAUDE_DAILY_LIMIT", "20"))
DB_PATH = BASE_DIR / "memory.db"

TELEGRAM_MAX_LEN = 4096
# The local model saying it will ask Claude, e.g. "Let me ask Claude."
ASKS_CLAUDE = re.compile(
    r"\b(ask|check with|consult|get help from|reach out to)\s+(my\s+)?(\w+\s+){0,3}?claude\b",
    re.IGNORECASE,
)
CLAUDE_MAX_CONTINUATIONS = 3
PHOTO_MAX_SIDE = 800

claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("assistant")


# ---------- memory ----------

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL UNIQUE COLLATE NOCASE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS claude_usage (
                day TEXT PRIMARY KEY,
                calls INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0
            );
            """
        )
    os.chmod(DB_PATH, 0o600)  # readable by this Mac user only


def save_message(role: str, content: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)", (role, content)
        )


def recent_messages(limit: int) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def save_fact(fact: str) -> bool:
    fact = fact.strip()
    if not fact:
        return False
    with db() as conn:
        cur = conn.execute("INSERT OR IGNORE INTO facts (fact) VALUES (?)", (fact,))
        return cur.rowcount > 0


def all_facts() -> list[sqlite3.Row]:
    with db() as conn:
        return conn.execute("SELECT id, fact FROM facts ORDER BY id").fetchall()


def delete_fact(fact_id: int) -> bool:
    with db() as conn:
        return conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,)).rowcount > 0


def clear_history() -> None:
    with db() as conn:
        conn.execute("DELETE FROM messages")


def clear_facts() -> None:
    with db() as conn:
        conn.execute("DELETE FROM facts")


def claude_usage_today() -> sqlite3.Row:
    with db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO claude_usage (day) VALUES (?)", (date.today().isoformat(),)
        )
        return conn.execute(
            "SELECT * FROM claude_usage WHERE day = ?", (date.today().isoformat(),)
        ).fetchone()


def record_claude_usage(input_tokens: int, output_tokens: int) -> None:
    claude_usage_today()
    with db() as conn:
        conn.execute(
            "UPDATE claude_usage SET calls = calls + 1, input_tokens = input_tokens + ?, "
            "output_tokens = output_tokens + ? WHERE day = ?",
            (input_tokens, output_tokens, date.today().isoformat()),
        )


def claude_available() -> bool:
    return claude is not None and claude_usage_today()["calls"] < CLAUDE_DAILY_LIMIT


# ---------- local model ----------

def system_prompt() -> str:
    facts = "\n".join(f"- {r['fact']}" for r in all_facts()) or "- (nothing yet)"
    now = datetime.now().strftime("%A, %d %B %Y, %H:%M")
    return (
        f"You are {ASSISTANT_NAME}, the user's personal assistant and close friend. "
        "Talk like a warm, relaxed friend: short, natural messages, no lectures, "
        "no bullet lists unless asked. Use what you know about the user naturally, "
        "without reciting it back. If you don't know something, say so.\n\n"
        f"If asked what you run on: you are the open model {OLLAMA_MODEL}, running "
        "locally on the user's own Mac through Ollama. Never invent another "
        "model name.\n\n"
        "Claude is a cloud model that can help you. When you cannot answer "
        'something yourself, reply with only "Let me ask Claude." and nothing '
        "else; the answer is then looked up and given to you. Never say that "
        "Claude said, did or will do anything unless it is in looked-up "
        "information in square brackets.\n\n"
        "Sometimes a message is followed by looked-up information in square "
        "brackets. Trust it over your own memory and base your answer on it, "
        "keeping the facts, numbers and dates exactly as given.\n\n"
        f"Current date and time: {now}\n\n"
        f"What you know about the user:\n{facts}"
    )


async def ollama_online() -> bool:
    """Quick check that Ollama answers; False when the Mac is off or asleep."""
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            return (await client.get(f"{OLLAMA_URL}/api/version")).status_code == 200
    except httpx.HTTPError:
        return False


async def ollama_chat(messages: list[dict], json_mode: bool = False) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
    }
    if json_mode:
        payload["format"] = "json"
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


async def learn_from(user_text: str) -> None:
    """Pull lasting facts about the user out of a message and store them."""
    prompt = (
        "From the user's message below, extract facts about the user worth "
        "remembering long-term (name, family, work, preferences, habits, plans, "
        "important dates). Ignore small talk, questions and one-off requests. "
        "Never store passwords, card numbers or ID numbers. "
        'Reply with JSON only: {"facts": ["short fact", ...]}. '
        'If there is nothing worth remembering, reply {"facts": []}.\n\n'
        f"Message: {user_text}"
    )
    try:
        raw = await ollama_chat([{"role": "user", "content": prompt}], json_mode=True)
        facts = json.loads(raw).get("facts", [])
    except Exception as exc:
        log.warning("Could not extract facts: %s", exc)
        return
    for fact in facts:
        if isinstance(fact, str) and save_fact(fact):
            log.info("Remembered a new fact")


async def decide_route(user_text: str) -> tuple[str, str]:
    """Let the local model decide whether it needs Claude. Returns (route, question)."""
    context = "\n".join(
        f"{m['role']}: {m['content'][:300]}" for m in recent_messages(4)[:-1]
    )
    prompt = (
        "A small local assistant is chatting with its user. Decide whether it can "
        "answer the user's latest message by itself. Reply with JSON only: "
        '{"route": "local" | "web" | "expert", "question": "..."}.\n'
        '- "local": greetings, small talk, feelings, personal chat, anything about '
        "the user, opinions, simple writing help, common knowledge. This is the "
        "default. Choose it whenever you are unsure.\n"
        '- "web": the answer needs current or live information: news, weather, '
        "prices, scores, schedules, recent events, anything described as latest "
        "or today.\n"
        '- "expert": the answer needs precise facts, calculations or technical, '
        "medical or legal detail that a small model would likely get wrong.\n"
        'For "web" and "expert", "question" is the request rewritten as one '
        "complete standalone question, leaving out the user's personal details. "
        'For "local" it is an empty string.\n'
        "Asking for help costs money, so use it only when really needed.\n"
        "Examples:\n"
        '"how are you?" -> {"route": "local", "question": ""}\n'
        '"what is the capital of Japan?" -> {"route": "local", "question": ""}\n'
        '"explain what a database is" -> {"route": "local", "question": ""}\n'
        '"will it rain tomorrow in Dhaka?" -> {"route": "web", "question": "Will it rain tomorrow in Dhaka?"}\n'
        '"who won the match yesterday?" -> {"route": "web", "question": "Who won the match yesterday?"}\n'
        '"what dose of ibuprofen is safe?" -> {"route": "expert", "question": "What dose of ibuprofen is safe for an adult?"}\n\n'
        f"Earlier messages:\n{context or '(none)'}\n\n"
        f"Latest message: {user_text}"
    )
    try:
        raw = await ollama_chat([{"role": "user", "content": prompt}], json_mode=True)
        decision = json.loads(raw)
        route = decision.get("route", "local")
        question = str(decision.get("question", "")).strip()
    except Exception as exc:
        log.warning("Could not decide route, staying local: %s", exc)
        return "local", ""
    if route not in ("web", "expert") or not question:
        return "local", ""
    return route, question


# ---------- claude helper ----------

async def ask_claude(content, system: str, use_web: bool = False) -> Optional[str]:
    """One short, self-contained request to Claude. Returns None on any failure."""
    if not claude_available():
        return None
    haiku = CLAUDE_MODEL.startswith("claude-haiku")
    kwargs = {
        "model": CLAUDE_MODEL,
        "max_tokens": 16000,
        "system": system,
        "output_config": {"effort": "low"},
    }
    if haiku:
        # Haiku can skip thinking, which saves output tokens.
        kwargs["thinking"] = {"type": "disabled"}
    else:
        # If a request is declined on safety grounds, the API reruns it on a
        # fallback model inside the same call. Haiku has no such fallback.
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    if use_web:
        # One search per question keeps the search results, and the bill, small.
        kwargs["tools"] = [
            {
                "type": "web_search_20250305" if haiku else "web_search_20260209",
                "name": "web_search",
                "max_uses": 1,
            }
        ]
    messages = [{"role": "user", "content": content}]
    try:
        for _ in range(CLAUDE_MAX_CONTINUATIONS):
            response = await claude.beta.messages.create(messages=messages, **kwargs)
            record_claude_usage(response.usage.input_tokens, response.usage.output_tokens)
            log.info(
                "Claude call: %s input + %s output tokens",
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            if response.stop_reason != "pause_turn":
                break
            # A long web search paused; resend with the partial turn to resume.
            messages = [messages[0], {"role": "assistant", "content": response.content}]
    except anthropic.AuthenticationError:
        log.error("Claude rejected the API key. Check ANTHROPIC_API_KEY in .env.")
        return None
    except anthropic.RateLimitError:
        log.warning("Claude rate limit reached, answering locally instead.")
        return None
    except anthropic.APIStatusError as exc:
        log.error("Claude API error %s: %s", exc.status_code, exc.message)
        return None
    except anthropic.APIConnectionError:
        log.warning("Could not reach Claude, answering locally instead.")
        return None

    if response.stop_reason == "refusal":
        log.info("Claude declined the request.")
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or None


RESEARCH_SYSTEM = (
    "You are a research helper for another assistant, which will relay your "
    "answer to its user. Give only the answer, in at most 50 words: plain text, "
    "no markdown, no preamble, no sources, no extra background. Give dates for "
    "anything time-sensitive. If you cannot find or do not know the answer, say "
    "so in one sentence."
)

MAC_OFF_PREFIX = "Your Mac is off, so Claude is answering:"

MAC_OFF_NOTICE = (
    "Your Mac is off or asleep, so I can't get a response from Ollama, and I "
    "couldn't get an answer from Claude either. Message me again when the Mac "
    "is back on."
)

MAC_OFF_SYSTEM = (
    f"You are {ASSISTANT_NAME}, the user's friendly personal assistant. Answer "
    "the user's message in at most 40 words. Plain text, no markdown, no "
    "preamble."
)

VISION_SYSTEM = (
    "You describe photos for an assistant that cannot see them. Start with one "
    "sentence that sums up the photo, then only the details that matter, "
    "including any important visible text. If the user asked a question about "
    "the photo, answer it. At most 60 words, plain text, no markdown."
)


# ---------- telegram ----------

async def keep_typing(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    while True:
        await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
        await asyncio.sleep(4)


async def send_long(update: Update, text: str) -> None:
    for i in range(0, len(text), TELEGRAM_MAX_LEN):
        await update.message.reply_text(text[i : i + TELEGRAM_MAX_LEN])


async def reply_without_mac(update: Update, user_text: str, info: Optional[str]) -> None:
    """Ollama is unreachable: say so, and let Claude give a short answer.

    The notice is a fixed line, so only the answer itself costs tokens.
    """
    log.info("Ollama unreachable, replying without it")
    # If Claude already looked this up, don't pay for a second request.
    answer = info
    if not answer and user_text:
        answer = await ask_claude(user_text, MAC_OFF_SYSTEM)
    reply = f"{MAC_OFF_PREFIX}\n\n{answer}" if answer else MAC_OFF_NOTICE
    save_message("assistant", reply)
    await send_long(update, reply)


async def reply_locally(
    update: Update, user_text: str, info: Optional[str] = None
) -> Optional[str]:
    """Answer the conversation so far with the local model.

    Returns the reply, or None when Ollama was unreachable and Claude filled in.
    """
    if not await ollama_online():
        await reply_without_mac(update, user_text, info)
        return None
    messages = [{"role": "system", "content": system_prompt()}]
    messages += recent_messages(HISTORY_LIMIT)
    try:
        reply = await ollama_chat(messages)
    except httpx.HTTPError as exc:
        log.error("Ollama request failed: %s", exc)
        await reply_without_mac(update, user_text, info)
        return None
    reply = reply or "…"
    save_message("assistant", reply)
    await send_long(update, reply)
    return reply


async def follow_up_with_claude(update: Update, user_text: str) -> None:
    """The local model said it would ask Claude: do it and send the answer unprompted."""
    log.info("Local model asked for Claude, following up")
    info = None
    if claude_available():
        info = await look_up(
            f"A user asked their Telegram assistant bot this: {user_text}", use_web=False
        )
    if info:
        await reply_locally(update, user_text, info)
    else:
        await update.message.reply_text(
            "I couldn't get an answer from Claude right now. It may not be set up, "
            "or I've used today's requests."
        )


async def look_up(question: str, use_web: bool) -> Optional[str]:
    """Ask Claude and add its answer to the conversation as looked-up notes."""
    info = await ask_claude(question, RESEARCH_SYSTEM, use_web=use_web)
    if info:
        save_message(
            "user",
            f"[Information looked up just now, not written by the user: {info}]",
        )
    return info


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    save_message("user", user_text)

    typing = asyncio.create_task(keep_typing(context, update.effective_chat.id))
    answered_locally = False
    try:
        info = None
        if claude_available() and await ollama_online():
            route, question = await decide_route(user_text)
            if route != "local":
                log.info("Asking Claude (%s)", route)
                info = await look_up(question, use_web=(route == "web"))
        reply = await reply_locally(update, user_text, info)
        answered_locally = reply is not None
        # If the model only promised to ask Claude, keep that promise now so
        # the user doesn't have to send another message.
        if reply and info is None and ASKS_CLAUDE.search(reply):
            await follow_up_with_claude(update, user_text)
    finally:
        typing.cancel()

    # Learning facts needs the local model, so it is skipped while the Mac is off.
    if answered_locally:
        context.application.create_task(learn_from(user_text))


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    caption = (update.message.caption or "").strip()
    if claude is None:
        await update.message.reply_text(
            "I can't see photos yet. Add ANTHROPIC_API_KEY to the .env file and restart me."
        )
        return
    if not claude_available():
        await update.message.reply_text(
            f"I've used all {CLAUDE_DAILY_LIMIT} of today's Claude requests, so I can't look at photos until tomorrow."
        )
        return

    typing = asyncio.create_task(keep_typing(context, update.effective_chat.id))
    try:
        # Telegram offers each photo in several sizes; a medium one costs far
        # fewer tokens than the full-size one and is enough to describe it.
        sizes = update.message.photo
        small_enough = [s for s in sizes if max(s.width, s.height) <= PHOTO_MAX_SIDE]
        photo_file = await (small_enough[-1] if small_enough else sizes[0]).get_file()
        image = base64.standard_b64encode(await photo_file.download_as_bytearray()).decode()
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": image},
            },
            {
                "type": "text",
                "text": f"The user's caption: {caption}" if caption else "Describe this photo.",
            },
        ]
        description = await ask_claude(content, VISION_SYSTEM)
        if not description:
            await update.message.reply_text("Sorry, I couldn't read that photo. Try again?")
            return

        save_message(
            "user",
            f"[The user sent a photo{' with the caption: ' + caption if caption else ''}. "
            f"What the photo shows: {description}]",
        )
        summary = description.split(". ")[0][:200]
        save_fact(f"Photo sent on {date.today():%d %B %Y}: {summary}")
        await reply_locally(update, caption, description)
    finally:
        typing.cancel()

    if caption:
        context.application.create_task(learn_from(caption))


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"Hey, I'm {ASSISTANT_NAME}. Just talk to me, or send me a photo.\n\n"
        "/ask <question> – look something up with Claude\n"
        "/usage – how much Claude I've used today\n"
        "/remember <fact> – tell me something to keep\n"
        "/memory – see what I know about you\n"
        "/forget <number> – remove one thing (or /forget all)\n"
        "/reset – clear our chat history, keep what I know"
    )


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = " ".join(context.args)
    if not question:
        await update.message.reply_text("Usage: /ask what's the weather in Dhaka today?")
        return
    if claude is None:
        await update.message.reply_text(
            "Claude isn't set up. Add ANTHROPIC_API_KEY to the .env file and restart me."
        )
        return
    save_message("user", question)
    typing = asyncio.create_task(keep_typing(context, update.effective_chat.id))
    try:
        info = await look_up(question, use_web=True)
        if not info and await ollama_online():
            await update.message.reply_text(
                "I couldn't get help from Claude just now, so this is only my own answer."
            )
        await reply_locally(update, question, info)
    finally:
        typing.cancel()


async def cmd_usage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if claude is None:
        await update.message.reply_text("Claude isn't set up, so I've used none.")
        return
    used = claude_usage_today()
    await update.message.reply_text(
        f"Claude today: {used['calls']} of {CLAUDE_DAILY_LIMIT} requests, "
        f"{used['input_tokens']} input and {used['output_tokens']} output tokens "
        f"({CLAUDE_MODEL})."
    )


async def cmd_remember(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    fact = " ".join(context.args)
    if not fact:
        await update.message.reply_text("Usage: /remember I'm allergic to peanuts")
    elif save_fact(fact):
        await update.message.reply_text("Got it, I'll remember that.")
    else:
        await update.message.reply_text("I already know that.")


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    facts = all_facts()
    if not facts:
        await update.message.reply_text("I don't know anything about you yet.")
        return
    await send_long(update, "\n".join(f"{r['id']}. {r['fact']}" for r in facts))


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    arg = context.args[0].lower() if context.args else ""
    if arg == "all":
        clear_facts()
        await update.message.reply_text("Done. I've forgotten everything about you.")
    elif arg.isdigit() and delete_fact(int(arg)):
        await update.message.reply_text("Forgotten.")
    else:
        await update.message.reply_text(
            "Usage: /forget 3 (see numbers with /memory) or /forget all"
        )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    clear_history()
    await update.message.reply_text("Chat history cleared. I still know you, though.")


async def sync_bot_name(app: Application) -> None:
    """Make the name shown at the top of the Telegram chat match ASSISTANT_NAME."""
    try:
        if (await app.bot.get_my_name()).name != ASSISTANT_NAME:
            await app.bot.set_my_name(ASSISTANT_NAME)
            log.info("Telegram display name set to %s", ASSISTANT_NAME)
    except TelegramError as exc:
        log.warning("Could not set the Telegram display name: %s", exc)


def main() -> None:
    if not BOT_TOKEN or not OWNER_ID.isdigit():
        sys.exit("Fill in TELEGRAM_BOT_TOKEN and OWNER_ID in the .env file first.")

    init_db()

    # Keep the Mac awake for as long as the bot is running (lid open, on power).
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])

    # Owner lock: every handler ignores anyone who isn't OWNER_ID.
    owner = filters.User(user_id=int(OWNER_ID))

    app = Application.builder().token(BOT_TOKEN).post_init(sync_bot_name).build()
    app.add_handler(CommandHandler("start", cmd_start, filters=owner))
    app.add_handler(CommandHandler("ask", cmd_ask, filters=owner))
    app.add_handler(CommandHandler("usage", cmd_usage, filters=owner))
    app.add_handler(CommandHandler("remember", cmd_remember, filters=owner))
    app.add_handler(CommandHandler("memory", cmd_memory, filters=owner))
    app.add_handler(CommandHandler("forget", cmd_forget, filters=owner))
    app.add_handler(CommandHandler("reset", cmd_reset, filters=owner))
    app.add_handler(MessageHandler(owner & filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(owner & filters.TEXT & ~filters.COMMAND, on_message))

    log.info(
        "%s is running with model %s at %s. Press Ctrl+C to stop.",
        ASSISTANT_NAME, OLLAMA_MODEL, OLLAMA_URL,
    )
    if claude is None:
        log.info("Claude helper is off (no ANTHROPIC_API_KEY in .env).")
    else:
        log.info("Claude helper is on: %s, up to %s requests a day.", CLAUDE_MODEL, CLAUDE_DAILY_LIMIT)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
