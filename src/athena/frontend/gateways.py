"""Gateway abstraction for external messaging platforms (Telegram, Discord, etc.)."""

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Callable

MessageHandler = Callable[[str, str], str]


class Gateway(ABC):
    """Base class for external messaging gateways.

    Each gateway receives messages from an external platform, injects them
    into the shared conversation via a message handler callback, and routes
    the LLM response back to the platform.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier like 'telegram', 'discord'."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Start listening for messages. Must be non-blocking (spawn a thread)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Shut down cleanly."""
        ...

    @abstractmethod
    def send_message(self, text: str) -> None:
        """Proactively send a message to the platform (e.g. scheduled task output).

        May silently no-op if the gateway doesn't have a known recipient yet.
        """
        ...

    def set_message_handler(self, handler: MessageHandler) -> None:
        """Register the callback that processes messages through the LLM."""
        self._message_handler = handler


class TelegramGateway(Gateway):
    """Telegram bot gateway using python-telegram-bot v20+."""

    name = "telegram"

    def __init__(self, bot_token: str, chat_id: int | None = None):
        """Initialize the Telegram gateway.

        Args:
            bot_token: Telegram Bot API token.
            chat_id: Optional chat ID to send proactive messages to. If
                ``None``, the gateway learns it from the first incoming message.
        """
        self._bot_token = bot_token
        self._tg_app = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._message_handler: MessageHandler | None = None
        self._chat_id: int | None = chat_id
        self._on_chat_id_changed: Callable[[int], None] | None = None

    def start(self) -> None:
        from telegram.ext import ApplicationBuilder, MessageHandler as TgHandler, filters

        self._tg_app = ApplicationBuilder().token(self._bot_token).build()
        self._tg_app.add_handler(
            TgHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

        self._thread = threading.Thread(target=self._run_polling, daemon=True)
        self._thread.start()

    def _run_polling(self) -> None:
        """Run the bot's polling loop in a dedicated background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._tg_app.initialize())
        self._loop.run_until_complete(self._tg_app.start())
        self._loop.run_until_complete(self._tg_app.updater.start_polling(poll_interval=1.0))
        self._loop.run_forever()

    _MAX_MESSAGE_LENGTH = 4096

    async def _on_message(self, update, context) -> None:
        """Handle an incoming Telegram message."""
        text = update.message.text
        if not text or not self._message_handler:
            return

        # Remember chat_id so we can proactively send messages later
        new_chat_id = update.message.chat_id
        if new_chat_id != self._chat_id:
            self._chat_id = new_chat_id
            if self._on_chat_id_changed:
                self._on_chat_id_changed(new_chat_id)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, self._message_handler, text, "telegram"
        )

        for chunk in self._split_message(response):
            await update.message.reply_text(chunk)

    def send_message(self, text: str) -> None:
        """Send a message to the most recent Telegram chat."""
        if not self._chat_id or not self._loop or not self._tg_app:
            print(f"[Telegram] send_message skipped: chat_id={self._chat_id}, "
                  f"loop={self._loop is not None}, app={self._tg_app is not None}")
            return

        async def _send():
            for chunk in self._split_message(text):
                await self._tg_app.bot.send_message(
                    chat_id=self._chat_id, text=chunk
                )

        future = asyncio.run_coroutine_threadsafe(_send(), self._loop)
        future.add_done_callback(self._log_send_result)

    @staticmethod
    def _log_send_result(future) -> None:
        """Log errors from proactive send_message calls."""
        exc = future.exception()
        if exc:
            print(f"[Telegram] send_message failed: {exc}", flush=True)

    @staticmethod
    def _split_message(text: str, limit: int = 4096) -> list[str]:
        """Split text into chunks that fit within Telegram's message limit.

        Tries to split on newlines first, then on spaces, to avoid breaking
        mid-word or mid-line.

        Args:
            text: The message text to split.
            limit: Maximum character length per chunk.

        Returns:
            List of text chunks, each at most *limit* characters long.
        """
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break

            # Try to split at last newline within limit
            split_at = text.rfind("\n", 0, limit)
            if split_at == -1:
                # Fall back to last space within limit
                split_at = text.rfind(" ", 0, limit)
            if split_at == -1:
                # Hard split as last resort
                split_at = limit

            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")

        return chunks

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


def create_gateways(config: dict) -> list[Gateway]:
    """Instantiate gateways from the ``gateways`` section of config.json.

    Args:
        config: Full application config dict (expects a ``gateways`` key).

    Returns:
        List of initialized (but not yet started) ``Gateway`` instances.
    """
    gateways_config = config.get("gateways", {})
    result: list[Gateway] = []

    if "telegram" in gateways_config:
        tg_conf = gateways_config["telegram"]
        bot_token = tg_conf.get("bot_token")
        if bot_token:
            chat_id = tg_conf.get("chat_id")
            result.append(TelegramGateway(bot_token, chat_id=chat_id))

    return result
