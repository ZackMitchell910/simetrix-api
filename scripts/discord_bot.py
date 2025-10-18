from __future__ import annotations

import logging
import os
from typing import Optional

import discord
import httpx
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not DISCORD_BOT_TOKEN:
    raise RuntimeError("DISCORD_BOT_TOKEN must be set in the environment.")

SIMETRIX_API_BASE = (os.getenv("SIMETRIX_API_BASE") or "http://localhost:8000").rstrip("/")
SIMETRIX_API_KEY = os.getenv("SIMETRIX_API_KEY", "").strip()
SIMETRIX_DEFAULT_SYMBOL = os.getenv("SIMETRIX_DEFAULT_SYMBOL", "NVDA").upper()
SIMETRIX_DEFAULT_HORIZON = int(os.getenv("SIMETRIX_DEFAULT_HORIZON", "30"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simetrix-discord-bot")


class SimetrixBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.dm_messages = True
        super().__init__(command_prefix="!", intents=intents)
        self.add_command(commands.Command(self.sim, name="sim"))

    async def on_ready(self) -> None:
        logger.info(
            "Logged in as %s (ID: %s)",
            self.user,
            self.user.id if self.user else "unknown",
        )

    async def fetch_summary(self, symbol: str, horizon: int) -> dict:
        params = {"symbol": symbol, "horizon_days": horizon}
        headers = {"Accept": "application/json"}
        if SIMETRIX_API_KEY:
            headers["X-API-Key"] = SIMETRIX_API_KEY
        url = f"{SIMETRIX_API_BASE}/quant/daily/today"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "application/json" not in content_type.lower():
                preview = response.text[:200]
                raise ValueError(
                    f"Unexpected content type '{content_type}' from {url}. "
                    f"Preview: {preview}"
                )
            try:
                return response.json()
            except ValueError as exc:
                preview = response.text[:200]
                raise ValueError(f"Failed to decode JSON from {url}. Preview: {preview}") from exc

    async def sim(
        self,
        ctx: commands.Context,
        symbol: Optional[str] = None,
        horizon: Optional[int] = None,
    ) -> None:
        ticker = (symbol or SIMETRIX_DEFAULT_SYMBOL).upper()
        horizon_days = int(horizon or SIMETRIX_DEFAULT_HORIZON)
        try:
            data = await self.fetch_summary(ticker, horizon_days)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            await ctx.send(
                f"[warning] API error for {ticker}: {exc.response.status_code} {detail}"
            )
            return
        except Exception as exc:
            await ctx.send(f"[warning] Unexpected error fetching data for {ticker}: {exc}")
            return

        message = (
            f"**{data['symbol']} ({horizon_days}d)**\n"
            f"- Probability Up: {data['prob_up_30d']:.1%}\n"
            f"- Base Price: ${data['base_price']:.2f}\n"
            f"- Predicted Price (mode): ${data['predicted_price']:.2f}\n"
            f"- Bullish Price (p95): ${data['bullish_price']:.2f}"
        )
        await ctx.send(message)


def main() -> None:
    bot = SimetrixBot()
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
