from __future__ import annotations

import os
from typing import Sequence

from src.core import settings

TOP_STOCKS = [
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "GOOGL",
    "META",
    "RIOT",
    "KR",
    "TSM",
    "AVGO",
    "TSLA",
    "WMT",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "MA",
    "PG",
    "JNJ",
    "COST",
    "HD",
    "ASML",
    "CVX",
    "ABBV",
    "TMUS",
    "MRK",
    "LLY",
    "WFC",
    "NFLX",
    "AMD",
    "KO",
    "BAC",
    "CRM",
    "ABT",
    "DHR",
    "TXN",
    "LIN",
    "ACN",
    "QCOM",
    "PM",
    "NEE",
    "COP",
    "ORCL",
    "GE",
    "AMGN",
    "T",
    "SPGI",
    "UBER",
    "ISRG",
    "RTX",
    "VZ",
    "PFE",
    "ABNB",
    "C",
    "ETN",
    "UNP",
    "IBM",
    "SYK",
    "BSX",
    "MU",
    "CAT",
    "SCHW",
    "KLAC",
    "TJX",
    "DE",
    "LMT",
    "MDT",
    "ADP",
    "GILD",
    "ZTS",
    "CB",
    "LOW",
    "HON",
    "USB",
    "INTU",
    "PGR",
    "BKNG",
    "AXP",
    "GS",
    "MMC",
    "BLK",
    "AMT",
    "PLD",
    "SBUX",
    "CMG",
    "BX",
    "REGN",
    "CBRE",
    "SNPS",
    "CDNS",
    "ICE",
    "PANW",
    "MELI",
    "ADI",
    "MDLZ",
    "MO",
    "CSX",
    "BMY",
    "KL",
    "STZ",
]

TOP_CRYPTOS = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "XRP-USD",
    "SOL-USD",
    "DOGE-USD",
    "TRX-USD",
    "ADA-USD",
    "HYPE-USD",
    "LINK-USD",
    "XLM-USD",
    "BCH-USD",
    "SUI-USD",
    "AVAX-USD",
    "LEO-USD",
    "HBAR-USD",
    "LTC-USD",
    "SHIB-USD",
    "MNT-USD",
    "TON-USD",
    "XMR-USD",
    "CRO-USD",
    "DOT-USD",
    "UNI-USD",
    "TAO-USD",
    "OKB-USD",
    "AAVE-USD",
    "ZEC-USD",
    "BGB-USD",
    "PEPE-USD",
    "NEAR-USD",
    "ENA-USD",
    "ASTER-USD",
    "APT-USD",
    "ETC-USD",
    "ONDO-USD",
    "POL-USD",
    "WLD-USD",
    "ICP-USD",
    "ARB-USD",
    "ALGO-USD",
    "ATOM-USD",
    "KAS-USD",
    "VET-USD",
    "PENGU-USD",
    "FLR-USD",
    "RENDER-USD",
    "SKY-USD",
    "GT-USD",
    "SEI-USD",
    "PUMP-USD",
    "CAKE-USD",
    "JUP-USD",
    "FIL-USD",
    "IMX-USD",
    "SPX-USD",
    "XDC-USD",
    "QNT-USD",
    "INJ-USD",
    "TIA-USD",
    "LDO-USD",
    "STX-USD",
    "OP-USD",
    "FET-USD",
    "AERO-USD",
    "CRV-USD",
    "NEXO-USD",
    "GRT-USD",
    "PYTH-USD",
    "KAIA-USD",
    "SNX-USD",
    "FLOKI-USD",
    "ATH-USD",
    "XTZ-USD",
    "ENS-USD",
    "ETHFI-USD",
    "MORPHO-USD",
    "PENDLE-USD",
    "IOTA-USD",
]


def _dedupe_upper(seq: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in seq:
        s = str(raw or "").strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def norm_crypto_symbol(symbol: str) -> str:
    """
    Normalize crypto tickers to BASE-USD (e.g., 'BTC' -> 'BTC-USD').
    """
    s = (symbol or "").strip().upper()
    if not s:
        return ""
    if s.startswith("X:"):
        return s
    if "-" in s:
        return s
    if s.endswith("USD"):
        base = s[:-3]
        return f"{base}-USD"
    return f"{s}-USD"


def _dedupe_crypto(seq: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in seq:
        sym = norm_crypto_symbol(raw)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def get_equity_watchlist(update_settings: bool = True) -> list[str]:
    equity_items = (
        settings.watchlist_equities.split(",")
        if settings.watchlist_equities
        else list(TOP_STOCKS)
    )
    watch = _dedupe_upper(equity_items)[:200]
    if update_settings:
        settings.equity_watch = watch.copy()
    return watch


def get_crypto_watchlist(update_settings: bool = True) -> list[str]:
    crypto_items = (
        settings.watchlist_cryptos.split(",")
        if settings.watchlist_cryptos
        else list(TOP_CRYPTOS)
    )
    watch = _dedupe_crypto(crypto_items)[:200]
    if update_settings:
        settings.crypto_watch = watch.copy()
    return watch


def get_retrain_daily_symbols() -> list[str]:
    if settings.retrain_daily_symbols:
        return _dedupe_upper(settings.retrain_daily_symbols.split(","))
    equity_watch = get_equity_watchlist(update_settings=False)
    return equity_watch[: min(15, len(equity_watch))]


def get_retrain_weekly_symbols() -> list[str]:
    if settings.retrain_weekly_symbols:
        return _dedupe_upper(settings.retrain_weekly_symbols.split(","))
    equity_watch = get_equity_watchlist(update_settings=False)
    crypto_watch = get_crypto_watchlist(update_settings=False)
    return _dedupe_upper(equity_watch[15:50] + crypto_watch[:15])


__all__ = [
    "TOP_STOCKS",
    "TOP_CRYPTOS",
    "get_equity_watchlist",
    "get_crypto_watchlist",
    "get_retrain_daily_symbols",
    "get_retrain_weekly_symbols",
    "norm_crypto_symbol",
]
