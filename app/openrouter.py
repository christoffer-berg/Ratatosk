"""httpx client for OpenRouter with retries and price cache."""

import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 2  # beyond the first attempt


class OpenRouterError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class OpenRouterClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        pricing_refresh_hours: float = 24.0,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
        )
        self._pricing_ttl = pricing_refresh_hours * 3600
        self._prices: dict[str, tuple[float, float]] = {}
        self._prices_fetched_at: float = 0.0
        self._price_lock = asyncio.Lock()

    async def chat_completion(self, payload: dict) -> dict:
        """POST /chat/completions with exponential backoff on 429/5xx."""
        delay = 0.5
        last_error: OpenRouterError | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                last_error = OpenRouterError(f"{type(exc).__name__}: {exc}")
            else:
                if resp.status_code == 200:
                    return resp.json()
                last_error = OpenRouterError(
                    resp.text[:500], status_code=resp.status_code
                )
                if resp.status_code not in RETRYABLE_STATUS:
                    raise last_error
            if attempt < MAX_RETRIES:
                await asyncio.sleep(delay)
                delay *= 2
        assert last_error is not None
        raise last_error

    async def prices(self) -> dict[str, tuple[float, float]]:
        """Prices (USD per token, in/out) per model ID, cached from GET /models."""
        stale = (
            not self._prices
            or time.monotonic() - self._prices_fetched_at > self._pricing_ttl
        )
        if stale:
            async with self._price_lock:
                stale = (
                    not self._prices
                    or time.monotonic() - self._prices_fetched_at > self._pricing_ttl
                )
                if stale:
                    await self._refresh_prices()
        return self._prices

    async def _refresh_prices(self) -> None:
        resp = await self._client.get("/models")
        resp.raise_for_status()
        prices: dict[str, tuple[float, float]] = {}
        for model in resp.json().get("data", []):
            pricing = model.get("pricing") or {}
            try:
                prices[model["id"]] = (
                    float(pricing.get("prompt") or 0),
                    float(pricing.get("completion") or 0),
                )
            except (KeyError, TypeError, ValueError):
                continue
        self._prices = prices
        self._prices_fetched_at = time.monotonic()
        logger.info("Updated prices for %d models", len(prices))

    async def price_for(self, model: str) -> tuple[float, float]:
        """(price_in, price_out) in USD/token; (0, 0) if unknown or price cache is down."""
        try:
            prices = await self.prices()
        except Exception:
            logger.warning("Failed to fetch prices from OpenRouter", exc_info=True)
            return (0.0, 0.0)
        return prices.get(model, (0.0, 0.0))

    async def close(self) -> None:
        await self._client.aclose()
