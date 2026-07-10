"""Cost calculation per call based on usage and price cache."""


def attempt_cost(usage: dict | None, price_in: float, price_out: float) -> float:
    """Cost in USD: prompt_tokens * price_in + completion_tokens * price_out."""
    if not usage:
        return 0.0
    return (
        (usage.get("prompt_tokens") or 0) * price_in
        + (usage.get("completion_tokens") or 0) * price_out
    )
