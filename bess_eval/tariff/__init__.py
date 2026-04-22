from .comed_delivery import ComEdVLLDelivery, is_on_peak_hour, on_peak_mask, us_federal_holidays
from .supply_index import IndexSupply
from .supply_freepoint import FreepointSupply

__all__ = [
    "ComEdVLLDelivery",
    "IndexSupply",
    "FreepointSupply",
    "is_on_peak_hour",
    "on_peak_mask",
    "us_federal_holidays",
]


def build_supply(cfg: dict):
    """Factory: pick supply-tariff class by the config's `primary` field."""
    kind = cfg.get("primary", "index")
    if kind == "index":
        return IndexSupply(cfg["index"])
    if kind == "freepoint":
        return FreepointSupply(cfg["freepoint"])
    raise ValueError(f"Unknown supply tariff kind: {kind!r}")
