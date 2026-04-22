from .comed_delivery import ComEdVLLDelivery, is_on_peak_hour, on_peak_mask, us_federal_holidays
from .peco_delivery import PECOHighTensionDelivery
from .supply_index import IndexSupply
from .supply_freepoint import FreepointSupply

__all__ = [
    "ComEdVLLDelivery",
    "PECOHighTensionDelivery",
    "IndexSupply",
    "FreepointSupply",
    "is_on_peak_hour",
    "on_peak_mask",
    "us_federal_holidays",
]


def build_delivery(cfg: dict):
    """Factory: pick delivery-tariff class by the config's `kind` field.
    Defaults to ComEd VLL Secondary for backward compatibility."""
    kind = cfg.get("kind", "comed_vll_secondary")
    if kind == "comed_vll_secondary":
        return ComEdVLLDelivery(cfg)
    if kind == "peco_ht_over_500kw":
        return PECOHighTensionDelivery(cfg)
    raise ValueError(f"Unknown delivery tariff kind: {kind!r}")


def build_supply(cfg: dict):
    """Factory: pick supply-tariff class by the config's `primary` field."""
    kind = cfg.get("primary", "index")
    if kind == "index":
        return IndexSupply(cfg["index"])
    if kind == "freepoint":
        return FreepointSupply(cfg["freepoint"])
    raise ValueError(f"Unknown supply tariff kind: {kind!r}")
