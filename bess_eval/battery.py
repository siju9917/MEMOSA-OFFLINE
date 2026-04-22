"""Battery physics model."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class BatterySpec:
    power_kw: float
    energy_kwh: float
    rte_ac_ac: float              # round-trip AC-to-AC efficiency
    soc_min_frac: float
    soc_max_frac: float
    soc_init_frac: float
    aux_load_frac_per_day: float  # standby as fraction of rated energy/day
    coupling: str = "AC"

    @property
    def eta_chg(self) -> float:
        return np.sqrt(self.rte_ac_ac)

    @property
    def eta_dis(self) -> float:
        return np.sqrt(self.rte_ac_ac)

    @property
    def soc_min_kwh(self) -> float:
        return self.energy_kwh * self.soc_min_frac

    @property
    def soc_max_kwh(self) -> float:
        return self.energy_kwh * self.soc_max_frac

    @property
    def soc_init_kwh(self) -> float:
        return self.energy_kwh * self.soc_init_frac

    @property
    def aux_kw(self) -> float:
        """Continuous aux/standby draw (kW) to model."""
        return self.energy_kwh * self.aux_load_frac_per_day / 24.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "BatterySpec":
        return cls(
            power_kw=cfg["power_kw"],
            energy_kwh=cfg["energy_kwh"],
            rte_ac_ac=cfg["rte_ac_ac"],
            soc_min_frac=cfg["soc_min_frac"],
            soc_max_frac=cfg["soc_max_frac"],
            soc_init_frac=cfg["soc_init_frac"],
            aux_load_frac_per_day=cfg["aux_load_frac_per_day"],
            coupling=cfg.get("coupling", "AC"),
        )
