"""Synthesize a 2025 hourly solar AC-output time series for the Joliet site.

Approach:
  1. Compute hourly clear-sky GHI for 2026 Jan-Apr using pvlib at (lat, lon, elevation).
  2. Fit a single linear scaling `k` such that k * clear_sky_ghi[h] * clear_sky_fraction[month]
     best matches the observed 2026 meter production over that window. `k` captures system
     DC size × tilt/azimuth-driven transposition × inverter derate.
  3. For 2025 hours, compute clear-sky GHI and multiply by the same `k` and the monthly
     climatological clear-sky fraction. Apply AC inverter cap.
  4. Clip negatives to zero.

Caveats:
  - No site-specific cloud data — monthly average clear-sky fraction only
  - Captures mean seasonal production; does NOT capture hour-to-hour cloud variability
  - For a BESS value study this is acceptable since battery dispatch decisions care about
    average solar over each hour, not instantaneous spikes
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pvlib


def _clear_sky_ghi_series(index: pd.DatetimeIndex, lat: float, lon: float, alt_m: float) -> pd.Series:
    """Hourly clear-sky GHI at the site using pvlib's Ineichen model."""
    loc = pvlib.location.Location(latitude=lat, longitude=lon, altitude=alt_m)
    cs = loc.get_clearsky(index, model="ineichen")
    return cs["ghi"]


def _tilt_pv_factor(index: pd.DatetimeIndex, lat: float, lon: float, alt_m: float,
                    tilt: float, azimuth: float) -> pd.Series:
    """POA-like factor = cos(AOI) × clear-sky GHI fraction captured by a tilted array.

    Simple analytical: POA_direct ≈ GHI_direct × cos(AOI) + GHI_diffuse × (1+cos(tilt))/2.
    For a BESS study the resolution here is fine.
    """
    loc = pvlib.location.Location(latitude=lat, longitude=lon, altitude=alt_m)
    sp = loc.get_solarposition(index)
    aoi = pvlib.irradiance.aoi(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=sp["apparent_zenith"].values,
        solar_azimuth=sp["azimuth"].values,
    )
    cos_aoi = np.clip(np.cos(np.radians(aoi)), 0, 1)
    cs = loc.get_clearsky(index, model="ineichen")
    dni = cs["dni"].values
    dhi = cs["dhi"].values
    ghi = cs["ghi"].values
    poa_dir = dni * cos_aoi
    sky_diffuse = dhi * (1 + np.cos(np.radians(tilt))) / 2
    gnd_reflected = ghi * 0.2 * (1 - np.cos(np.radians(tilt))) / 2
    poa = poa_dir + sky_diffuse + gnd_reflected
    return pd.Series(poa, index=index)


def fit_solar_model(measured_solar_kw: pd.Series, site_cfg: dict, solar_cfg: dict) -> dict:
    """Fit scale `k` so that k * POA * monthly_CSF ~= measured.

    Returns fit params and residual stats.
    """
    idx = measured_solar_kw.index
    poa = _tilt_pv_factor(
        idx,
        lat=site_cfg["lat"],
        lon=site_cfg["lon"],
        alt_m=site_cfg["elevation_m"],
        tilt=solar_cfg["tilt_deg"],
        azimuth=solar_cfg["azimuth_deg"],
    )
    csf_month = solar_cfg["clear_sky_fraction_monthly"]
    csf = pd.Series([csf_month[m] for m in idx.month], index=idx)
    envelope = poa * csf
    # least-squares fit k minimizing sum((measured - k*envelope)^2)
    mask = envelope > 10  # only daylight
    num = float((measured_solar_kw[mask] * envelope[mask]).sum())
    den = float((envelope[mask] ** 2).sum())
    k = num / den if den > 0 else 0.0
    pred = k * envelope
    resid = measured_solar_kw - pred
    return {
        "k": k,
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "mae": float(resid.abs().mean()),
        "measured_sum_kwh": float(measured_solar_kw.sum()),
        "predicted_sum_kwh": float(pred.sum()),
        "tilt": solar_cfg["tilt_deg"],
        "azimuth": solar_cfg["azimuth_deg"],
    }


def synthesize_year(year: int, tz: str, site_cfg: dict, solar_cfg: dict, k: float) -> pd.Series:
    """Produce hourly AC output in kW for all hours of `year` in site timezone."""
    idx = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="1h", tz=tz, inclusive="left")
    poa = _tilt_pv_factor(
        idx,
        lat=site_cfg["lat"],
        lon=site_cfg["lon"],
        alt_m=site_cfg["elevation_m"],
        tilt=solar_cfg["tilt_deg"],
        azimuth=solar_cfg["azimuth_deg"],
    )
    csf_month = solar_cfg["clear_sky_fraction_monthly"]
    csf = pd.Series([csf_month[m] for m in idx.month], index=idx)
    ac_kw = k * poa * csf
    inv_cap = solar_cfg.get("ac_inverter_limit_kw")
    if inv_cap:
        ac_kw = ac_kw.clip(upper=inv_cap)
    ac_kw = ac_kw.clip(lower=0)
    ac_kw.name = "solar_kw"
    return ac_kw
