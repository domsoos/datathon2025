#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a UV-style 0–10 'Itch Index' for Hampton Roads from:
 - Mosquito trap counts with lat/lon
 - NWI wetlands summary with area by attribute code

Outputs:
 - outputs/itch_index_by_trap.csv
 - outputs/itch_index_latest_by_location.csv
 - outputs/itch_index_heatgrid.csv
 - outputs/itch_index_scatter.png
 - outputs/itch_index_bar_top_sites.png
"""

import argparse
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Helpers
def parse_date(s: str) -> pd.Timestamp:
    # Accept formats like "10/09/2025" or ISO
    return pd.to_datetime(s, errors="coerce")

def species_weight(species: str) -> float:
    """
    Heuristic species weights (0–1).
    - High annoyance/day-biter container breeders (Aedes albopictus etc.) score highest.
    - 'Males' score 0 (do not bite).
    - Unknowns get a middling score.
    """
    if not isinstance(species, str) or species.strip() == "":
        return 0.5

    s = species.lower().strip()

    # Explicit "Males"
    if s == "males":
        return 0.0

    # Aedes (e.g., albopictus, aegypti if present)
    if "aedes" in s:
        if "albopictus" in s:
            return 0.95  # slight headroom
        return 0.9

    # Culex complex
    if "culex" in s:
        return 0.6

    # Anopheles
    if "anopheles" in s:
        return 0.7

    # Coquillettidia, Psorophora, etc. (moderate to high biters)
    if any(x in s for x in ["coquillettidia", "psorophora", "mansonia", "ochlerotatus"]):
        return 0.7

    # Genus-only buckets like "Mansonia sp.", "Aedes spp."
    if any(x in s for x in [" spp", " sp.", "spp.", " genus"]):
        return 0.65

    return 0.5

def wetlands_code_weight(code: str) -> float:
    """
    Assign a breeding suitability weight to an NWI ATTRIBUTE code.
    The code often begins with a system class letter:
      E=Estuarine, P=Palustrine, R=Riverine, L=Lacustrine, U=Upland/Unknown (in some exports blank)
    We emphasize palustrine/emergent and estuarine marshes.
    """
    if not isinstance(code, str):
        return 0.4
    c = code.strip().upper()
    if c == "" or c == " ":
        return 0.4

    sys = c[0]
    base = {
        "P": 0.9,  # Palustrine
        "E": 0.75, # Estuarine
        "R": 0.55, # Riverine
        "L": 0.5,  # Lacustrine
        "U": 0.4,  # Upland/unknown
    }.get(sys, 0.5)

    boosters = 0
    if any(x in c for x in ["EM", "E1EM", "PEM", "E2EM"]):  # emergent marsh
        boosters += 0.15
    if any(x in c for x in ["SS", "PSS", "E2SS"]):          # scrub-shrub
        boosters += 0.07
    if any(x in c for x in ["FO", "PFO"]):                  # forested palustrine
        boosters += 0.05
    if any(x in c for x in ["H", "D"]):                     # impounded/ditched
        boosters += 0.05
    if any(x in c for x in ["4L", "4", "3"]):               # flood frequency hints
        boosters += 0.03

    return float(np.clip(base + boosters, 0.0, 1.0))

def compute_habitat_multiplier(wetlands_df: pd.DataFrame) -> float:
    """
    Compute region-wide Wetland Suitability Index (0–1) then map to 0.95–1.15 multiplier.
    """
    if wetlands_df.empty:
        return 1.0
    df = wetlands_df.copy()
    df["ATTRIBUTE"] = df["ATTRIBUTE"].astype(str)
    area_col = "SHAPEAREA" if "SHAPEAREA" in df.columns else None
    if area_col is None:
        return 1.0

    df["_w"] = df["ATTRIBUTE"].map(wetlands_code_weight)
    df["_aw"] = df["_w"] * pd.to_numeric(df[area_col], errors="coerce").fillna(0.0)
    total_area = pd.to_numeric(df[area_col], errors="coerce").fillna(0.0).sum()
    if total_area <= 0:
        return 1.0
    wsi = float(df["_aw"].sum() / total_area)  # 0..1
    multiplier = 0.95 + 0.2 * wsi              # 0.95..1.15
    return float(np.clip(multiplier, 0.95, 1.15))

def nice_bins():
    return [0, 2, 5, 7, 9, 10]

def itch_band(x: float) -> str:
    if x < 2: return "Low (0–2)"
    if x < 5: return "Moderate (3–4)"
    if x < 7: return "High (5–6)"
    if x < 9: return "Very High (7–8)"
    return "Extreme (9–10)"

def inverse_distance_weighting(xy_known, v_known, xy_query, power=2.0, eps=1e-9):
    """
    Simple IDW for quick grids (not for production kriging).
    xy_known: (n,2), v_known: (n,), xy_query: (m,2)
    """
    dx = xy_query[:, [0]] - xy_known[:, 0]
    dy = xy_query[:, [1]] - xy_known[:, 1]
    d2 = dx*dx + dy*dy
    w = 1.0 / np.power(d2 + eps, power/2.0)
    w_sum = w.sum(axis=1, keepdims=True)
    w_norm = w / np.maximum(w_sum, eps)
    return (w_norm @ v_known.reshape(-1,1)).ravel()

# Piecewise quantile mapping
def piecewise_quantile_map(x: np.ndarray, q10: float, q50: float, q90: float, q99: float) -> np.ndarray:
    """
    Map base risk values to 0..10 using fixed UV-like anchors:
      q10 -> 2, q50 -> 5, q90 -> 8.5, q99 -> 10 (cap).
    Linear segments between anchors; below q10 maps into 0..2.
    """
    # guard against degenerate quantiles
    eps = 1e-9
    q50 = max(q50, q10 + eps)
    q90 = max(q90, q50 + eps)
    q99 = max(q99, q90 + eps)

    y = np.zeros_like(x, dtype=float)

    # segment 0: (-inf, q10] -> [0, 2]
    m0 = 2.0 / max(q10, eps)
    y = np.where(x <= q10, m0 * x, 0.0)

    # segment 1: (q10, q50] -> (2, 5]
    m1 = (5.0 - 2.0) / (q50 - q10)
    b1 = 2.0 - m1 * q10
    y = np.where((x > q10) & (x <= q50), m1 * x + b1, y)

    # segment 2: (q50, q90] -> (5, 8.5]
    m2 = (8.5 - 5.0) / (q90 - q50)
    b2 = 5.0 - m2 * q50
    y = np.where((x > q50) & (x <= q90), m2 * x + b2, y)

    # segment 3: (q90, q99] -> (8.5, 10]
    m3 = (10.0 - 8.5) / (q99 - q90)
    b3 = 8.5 - m3 * q90
    y = np.where((x > q90) & (x <= q99), m3 * x + b3, y)

    # segment 4: > q99 -> 10
    y = np.where(x > q99, 10.0, y)

    return np.clip(y, 0, 10)

# Main pipeline
def main(trap_path: str, wetlands_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # Load traps
    traps = pd.read_csv(trap_path)
    # Normalize columns
    traps = traps.rename(columns={
        "Date": "date",
        "Trap Type": "trap_type",
        "Mosquito Species": "species",
        "Number of Mosquitos Collected": "count",
        "Address": "address",
        "Latitude": "lat",
        "Longitude": "lon",
    })

    traps["date"] = traps["date"].apply(parse_date)
    traps["count"] = pd.to_numeric(traps["count"], errors="coerce").fillna(0).astype(float)
    traps["lat"] = pd.to_numeric(traps["lat"], errors="coerce")
    traps["lon"] = pd.to_numeric(traps["lon"], errors="coerce")
    traps = traps.dropna(subset=["lat", "lon", "date"]).copy()

    # Species weights
    traps["species_weight"] = traps["species"].apply(species_weight)

    # Recency weights relative to 'now'
    now = pd.Timestamp(datetime.now(timezone.utc).date())
    traps["age_days"] = (now - traps["date"].dt.normalize()).dt.days.clip(lower=0)
    half_life_days = 12.0
    traps["recency_weight"] = np.power(0.5, traps["age_days"] / half_life_days)

    # Saturating abundance by trap type (bounded 0..1)
    # Use sqrt(count) to reflect diminishing returns; K = 70th pct of sqrt(count) per trap type
    traps["_sroot"] = np.sqrt(traps["count"].clip(lower=0))
    k_by_type = (
        traps.groupby("trap_type")["_sroot"]
             .quantile(0.70)
             .replace(0, np.nan)
    )
    global_k = traps["_sroot"].quantile(0.70)
    if not np.isfinite(global_k) or global_k <= 0:
        global_k = 1.0

    def abund_row(row):
        k = k_by_type.get(row["trap_type"], np.nan)
        if not np.isfinite(k) or k <= 0:
            k = global_k
        sroot = row["_sroot"]
        return float(sroot / (sroot + k))

    traps["abundance_score"] = traps.apply(abund_row, axis=1)

    # Wetlands habitat multiplier (regional)
    wetlands = pd.read_csv(wetlands_path)
    wetlands_multiplier = compute_habitat_multiplier(wetlands)

    # Composite base (bounded-ish)
    traps["base"] = (
        traps["abundance_score"]
        * traps["species_weight"].clip(0, 0.95)
        * traps["recency_weight"]
        * wetlands_multiplier
    )

    # alibration: recent quantile mapping
    # Build recent (60-day) calibration pool, exclude "Males"
    recent_cut = traps["date"].max() - pd.Timedelta(days=60)
    cal_pool = traps[
        (traps["date"] >= recent_cut) &
        (traps["species"].str.lower() != "males")
    ]["base"].replace([np.inf, -np.inf], np.nan).dropna()

    # Fallback if pool is too small
    if cal_pool.shape[0] < 50:
        cal_pool = traps.loc[traps["species"].str.lower() != "males", "base"].replace([np.inf,-np.inf], np.nan).dropna()

    if cal_pool.empty:
        # extreme fallback: direct 0..10 mapping of base
        traps["itch_index"] = (10.0 * traps["base"]).clip(0, 10)
        q10 = q50 = q90 = q99 = float("nan")
    else:
        q10 = float(np.nanpercentile(cal_pool, 10))
        q50 = float(np.nanpercentile(cal_pool, 50))
        q90 = float(np.nanpercentile(cal_pool, 90))
        q99 = float(np.nanpercentile(cal_pool, 99))
        traps["itch_index"] = piecewise_quantile_map(traps["base"].values, q10, q50, q90, q99)

    traps["itch_band"] = traps["itch_index"].apply(itch_band)

    # Export full per-reading results
    out_csv = os.path.join(outdir, "itch_index_by_trap.csv")
    traps_out = traps[[
        "date","trap_type","species","count","address","lat","lon",
        "species_weight","recency_weight","abundance_score",
        "base","itch_index","itch_band"
    ]].sort_values(["date","address","itch_index"], ascending=[False, True, False])
    traps_out.to_csv(out_csv, index=False)

    # Latest score per unique site (lat,lon rounded)
    traps["lat_r"] = traps["lat"].round(5)
    traps["lon_r"] = traps["lon"].round(5)
    idx_latest = traps.sort_values("date").groupby(["lat_r","lon_r"], as_index=False).tail(1)
    latest_cols = ["date","address","lat","lon","itch_index","itch_band","count","species"]
    latest = idx_latest.sort_values("itch_index", ascending=False)[latest_cols]
    out_latest = os.path.join(outdir, "itch_index_latest_by_location.csv")
    latest.to_csv(out_latest, index=False)

    # Quick visuals (scatter + bar)
    if not latest.empty:
        plt.figure(figsize=(7,6))
        sc = plt.scatter(latest["lon"], latest["lat"], s=80, c=latest["itch_index"], cmap=None)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Hampton Roads Itch Index (latest per trap location)")
        cbar = plt.colorbar(sc)
        cbar.set_label("Itch Index (0–10)")
        for _, row in latest.head(10).iterrows():
            plt.annotate(f'{row["itch_index"]:.1f}', (row["lon"], row["lat"]), xytext=(3,3), textcoords="offset points", fontsize=8)
        scatter_png = os.path.join(outdir, "itch_index_scatter.png")
        plt.tight_layout()
        plt.savefig(scatter_png, dpi=150)
        plt.close()

        # Top sites bar chart
        topn = latest.head(min(12, len(latest)))
        plt.figure(figsize=(8,6))
        ylabels = (topn["address"].fillna("").str.slice(0,35) + "…").where(topn["address"].str.len()>36, topn["address"])
        plt.barh(ylabels[::-1], topn["itch_index"][::-1])
        plt.xlabel("Itch Index (0–10)")
        plt.title("Top Itch Index sites (latest)")
        plt.tight_layout()
        bar_png = os.path.join(outdir, "itch_index_bar_top_sites.png")
        plt.savefig(bar_png, dpi=150)
        plt.close()

    # Lightweight heat grid (IDW) for mapping to parks later
    if len(latest) >= 3:
        lats = latest["lat"].values
        lons = latest["lon"].values
        v = latest["itch_index"].values

        # ~500m in lat/lon (~0.005 deg lat; lon step scaled by cos φ)
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        lat_step = 0.005
        lon_step = 0.005 * np.cos(np.deg2rad(np.clip(np.mean(lats), -89, 89)))
        lat_grid = np.arange(lat_min, lat_max + lat_step/2, lat_step)
        lon_grid = np.arange(lon_min, lon_max + lon_step/2, lon_step)

        grid_pts = np.array([(la, lo) for la in lat_grid for lo in lon_grid])
        known = np.column_stack([lats, lons])
        grid_values = inverse_distance_weighting(known, v, grid_pts, power=2.0)

        heat = pd.DataFrame({
            "lat": grid_pts[:,0],
            "lon": grid_pts[:,1],
            "itch_index_idw": grid_values
        })
        heat_csv = os.path.join(outdir, "itch_index_heatgrid.csv")
        heat.to_csv(heat_csv, index=False)

    # Print friendly summary
    summary = {
        "rows_in": len(traps),
        "sites_latest": len(latest),
        "wetlands_multiplier": round(compute_habitat_multiplier(pd.read_csv(wetlands_path)), 3),
        "half_life_days": half_life_days,
        "abundance_model": "saturating sqrt(count) / (sqrt(count)+K[trap_type], K=p70)",
        "calibration_quantiles_recent60d": {
            "q10": None if not np.isfinite(q10) else round(q10,6),
            "q50": None if not np.isfinite(q50) else round(q50,6),
            "q90": None if not np.isfinite(q90) else round(q90,6),
            "q99": None if not np.isfinite(q99) else round(q99,6),
        },
        "outputs": [out_csv, out_latest,
                    os.path.join(outdir, "itch_index_scatter.png"),
                    os.path.join(outdir, "itch_index_bar_top_sites.png"),
                    os.path.join(outdir, "itch_index_heatgrid.csv")],
    }
    print(pd.Series(summary))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traps", required=True, help="Path to Mosquito_Trap_Counts_YYYYMMDD.csv")
    ap.add_argument("--wetlands", required=True, help="Path to Hampton_Roads_NWI_Wetlands.csv")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    args = ap.parse_args()
    main(args.traps, args.wetlands, args.outdir)

