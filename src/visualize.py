#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds storytelling visuals for the Itch Index.
Inputs:
  - itch_index_by_trap.csv
  - itch_index_latest_by_location.csv 

Outputs (PNG/HTML) go to --outdir:
  - fig_hist_latest.png
  - fig_trend_rolling.png
  - fig_species_contrib.png
  - fig_hotspot_stability.png
  - fig_traptype_box.png
  - itch_map_banded.html
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def itch_band(v: float) -> str:
    if v < 2:  return "Low (0–2)"
    if v < 5:  return "Moderate (3–4)"
    if v < 7:  return "High (5–6)"
    if v < 9:  return "Very High (7–8)"
    return "Extreme (9–10)"

def band_edges(): return [0,2,5,7,9,10]
def band_labels(): return ["Low","Moderate","High","Very High","Extreme"]

def safe_read(p):
    if not os.path.exists(p):
        raise SystemExit(f"Missing file: {p}")
    return pd.read_csv(p)

def main(latest_csv, bytrap_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    latest = safe_read(latest_csv)
    bytrap = safe_read(bytrap_csv)

    # Ensure dates are parsed
    for df in (latest, bytrap):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 1) Distribution (latest only)
    v = latest["itch_index"].dropna().clip(0,10).values
    plt.figure(figsize=(8,5))
    plt.hist(v, bins=20)
    for x in band_edges():
        plt.axvline(x, linestyle="--", linewidth=1)
    plt.title("Itch Index distribution (latest per site)")
    plt.xlabel("Itch Index (0–10)")
    plt.ylabel("Number of sites")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"fig_hist_latest.png"), dpi=150)
    plt.close()

    # 2) Rolling trend (14-day)
    df = bytrap.dropna(subset=["date"]).copy()
    df["date_only"] = df["date"].dt.date
    daily = df.groupby("date_only")["itch_index"].agg(["mean", lambda s: s.quantile(0.95), "max"]).rename(columns={"<lambda_0>":"p95"})
    daily = daily.sort_index()
    roll = daily.rolling(14, min_periods=3).mean()

    plt.figure(figsize=(10,5))
    plt.plot(roll.index, roll["mean"], label="Mean (14-day)")
    plt.plot(roll.index, roll["p95"], label="95th (14-day)")
    plt.plot(roll.index, roll["max"], label="Max (14-day)")
    plt.ylabel("Itch Index (0–10)")
    plt.title("Hampton Roads Itch Index – Rolling 14-day trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"fig_trend_rolling.png"), dpi=150)
    plt.close()
    # 2b) Recent trend (last 30 days, 7-day rolling, time-based)
    df_recent = df.dropna(subset=["date"]).copy()
    df_recent["day"] = df_recent["date"].dt.floor("D")
    last_day = df_recent["day"].max()
    lookback_days = 30
    start_day = last_day - pd.Timedelta(days=lookback_days)

    df_recent = df_recent[df_recent["day"] >= start_day]

    daily_recent = (df_recent
        .groupby("day")["itch_index"]
        .agg(mean="mean", p95=lambda s: s.quantile(0.95), max="max")
        .sort_index())

    full_range = pd.date_range(start=start_day, end=last_day, freq="D")
    daily_recent = daily_recent.reindex(full_range)

    roll7 = daily_recent.rolling("7D", min_periods=2).mean()

    plt.figure(figsize=(8,4))
    plt.plot(roll7.index, roll7["mean"], label="Mean (7-day)")
    plt.plot(roll7.index, roll7["p95"],  label="95th (7-day)")
    plt.plot(roll7.index, roll7["max"],  label="Max (7-day)")
    ax = plt.gca()
    # Show ticks every 3 days, readable short format (e.g., "Sep 10")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # tweaks for readability
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")
    ax.margins(x=0.01)
    ax.grid(True, alpha=0.3)
    plt.ylabel("Itch Index (0–10)")
    plt.title("Hampton Roads Itch Index — Recent Trend (last 30 days, 7-day rolling)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig_trend_recent_7day.png"), dpi=150)
    plt.close()

    # 3) Species contribution
    # bite-weighted recent contribution = species_weight * recency_weight * abundance_score
    # (averaged over the last 30 days for smoother story)
    cutoff = df["date"].max() - pd.Timedelta(days=30)
    recent = df[df["date"]>=cutoff].copy()
    if "species_weight" in recent.columns and "recency_weight" in recent.columns and "abundance_score" in recent.columns:
        recent["contrib"] = recent["species_weight"] * recent["recency_weight"] * recent["abundance_score"]
    else:
        # fallback if someone runs this against earlier outputs
        recent["contrib"] = recent["itch_index"] / 10.0

    top = (recent.groupby("species")["contrib"].mean()
                 .sort_values(ascending=False).head(10))

    plt.figure(figsize=(9,5))
    plt.barh(top.index[::-1], top.values[::-1])
    plt.xlabel("Relative contribution (unitless)")
    plt.title("What’s driving the itch? (last 30 days)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"fig_species_contrib.png"), dpi=150)
    plt.close()

    # 4) Hotspot stability: latest vs last 4 weeks mean at same site
    # Use rounded lat/lon as site key (same as pipeline)
    df["lat_r"] = df["lat"].round(5)
    df["lon_r"] = df["lon"].round(5)
    recent4 = df[df["date"] >= (df["date"].max() - pd.Timedelta(days=28))].copy()
    site_mean = (recent4.groupby(["lat_r","lon_r"])["itch_index"]
                        .mean().rename("last4w_mean"))
    latest_keyed = latest.copy()
    latest_keyed["lat_r"] = latest_keyed["lat"].round(5)
    latest_keyed["lon_r"] = latest_keyed["lon"].round(5)
    merged = latest_keyed.merge(site_mean, on=["lat_r","lon_r"], how="left")
    merged["last4w_mean"] = merged["last4w_mean"].fillna(0)

    # Top 12 by latest
    top12 = merged.sort_values("itch_index", ascending=False).head(12)

    plt.figure(figsize=(10,6))
    # two bars per site: last 4w mean vs latest
    y = (top12["address"].fillna("").str.slice(0,35) + "…").where(top12["address"].str.len()>36, top12["address"])
    y = y[::-1]
    idx = top12.index[::-1]
    plt.barh(np.arange(len(idx))-0.15, top12.loc[idx,"last4w_mean"], height=0.3, label="Last 4 weeks mean")
    plt.barh(np.arange(len(idx))+0.15, top12.loc[idx,"itch_index"], height=0.3, label="Latest")
    plt.yticks(np.arange(len(idx)), y)
    plt.xlabel("Itch Index (0–10)")
    plt.title("Are hotspots stable? (latest vs last 4 weeks)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"fig_hotspot_stability.png"), dpi=150)
    plt.close()

    # 5) Trap type check (latest)
    if "trap_type" in bytrap.columns:
        # pick latest per site from bytrap to ensure fair comparison
        bytrap["lat_r"] = bytrap["lat"].round(5)
        bytrap["lon_r"] = bytrap["lon"].round(5)
        latest_by_site = (bytrap.sort_values("date")
                                .groupby(["lat_r","lon_r"], as_index=False)
                                .tail(1))
        plt.figure(figsize=(7,5))
        data = [g["itch_index"].clip(0,10).values for _, g in latest_by_site.groupby("trap_type")]
        labels = [str(k) for k, _ in latest_by_site.groupby("trap_type")]
        plt.boxplot(data, labels=labels, vert=True, showfliers=False)
        plt.ylabel("Itch Index (0–10)")
        plt.title("Itch Index by trap type (latest per site)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,"fig_traptype_box.png"), dpi=150)
        plt.close()

    # ------------------------------
    # Park-level Itch Index overlay
    # ------------------------------
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        print("Calculating park-level Itch Index (nearest-3 traps)...")

        # load parks (Norfolk)
        parks_path = os.path.join("data", "Parks_-_City_of_Norfolk.geojson")
        parks = gpd.read_file(parks_path)
        # ensure we operate in metric units for distance calculations
        parks = parks.to_crs(epsg=3857)

        # prepare trap GeoDataFrame (latest trap points)
        traps = latest.dropna(subset=["lat", "lon", "itch_index"]).copy()
        traps_gdf = gpd.GeoDataFrame(
            traps,
            geometry=gpd.points_from_xy(traps["lon"], traps["lat"]),
            crs="EPSG:4326"
        )
        traps_gdf = traps_gdf.to_crs(epsg=3857)

        # park centroids in meters
        park_centroids = parks.geometry.centroid
        park_coords = np.vstack([[pt.x, pt.y] for pt in park_centroids])

        # trap coords in meters
        trap_coords = np.vstack([[pt.x, pt.y] for pt in traps_gdf.geometry])
        trap_vals = traps_gdf["itch_index"].values
        trap_addresses = traps_gdf["address"].fillna("").values

        # nearest-3 by Euclidean distance in projected meters
        park_itch_means = []
        park_itch_n = []
        park_trap_lists = []
        # nearest-k weighted average by inverse-distance (power)
        k = 5
        power = 2.25   # exponent: 1.0 -> 1/d, 2.0 -> 1/d^2 (stronger locality)
        eps = 1e-3    # meters; avoids div-by-zero for coincident points

        park_itch_means = []
        park_itch_n = []
        park_trap_lists = []

        # Precompute trap arrays
        if len(trap_coords) == 0:
            # no traps at all: fill with NaNs / zeros
            for _ in range(len(park_coords)):
                park_itch_means.append(np.nan)
                park_itch_n.append(0)
                park_trap_lists.append([])
        else:
            for pc in park_coords:
                # squared Euclidean distances (in projected meters)
                d2 = np.sum((trap_coords - pc) ** 2, axis=1)
                d = np.sqrt(d2)

                # indices of nearest k traps (or fewer if not enough)
                idx = np.argsort(d)[:k]
                sel_vals = trap_vals[idx]            # itch_index values (may contain NaN)
                sel_addrs = [str(trap_addresses[i]) for i in idx]
                sel_d = d[idx]

                # handle case where all selected values are NaN
                valid_mask = ~np.isnan(sel_vals)
                n_valid = int(np.sum(valid_mask))
                park_itch_n.append(n_valid)

                if n_valid == 0:
                    park_itch_means.append(np.nan)
                    park_trap_lists.append([])
                    continue

                # compute inverse-distance weights; use eps to prevent inf for zero distance
                # w_i = 1 / (d_i + eps)**power
                w = 1.0 / (np.maximum(sel_d, 0.0) + eps) ** power
                # zero out weights where value is NaN, then renormalize
                w = w * valid_mask.astype(float)
                w_sum = w.sum()
                if w_sum <= 0:
                    # fallback to simple (unweighted) mean of valid values
                    mean_val = float(np.nanmean(sel_vals))
                else:
                    w = w / w_sum
                    mean_val = float(np.nansum(w * sel_vals))

                park_itch_means.append(mean_val)
                # store addresses of contributing traps (only those with valid values)
                contrib_addrs = [sel_addrs[i] for i in range(len(idx)) if valid_mask[i]]
                park_trap_lists.append(contrib_addrs)

        parks["itch_mean"] = park_itch_means
        parks["itch_n"] = park_itch_n
        parks["itch_trap_ids"] = [",".join(l) if l else "" for l in park_trap_lists]

        # save parks with new properties back to GeoJSON (in WGS84)
        parks_out = os.path.join("data", "parks_itch_index.geojson")
        parks.to_crs(epsg=4326).to_file(parks_out, driver="GeoJSON")
        print(f"Saved park-level scores to {parks_out}")

    except Exception as e:
        print("Park overlay skipped (geopandas required):", e)

    # ------------------------------
    # Create the folium map with 3 toggleable layers:
    #  - Traps (circle markers)
    #  - Parks (solid fills, black outline)
    #  - IDW grid (gradient as many small circles)
    # ------------------------------
    try:
        import folium
        from folium import features

        def band_color(v):
            v = float(np.clip(v, 0, 10))
            if v < 2:  return "#6BBE45"  # Low
            if v < 5:  return "#FFD23F"  # Moderate
            if v < 7:  return "#FF9F1C"  # High
            if v < 9:  return "#F6511D"  # Very High
            return "#A30000"             # Extreme

        mcenter = [latest["lat"].mean(), latest["lon"].mean()]
        m = folium.Map(location=mcenter, zoom_start=11, tiles="cartodbpositron")

        # -- Traps layer (points)
        traps_fg = folium.FeatureGroup(name="Traps (points)", show=True)
        for _, r in latest.iterrows():
            val = float(np.clip(r["itch_index"], 0, 10)) if not pd.isna(r["itch_index"]) else None
            c = band_color(val)
            popup_html = (f"<b>Itch:</b> {val:.1f} ({itch_band(val)})<br>"
                          f"<b>Address:</b> {r.get('address','') or '—'}<br>"
                          f"<b>Species:</b> {r.get('species','') or '—'}<br>"
                          f"<b>Date:</b> {r.get('date','') or '—'}<br>"
                          f"<b>Count:</b> {r.get('count','') or '—'}")
            popup = folium.Popup(popup_html, max_width=260)
            folium.CircleMarker(
                [r["lat"], r["lon"]],
                radius=6,
                color=c, fill=True, fill_color=c, fill_opacity=0.85,
                weight=1, popup=popup,
                tooltip=f"{val:.1f} – {itch_band(val)}" if val is not None else "No data"
            ).add_to(traps_fg)
        traps_fg.add_to(m)

        # -- Parks layer (solid color + black outline)
        parks_fg = folium.FeatureGroup(name="Parks (nearest-5 avg)", show=False)
        parks_geo = os.path.join("data", "parks_itch_index.geojson")
        if os.path.exists(parks_geo):
            import json
            with open(parks_geo, "r", encoding="utf-8") as f:
                parks_data = json.load(f)

            def park_style(feature):
                props = feature.get("properties", {})
                v = props.get("itch_mean", None)
                # color fill by band, outline in black
                fill = band_color(v)
                return {"color": "#000000", "weight": 1.25, "fillColor": fill, "fillOpacity": 0.75}

            def park_popup(feature):
                p = feature.get("properties", {})
                name = p.get("PARK_NAME")
                mean = p.get("itch_mean")
                n = p.get("itch_n", 0)
                traplist = p.get("itch_trap_ids", "")
                html = f"<b>{name}</b><br><b>Mean Itch:</b> {mean if mean is not None else '—'}<br><b>Contributing traps:</b> {n}<br><b>Trap addresses:</b> {traplist}"
                return html

            folium.GeoJson(
                parks_data,
                name="Parks (nearest-5 avg)",
                style_function=park_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=["PARK_NAME","itch_mean","itch_n"],
                    aliases=["Park:","Mean Itch:","# contributing traps:"],
                    localize=True,
                    labels=True,
                    sticky=False
                ),
                popup=folium.GeoJsonPopup(fields=[], labels=False)
            ).add_to(parks_fg)
        else:
            print(f"No park GeoJSON found at {parks_geo}")
        parks_fg.add_to(m)

        # -- Gradient / IDW grid layer
        gradient_fg = folium.FeatureGroup(name="IDW grid (city gradient)", show=False)
        heatgrid_path = os.path.join(os.path.dirname(bytrap_csv), "itch_index_heatgrid.csv")
        if os.path.exists(heatgrid_path):
            hg = pd.read_csv(heatgrid_path).dropna(subset=["lat","lon","itch_index_idw"])
            # small circles at grid points to give a gradient effect
            for _, row in hg.iterrows():
                v = float(row["itch_index_idw"])
                c = band_color(v)
                # radius tuned to ~200-500m (depends on zoom); small value for many points
                folium.CircleMarker(
                    [row["lat"], row["lon"]],
                    radius=6,
                    color=None, fill=True, fill_color=c, fill_opacity=0.65,
                    weight=0, popup=None
                ).add_to(gradient_fg)
        else:
            print(f"No heatgrid found at {heatgrid_path}; city gradient layer skipped.")
        gradient_fg.add_to(m)

        # Add controls + legend
        folium.LayerControl(collapsed=False).add_to(m)

        legend = """
        <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: white; padding: 10px 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 13px;">
          <div style="font-weight:600; margin-bottom:6px;">Itch Index (0–10)</div>
          <div><span style="background:#6BBE45; width:12px; height:12px; display:inline-block; margin-right:6px;"></span> Low (0–2)</div>
          <div><span style="background:#FFD23F; width:12px; height:12px; display:inline-block; margin-right:6px;"></span> Moderate (3–4)</div>
          <div><span style="background:#FF9F1C; width:12px; height:12px; display:inline-block; margin-right:6px;"></span> High (5–6)</div>
          <div><span style="background:#F6511D; width:12px; height:12px; display:inline-block; margin-right:6px;"></span> Very High (7–8)</div>
          <div><span style="background:#A30000; width:12px; height:12px; display:inline-block; margin-right:6px;"></span> Extreme (9–10)</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend))

        out_html = os.path.join(outdir, "itch_map_banded.html")
        m.save(out_html)
        print(f"Saved {out_html}")

    except Exception as e:
        print("Map skipped (install folium to enable):", e)

    # Summary for narration
    latest_bands = pd.cut(latest["itch_index"].clip(0,10), bins=band_edges(), labels=band_labels(), include_lowest=True)
    share = latest_bands.value_counts(normalize=True).reindex(band_labels(), fill_value=0)
    keylines = {
        "sites": int(latest.shape[0]),
        "band_share_%": (share*100).round(1).to_dict(),
        "dates_covered": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "top_species_30d": top.index.tolist(),
    }
    print(pd.Series(keylines))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", required=True, help="path to itch_index_latest_by_location.csv")
    ap.add_argument("--bytrap", required=True, help="path to itch_index_by_trap.csv")
    ap.add_argument("--outdir", default="outputs_story", help="where to save visuals")
    args = ap.parse_args()
    main(args.latest, args.bytrap, args.outdir)

