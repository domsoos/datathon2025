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

    # 6) Interactive map with banded colors (discrete)
    try:
        import folium
        from folium import features

        def band_color(v):
            if v < 2:  return "#6BBE45"  # Low
            if v < 5:  return "#FFD23F"  # Moderate
            if v < 7:  return "#FF9F1C"  # High
            if v < 9:  return "#F6511D"  # Very High
            return "#A30000"             # Extreme

        mcenter = [latest["lat"].mean(), latest["lon"].mean()]
        m = folium.Map(location=mcenter, zoom_start=11, tiles="cartodbpositron")
        g = folium.FeatureGroup(name="Itch (latest)", show=True)

        for _, r in latest.iterrows():
            val = float(np.clip(r["itch_index"], 0, 10))
            c = band_color(val)
            popup = folium.Popup(f"<b>Itch:</b> {val:.1f} ({itch_band(val)})<br>"
                                 f"<b>Address:</b> {r.get('address','') or '—'}<br>"
                                 f"<b>Species:</b> {r.get('species','') or '—'}<br>"
                                 f"<b>Date:</b> {r.get('date','') or '—'}<br>"
                                 f"<b>Count:</b> {r.get('count','') or '—'}", max_width=260)
            folium.CircleMarker(
                [r["lat"], r["lon"]],
                radius=6,
                color=c, fill=True, fill_color=c, fill_opacity=0.85,
                weight=1, popup=popup,
                tooltip=f"{val:.1f} – {itch_band(val)}"
            ).add_to(g)
        g.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

        # Legend
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

