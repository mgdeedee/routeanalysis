#!/usr/bin/env python3
"""
Route Analysis Dashboard Generator

Reads a CSV file with appointment data (id, date, staff_id, address, duration, revenue),
geocodes addresses, calculates driving distances between sequential appointments per staff
per day, and generates an interactive HTML dashboard.

Usage:
    python3 route_analysis.py <csv_file>
    python3 route_analysis.py 116676-company-Jan.csv
"""

import sys
import os
import json
import math
import time
import csv
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from geopy.geocoders import ArcGIS

# ─── Constants ────────────────────────────────────────────────────────────────

GEOCODE_CACHE_FILE = "geocode_cache.json"
ROAD_FACTOR = 1.3          # Multiplier to estimate road distance from straight-line
AVG_DRIVING_SPEED_MPH = 30  # Average urban driving speed for time estimation
KM_TO_MILES = 0.621371

# ─── Geocoding ────────────────────────────────────────────────────────────────

def load_geocode_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}


def save_geocode_cache(cache, cache_file):
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def normalize_address(address):
    """Normalize address for consistent cache keys."""
    return " ".join(address.strip().split())


def geocode_addresses(df, cache_file=GEOCODE_CACHE_FILE):
    """Geocode all unique addresses using ArcGIS (free, no API key, fast)."""
    cache = load_geocode_cache(cache_file)
    geolocator = ArcGIS(timeout=10)

    unique_addresses = df["address"].dropna().unique()
    total = len(unique_addresses)
    geocoded = 0
    failed = 0
    cached = 0

    print(f"Geocoding {total} unique addresses...")

    for i, addr in enumerate(unique_addresses):
        key = normalize_address(addr)
        if key in cache:
            cached += 1
            continue

        try:
            location = geolocator.geocode(addr)
            if location:
                cache[key] = {
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                }
                geocoded += 1
            else:
                cache[key] = None
                failed += 1
        except Exception as e:
            cache[key] = None
            failed += 1
            print(f"  Error geocoding '{addr[:50]}...': {e}")

        # Progress update every 100 addresses
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{total} ({cached} cached, {geocoded} new, {failed} failed)")
            save_geocode_cache(cache, cache_file)

        # Small delay to be respectful to the API
        time.sleep(0.05)

    save_geocode_cache(cache, cache_file)
    print(f"Geocoding complete: {cached} cached, {geocoded} newly geocoded, {failed} failed")

    # Apply coordinates to dataframe
    lats = []
    lons = []
    for addr in df["address"]:
        key = normalize_address(str(addr))
        coords = cache.get(key)
        if coords:
            lats.append(coords["latitude"])
            lons.append(coords["longitude"])
        else:
            lats.append(None)
            lons.append(None)

    df["latitude"] = lats
    df["longitude"] = lons

    valid = df["latitude"].notna().sum()
    print(f"Coordinates assigned: {valid}/{len(df)} rows ({valid/len(df)*100:.1f}%)")

    return df


# ─── Distance Calculations ───────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in kilometers."""
    R = 6371  # Earth's radius in km
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def calculate_daily_metrics(df):
    """Calculate driving distance, drive time, appointment metrics per staff per day."""
    # Drop rows without coordinates
    df_valid = df.dropna(subset=["latitude", "longitude"]).copy()
    df_valid["date"] = pd.to_datetime(df_valid["date"])

    daily_records = []

    for (staff_id, date), group in df_valid.groupby(["staff_id", "date"]):
        # Sort by appointment id (proxy for chronological order)
        group = group.sort_values("id")

        # Calculate sequential distances
        total_distance_km = 0
        leg_distances = []
        for i in range(len(group) - 1):
            row1, row2 = group.iloc[i], group.iloc[i + 1]
            dist = haversine_km(
                row1["latitude"], row1["longitude"],
                row2["latitude"], row2["longitude"],
            )
            road_dist = dist * ROAD_FACTOR
            leg_distances.append(road_dist)
            total_distance_km += road_dist

        total_distance_miles = total_distance_km * KM_TO_MILES
        est_drive_time_min = (total_distance_miles / AVG_DRIVING_SPEED_MPH) * 60 if total_distance_miles > 0 else 0
        total_appt_duration = group["duration"].sum()
        total_revenue = group["revenue"].sum()
        num_appointments = len(group)

        # Efficiency metrics
        revenue_per_mile = total_revenue / total_distance_miles if total_distance_miles > 0 else 0
        revenue_per_hour = total_revenue / (total_appt_duration / 60) if total_appt_duration > 0 else 0
        total_working_min = total_appt_duration + est_drive_time_min
        productive_pct = (total_appt_duration / total_working_min * 100) if total_working_min > 0 else 0

        daily_records.append({
            "staff_id": str(staff_id),
            "date": date,
            "num_appointments": num_appointments,
            "total_distance_miles": round(total_distance_miles, 2),
            "est_drive_time_min": round(est_drive_time_min, 1),
            "total_appt_duration_min": total_appt_duration,
            "total_revenue": round(total_revenue, 2),
            "total_working_min": round(total_working_min, 1),
            "revenue_per_mile": round(revenue_per_mile, 2),
            "revenue_per_hour": round(revenue_per_hour, 2),
            "productive_pct": round(productive_pct, 1),
        })

    return pd.DataFrame(daily_records)


# ─── Dashboard Generation ────────────────────────────────────────────────────

def generate_dashboard(metrics_df, csv_filename, output_file="dashboard.html"):
    """Generate an interactive HTML dashboard from daily metrics."""

    # Prepare data
    metrics_df = metrics_df.copy()
    metrics_df["date_str"] = metrics_df["date"].dt.strftime("%Y-%m-%d")
    metrics_df["day_of_week"] = metrics_df["date"].dt.day_name()

    # Sort by date
    metrics_df = metrics_df.sort_values(["date", "staff_id"])

    # ── Aggregate summaries ──
    staff_summary = metrics_df.groupby("staff_id").agg(
        total_miles=("total_distance_miles", "sum"),
        total_drive_min=("est_drive_time_min", "sum"),
        total_appt_min=("total_appt_duration_min", "sum"),
        total_revenue=("total_revenue", "sum"),
        total_appointments=("num_appointments", "sum"),
        days_worked=("date", "nunique"),
        avg_revenue_per_mile=("revenue_per_mile", "mean"),
        avg_productive_pct=("productive_pct", "mean"),
    ).reset_index()
    staff_summary = staff_summary.sort_values("total_revenue", ascending=False)

    daily_summary = metrics_df.groupby("date_str").agg(
        total_miles=("total_distance_miles", "sum"),
        total_drive_min=("est_drive_time_min", "sum"),
        total_appt_min=("total_appt_duration_min", "sum"),
        total_revenue=("total_revenue", "sum"),
        total_appointments=("num_appointments", "sum"),
        staff_count=("staff_id", "nunique"),
    ).reset_index()

    # ── KPI totals ──
    total_miles = metrics_df["total_distance_miles"].sum()
    total_drive_hrs = metrics_df["est_drive_time_min"].sum() / 60
    total_appt_hrs = metrics_df["total_appt_duration_min"].sum() / 60
    total_revenue = metrics_df["total_revenue"].sum()
    total_appointments = metrics_df["num_appointments"].sum()
    avg_rev_per_mile = total_revenue / total_miles if total_miles > 0 else 0

    # ── Build figures ──

    # 1. Daily Overview - Stacked bar by staff (mileage)
    fig_daily_miles = px.bar(
        metrics_df, x="date_str", y="total_distance_miles", color="staff_id",
        title="Daily Driving Distance by Staff (Miles)",
        labels={"date_str": "Date", "total_distance_miles": "Miles", "staff_id": "Staff ID"},
    )
    fig_daily_miles.update_layout(barmode="stack", xaxis_tickangle=-45, height=500)

    # 2. Daily Drive Time vs Appointment Duration
    fig_daily_time = go.Figure()
    fig_daily_time.add_trace(go.Bar(
        x=daily_summary["date_str"], y=daily_summary["total_drive_min"],
        name="Driving Time (min)", marker_color="#EF553B",
    ))
    fig_daily_time.add_trace(go.Bar(
        x=daily_summary["date_str"], y=daily_summary["total_appt_min"],
        name="Appointment Duration (min)", marker_color="#636EFA",
    ))
    fig_daily_time.update_layout(
        barmode="group", title="Daily Driving Time vs Appointment Duration (All Staff)",
        xaxis_title="Date", yaxis_title="Minutes", xaxis_tickangle=-45, height=500,
    )

    # 3. Daily Revenue
    fig_daily_revenue = go.Figure()
    fig_daily_revenue.add_trace(go.Bar(
        x=daily_summary["date_str"], y=daily_summary["total_revenue"],
        name="Revenue", marker_color="#00CC96",
    ))
    fig_daily_revenue.add_trace(go.Scatter(
        x=daily_summary["date_str"], y=daily_summary["total_revenue"],
        mode="lines+markers", name="Trend", line=dict(color="#AB63FA", width=2),
    ))
    fig_daily_revenue.update_layout(
        title="Daily Total Revenue",
        xaxis_title="Date", yaxis_title="Revenue ($)", xaxis_tickangle=-45, height=500,
    )

    # 4. Staff Summary - Mileage vs Revenue scatter
    fig_staff_scatter = px.scatter(
        staff_summary, x="total_miles", y="total_revenue",
        size="total_appointments", color="staff_id",
        hover_data=["days_worked", "avg_revenue_per_mile"],
        title="Staff Efficiency: Total Miles vs Total Revenue (size = # appointments)",
        labels={"total_miles": "Total Miles Driven", "total_revenue": "Total Revenue ($)", "staff_id": "Staff ID"},
    )
    fig_staff_scatter.update_layout(height=500)

    # 5. Staff Daily Breakdown - Heatmap of mileage
    pivot_miles = metrics_df.pivot_table(
        index="staff_id", columns="date_str", values="total_distance_miles", aggfunc="sum", fill_value=0
    )
    fig_heatmap_miles = px.imshow(
        pivot_miles, aspect="auto",
        title="Daily Mileage Heatmap by Staff",
        labels=dict(x="Date", y="Staff ID", color="Miles"),
        color_continuous_scale="YlOrRd",
    )
    fig_heatmap_miles.update_layout(height=600, xaxis_tickangle=-45)

    # 6. Staff Daily Revenue Heatmap
    pivot_revenue = metrics_df.pivot_table(
        index="staff_id", columns="date_str", values="total_revenue", aggfunc="sum", fill_value=0
    )
    fig_heatmap_revenue = px.imshow(
        pivot_revenue, aspect="auto",
        title="Daily Revenue Heatmap by Staff",
        labels=dict(x="Date", y="Staff ID", color="Revenue ($)"),
        color_continuous_scale="Greens",
    )
    fig_heatmap_revenue.update_layout(height=600, xaxis_tickangle=-45)

    # 7. Efficiency: Revenue per Mile by Staff
    fig_efficiency = px.bar(
        staff_summary.sort_values("avg_revenue_per_mile", ascending=True),
        x="avg_revenue_per_mile", y="staff_id", orientation="h",
        title="Average Revenue per Mile by Staff (Higher = More Efficient)",
        labels={"avg_revenue_per_mile": "Revenue per Mile ($)", "staff_id": "Staff ID"},
        color="avg_revenue_per_mile", color_continuous_scale="RdYlGn",
    )
    fig_efficiency.update_layout(height=600)

    # 8. Productive Time % by Staff
    fig_productive = px.bar(
        staff_summary.sort_values("avg_productive_pct", ascending=True),
        x="avg_productive_pct", y="staff_id", orientation="h",
        title="Average Productive Time % by Staff (Appointment Time / Total Working Time)",
        labels={"avg_productive_pct": "Productive %", "staff_id": "Staff ID"},
        color="avg_productive_pct", color_continuous_scale="RdYlGn",
    )
    fig_productive.update_layout(height=600)

    # 9. Per-Staff Daily Detail Table
    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=["Staff ID", "Date", "Appts", "Miles", "Drive (min)",
                     "Appt Duration (min)", "Revenue ($)", "Rev/Mile", "Productive %"],
            fill_color="#636EFA", font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[
                metrics_df["staff_id"],
                metrics_df["date_str"],
                metrics_df["num_appointments"],
                metrics_df["total_distance_miles"],
                metrics_df["est_drive_time_min"],
                metrics_df["total_appt_duration_min"],
                metrics_df["total_revenue"],
                metrics_df["revenue_per_mile"],
                metrics_df["productive_pct"],
            ],
            fill_color="lavender",
            align="left",
        ),
    )])
    fig_table.update_layout(title="Detailed Daily Metrics by Staff", height=800)

    # ── Assemble HTML ──
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Route Analysis Dashboard - {csv_filename}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f7fa;
        }}
        .header {{
            text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border-radius: 12px; margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .kpi-row {{
            display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap;
        }}
        .kpi-card {{
            flex: 1; min-width: 180px; background: white; border-radius: 12px;
            padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
        }}
        .kpi-value {{
            font-size: 32px; font-weight: 700; margin: 8px 0;
        }}
        .kpi-label {{
            font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 0.5px;
        }}
        .chart-container {{
            background: white; border-radius: 12px; padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 24px;
        }}
        .chart-row {{
            display: flex; gap: 24px; margin-bottom: 24px; flex-wrap: wrap;
        }}
        .chart-half {{ flex: 1; min-width: 400px; }}
        .note {{
            text-align: center; color: #999; font-size: 12px; margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Route Analysis Dashboard</h1>
        <p>Data: {csv_filename} | {len(metrics_df["staff_id"].unique())} staff | {len(metrics_df["date_str"].unique())} days | {total_appointments} appointments</p>
    </div>

    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Total Miles Driven</div>
            <div class="kpi-value" style="color:#EF553B">{total_miles:,.0f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Total Drive Time</div>
            <div class="kpi-value" style="color:#FF6692">{total_drive_hrs:,.1f} hrs</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Total Appointment Time</div>
            <div class="kpi-value" style="color:#636EFA">{total_appt_hrs:,.1f} hrs</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value" style="color:#00CC96">${total_revenue:,.0f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Total Appointments</div>
            <div class="kpi-value" style="color:#AB63FA">{total_appointments:,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Revenue/Mile</div>
            <div class="kpi-value" style="color:#FFA15A">${avg_rev_per_mile:,.2f}</div>
        </div>
    </div>
""")

    # Add each chart
    charts = [
        ("daily_miles", fig_daily_miles, False),
        ("daily_time", fig_daily_time, False),
        ("daily_revenue", fig_daily_revenue, False),
        ("staff_scatter", fig_staff_scatter, False),
        ("heatmap_miles", fig_heatmap_miles, False),
        ("heatmap_revenue", fig_heatmap_revenue, False),
        ("efficiency", fig_efficiency, False),
        ("productive", fig_productive, False),
        ("detail_table", fig_table, False),
    ]

    for chart_id, fig, _ in charts:
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=60, r=30, t=60, b=60),
        )
        chart_json = fig.to_json()
        html_parts.append(f"""
    <div class="chart-container">
        <div id="{chart_id}" style="width:100%;"></div>
        <script>
            var data = {chart_json};
            Plotly.newPlot('{chart_id}', data.data, data.layout, {{responsive: true}});
        </script>
    </div>
""")

    html_parts.append("""
    <div class="note">
        Distances are estimated using haversine formula with 1.3x road factor.
        Drive times estimated at 30 mph average urban speed.
        Appointment order is based on appointment ID (chronological proxy).
    </div>
</body>
</html>
""")

    html_content = "".join(html_parts)
    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"Dashboard saved to: {output_file}")
    return output_file


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 route_analysis.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    csv_filename = os.path.basename(csv_file)
    output_file = csv_filename.replace(".csv", "_dashboard.html")
    metrics_csv = csv_filename.replace(".csv", "_metrics.csv")

    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows, {df['staff_id'].nunique()} staff, {df['date'].nunique()} dates")

    # Geocode addresses
    df = geocode_addresses(df)

    # Calculate daily metrics
    print("\nCalculating daily route metrics...")
    metrics_df = calculate_daily_metrics(df)
    print(f"Generated {len(metrics_df)} staff-day records")

    # Save metrics to CSV
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Metrics saved to: {metrics_csv}")

    # Generate dashboard
    print("\nGenerating dashboard...")
    generate_dashboard(metrics_df, csv_filename, output_file)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total miles driven:       {metrics_df['total_distance_miles'].sum():>10,.1f}")
    print(f"Total drive time (hrs):   {metrics_df['est_drive_time_min'].sum()/60:>10,.1f}")
    print(f"Total appt time (hrs):    {metrics_df['total_appt_duration_min'].sum()/60:>10,.1f}")
    print(f"Total revenue:            ${metrics_df['total_revenue'].sum():>9,.0f}")
    print(f"Avg revenue/mile:         ${metrics_df['total_revenue'].sum()/metrics_df['total_distance_miles'].sum():>9,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
