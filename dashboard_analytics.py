"""
Dashboard for retail analytics metrics visualization
Supports single CSV (seconds_and_counts.csv) and multi-feed CSVs (data/footfall/*.csv).
Expected columns: timestamp, entered, exited, occupancy [, feed_id]
"""
from __future__ import annotations

import os
import glob
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Try to use streamlit_autorefresh if available
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ---------- Page config ----------
st.set_page_config(
    page_title="CraftEye - Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styles ----------
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
:root{
  --bg:#0b1220; --bg2:#0c1426; --text:#f7faff; --muted:#dbe6ff;
  --panel:rgba(255,255,255,.05); --stroke:rgba(255,255,255,.14);
  --ring:#2d8bff; --ring2:#3aa6ff; --glow:rgba(58,166,255,.38); --ok:#2bd38a;
}
.stApp{background:linear-gradient(180deg,var(--bg) 0%, var(--bg2) 100%); color:var(--text); font-size:16.5px;}
.container{max-width:1180px; margin:0 auto; padding:24px 16px;}
.chart-panel{background:var(--panel); border:1px solid var(--stroke); border-radius:18px; padding:20px; margin-bottom:24px;}
.chart-panel:hover{box-shadow:0 10px 26px var(--glow); border-color:var(--ring); transition:box-shadow .2s ease, border-color .2s ease;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="container">', unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<h1 class="title">Analytics Dashboard</h1>
<p class="lead">Live footfall analytics from monitoring logs.</p>
""", unsafe_allow_html=True)

# --- Available use-cases (extendable) ---
available_cases = ["Footfall"]
sel = st.multiselect("Select use-cases to enable", available_cases, default=available_cases)

# Show enabled cases as buttons; user clicks a button to open the specific view
st.write("### Enabled use-cases")
cols = st.columns(len(sel) if sel else 1)
for i, case in enumerate(sel):
    with cols[i]:
        if st.button(f"Open {case}", key=f"open_case_{case}"):
            st.session_state.setdefault("dashboard_active_case", case)

active = st.session_state.get("dashboard_active_case")


# ---------- Helpers ----------
def _parse_timestamp_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().mean() > 0.5 and s.dtype == object:
        today = pd.Timestamp.now().normalize()
        try:
            hhmmss = pd.to_datetime(s, format="%H:%M:%S", errors="coerce").dt.time
            out = pd.to_datetime(hhmmss.astype(str), errors="coerce")
            out = out.map(lambda t: pd.Timestamp.combine(today, t) if pd.notnull(t) else pd.NaT)
        except Exception:
            pass
    return out


def _normalize_period(period_str: str) -> str:
    if not period_str:
        return "1min"
    period_str = str(period_str)
    repl = {"1H": "1h", "H": "h", "1T": "1min", "T": "min", "1S": "1s", "S": "s", "D": "d"}
    return repl.get(period_str, period_str.lower())


def _list_footfall_paths() -> List[str]:
    paths: List[str] = []
    base_single = "seconds_and_counts.csv"
    if os.path.exists(base_single):
        paths.append(base_single)
    paths.extend(glob.glob(os.path.join("data", "footfall", "*.csv")))
    return paths


def _paths_signature(paths: List[str]) -> Tuple[Tuple[str, float], ...]:
    """Signature used to invalidate cache when any file changes."""
    sig = []
    for p in paths:
        try:
            sig.append((p, os.path.getmtime(p)))
        except OSError:
            # file might be deleted between listing and stat
            continue
    return tuple(sorted(sig))


@st.cache_data(show_spinner=False)
def _load_and_concat(paths_sig: Tuple[Tuple[str, float], ...]) -> pd.DataFrame:
    """Read, normalize, and concatenate all CSVs. Cache keyed by (path, mtime)."""
    if not paths_sig:
        return pd.DataFrame()

    dfs = []
    for p, _mtime in paths_sig:
        try:
            d = pd.read_csv(p)
            cols_lower = {c.strip().lower(): c for c in d.columns}
            if "timestamp" in cols_lower:
                d = d.rename(columns={cols_lower["timestamp"]: "timestamp"})
            if "entered" not in d.columns:
                d["entered"] = 0
            if "exited" not in d.columns:
                d["exited"] = 0
            if "occupancy" not in d.columns:
                d["occupancy"] = 0

            d["timestamp"] = _parse_timestamp_series(d["timestamp"])
            d = d.dropna(subset=["timestamp"])

            if "feed_id" not in d.columns:
                inferred = os.path.splitext(os.path.basename(p))[0]
                d["feed_id"] = inferred

            dfs.append(d)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df["entered"] = pd.to_numeric(df["entered"], errors="coerce").fillna(0).astype(int)
    df["exited"] = pd.to_numeric(df["exited"], errors="coerce").fillna(0).astype(int)
    df["occupancy"] = pd.to_numeric(df["occupancy"], errors="coerce").fillna(0).astype(int)
    return df.sort_values("timestamp")


def load_footfall_data_simple() -> pd.DataFrame:
    """Public loader that lists files, builds a signature, and uses the cache."""
    paths = _list_footfall_paths()
    sig = _paths_signature(paths)
    return _load_and_concat(sig)


# ---------- Trend & Zone Analytics ----------
def add_trend_analysis(df: pd.DataFrame):
    if df.empty:
        return
    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.subheader("Footfall Trend Analysis")

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_name"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = df["timestamp"].dt.dayofweek.isin([5, 6])

    daily_stats = df.groupby('day_name').agg({'entered': 'sum', 'exited': 'sum', 'occupancy': 'mean'}).round(2)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_stats = daily_stats.reindex(days_order)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily_stats.index, y=daily_stats['entered'], name='Entries'))
    fig.add_trace(go.Bar(x=daily_stats.index, y=daily_stats['exited'], name='Exits'))
    fig.update_layout(
        title="Daily Footfall Patterns",
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f7faff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        weekday_hourly = df[~df['is_weekend']].groupby('hour').agg({'entered': 'mean', 'exited': 'mean'}).round(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekday_hourly.index, y=weekday_hourly['entered'], name='Avg. Entries'))
        fig.add_trace(go.Scatter(x=weekday_hourly.index, y=weekday_hourly['exited'], name='Avg. Exits'))
        fig.update_layout(
            title="Weekday Hourly Pattern",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f7faff'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True, title="Hour of Day"),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True, title="Average Count"),
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        weekend_hourly = df[df['is_weekend']].groupby('hour').agg({'entered': 'mean', 'exited': 'mean'}).round(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekend_hourly.index, y=weekend_hourly['entered'], name='Avg. Entries'))
        fig.add_trace(go.Scatter(x=weekend_hourly.index, y=weekend_hourly['exited'], name='Avg. Exits'))
        fig.update_layout(
            title="Weekend Hourly Pattern",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f7faff'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True, title="Hour of Day"),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True, title="Average Count"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Insights")
    if not df.empty:
        peak_hours = df.groupby('hour')['entered'].mean().sort_values(ascending=False).head(3)
        peak_hours_str = ", ".join([f"{int(h)}:00" for h in peak_hours.index])
        busiest_day = (daily_stats['entered'].idxmax() if not daily_stats.empty else "N/A")
        weekday_avg = df[~df['is_weekend']]['entered'].mean() if (~df['is_weekend']).any() else 0
        weekend_avg = df[df['is_weekend']]['entered'].mean() if (df['is_weekend']).any() else 0
        day_comparison = "higher" if weekend_avg > weekday_avg else "lower or similar"
        st.markdown(f"- Peak entry times: {peak_hours_str}\n- Busiest day: {busiest_day}\n- Weekend traffic is {day_comparison} than weekdays")

    st.markdown('</div>', unsafe_allow_html=True)


def add_zone_analytics(df: pd.DataFrame):
    if df.empty or 'feed_id' not in df.columns:
        return
    zones = df.groupby('feed_id').agg(
        total_entered=('entered', 'sum'),
        total_exited=('exited', 'sum'),
        peak_occupancy=('occupancy', 'max')
    ).reset_index()

    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.subheader("Zone Analytics (per feed)")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=zones['feed_id'], y=zones['total_entered'], name='Total Entries'))
    fig.add_trace(go.Bar(x=zones['feed_id'], y=zones['total_exited'], name='Total Exits'))
    fig.update_layout(
        title="Total Entries/Exits by Feed",
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f7faff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True, title="Feed"),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True, title="Count"),
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=zones['feed_id'], y=zones['peak_occupancy'], name='Peak Occupancy'))
    fig2.update_layout(
        title="Peak Occupancy by Feed",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f7faff'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True, title="Feed"),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', showgrid=True, title="Occupancy"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ---------- Main Footfall Panel ----------
def show_footfall_panel(df: pd.DataFrame):
    st.header("Footfall Summary")

    # --- Auto-refresh controls
    ar_col, btn_col, ts_col = st.columns([2, 1, 1])
    with ar_col:
        st.session_state.setdefault("auto_refresh_enabled", False)
        st.session_state.setdefault("auto_refresh_secs", 10)
        auto_on = st.checkbox("Auto-refresh", value=st.session_state["auto_refresh_enabled"])
        interval = st.slider("Interval (sec)", 3, 60, st.session_state["auto_refresh_secs"])
        st.session_state["auto_refresh_enabled"] = auto_on
        st.session_state["auto_refresh_secs"] = interval

        if auto_on:
            # Prefer st_autorefresh if available
            if st_autorefresh:
                st_autorefresh(interval=interval * 1000, key="auto_refresh_counter")
            else:
                # Fallback: meta refresh
                st.markdown(f"<meta http-equiv='refresh' content='{interval}'>", unsafe_allow_html=True)

    with btn_col:
        if st.button("🔄 Refresh data"):
            st.rerun()

    with ts_col:
        if not df.empty:
            last_ts = pd.to_datetime(df["timestamp"]).max()
            st.caption(f"Last updated: {last_ts.strftime('%Y-%m-%d %H:%M:%S')}")

    if df.empty:
        st.info("No footfall data available yet. Start monitoring with footfall enabled to collect data.")
        return

    # Feed filter
    feeds = sorted(df["feed_id"].astype(str).unique().tolist()) if "feed_id" in df.columns else []
    if feeds:
        feed_sel = st.multiselect("Filter by feed", feeds, default=feeds)
        df = df[df["feed_id"].isin(feed_sel)]
        if df.empty:
            st.info("No data for the selected feed(s).")
            return

    # KPIs
    total_entries = int(df["entered"].sum())
    total_exits = int(df["exited"].sum())
    current_occ = int(df["occupancy"].iloc[-1]) if not df.empty else 0
    peak_occ = int(df["occupancy"].max())

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Entries", total_entries)
    with c2: st.metric("Total Exits", total_exits)
    with c3: st.metric("Current Occupancy", current_occ)
    with c4: st.metric("Peak Occupancy",  peak_occ)

    # Time range
    time_range = st.selectbox("Time Range", ["Last Hour","Last 6 Hours","Last 24 Hours","All Data"], index=3)
    now = pd.Timestamp.now()
    if time_range == "Last Hour":
        start = now - pd.Timedelta(hours=1); resample_period = "1min"
    elif time_range == "Last 6 Hours":
        start = now - pd.Timedelta(hours=6); resample_period = "1min"
    elif time_range == "Last 24 Hours":
        start = now - pd.Timedelta(hours=24); resample_period = "1h"
    else:
        start = df["timestamp"].min();       resample_period = "1h"

    dff = df[df["timestamp"] >= start]
    if dff.empty:
        st.info("No data in selected range")
        return

    # Aggregate (lowercase aliases to avoid FutureWarning)
    period = _normalize_period(resample_period)
    agg = (
        dff.set_index("timestamp")
           .resample(period)
           .agg({"entered":"sum","exited":"sum","occupancy":"last"})
           .fillna(0)
    )

    st.subheader("Entered / Exited over time")
    st.line_chart(agg[["entered", "exited"]])

    st.subheader("Occupancy over time")
    st.line_chart(agg[["occupancy"]])

    # Advanced sections
    add_trend_analysis(dff)
    add_zone_analytics(dff)

    # Download current filtered range
    csv_name = f"footfall_{time_range.replace(' ', '_').lower()}.csv"
    st.download_button(
        "⬇️ Download this view as CSV",
        data=dff.to_csv(index=False).encode("utf-8"),
        file_name=csv_name,
        mime="text/csv",
    )


# --- main control: active case ---
if active == "Footfall":
    df = load_footfall_data_simple()
    show_footfall_panel(df)
else:
    st.info("Pick a use-case above and click its button to open a focused view.")

st.markdown('</div>', unsafe_allow_html=True)

