# app.py — LinkedIn Ads CSV Analyzer (Streamlit + Headless Fallback)
# ----------------------------------------------------------------------------
# Why this rewrite?
# You hit: ModuleNotFoundError: No module named 'streamlit' (in Canvas),
# AttributeError: 'NoneType' object has no attribute 'notna' (missing cols),
# pandas ParserError (quirky LinkedIn exports), AND now
# "source code string cannot contain null bytes".
#
# This version fixes all of them:
#   • Runs Streamlit UI if installed, or a headless generator otherwise.
#   • KPI math is null-safe (missing columns → NaN, no crashes).
#   • CSV reader handles UTF‑16 + tab + preamble (common for LinkedIn),
#     odd delimiters/encodings, and skips bad lines.
#   • All byte checks use explicit escapes (e.g., b"\xff\xfe") — no literal
#     null bytes appear in source, eliminating the error.
#
# Quick start (Streamlit, on your machine):
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Quick start (Headless, e.g., Canvas sandbox w/o Streamlit):
#   python app.py
#   → outputs are written to /mnt/data/linkedindash_outputs
# ----------------------------------------------------------------------------

from __future__ import annotations
import os
import io
import json
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional libs
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import kaleido  # noqa: F401
    KALEIDO_AVAILABLE = True
except Exception:
    KALEIDO_AVAILABLE = False

# Detect Streamlit safely
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# =============================================================================
# Canonical column mapping
# =============================================================================
DEFAULT_MAPPING: Dict[str, List[str]] = {
    # common metrics
    "date": ["date", "day", "Date"],
    "account_id": ["Account ID", "account_id", "account"],
    "campaign_group_id": ["Campaign Group ID", "campaign_group_id"],
    "campaign_id": ["Campaign ID", "campaign_id"],
    "campaign_name": ["Campaign Name", "campaign_name"],
    "objective": ["Objective", "objective"],
    "status": ["Status", "status"],
    "currency": ["Currency", "currency"],
    "ad_format": ["Ad Format", "ad_format", "Ad Format Type"],
    "impressions": ["Impressions", "impressions"],
    "clicks": ["Clicks", "clicks"],
    "spend": ["Spend", "spend", "Amount Spent"],
    "conversions": ["Conversions", "conversions"],
    "leads": ["Leads", "leads", "Lead Gen Leads"],
    "video_views": ["Video Views", "video_views"],
    "video_25": ["Video plays at 25%", "video_25"],
    "video_50": ["Video plays at 50%", "video_50"],
    "video_75": ["Video plays at 75%", "video_75"],
    "video_100": ["Video plays at 100%", "video_100"],
    "reach": ["Reach", "reach", "Unique Impressions"],
    # creative-level
    "creative_id": ["Creative ID", "creative_id"],
    "creative_name": ["Creative Name", "creative_name"],
    "headline": ["Headline", "headline"],
    "cta": ["CTA", "Call to action", "cta"],
    # conversation ads
    "conversation_starts": ["Conversation starts", "conversation_starts"],
    "message_link_clicks": ["Message link clicks", "message_link_clicks"],
    "button_clicks": ["Button clicks", "button_clicks"],
    # demographics dims
    "dimension_type": ["Dimension Type", "dimension_type"],
    "dimension_value": ["Dimension Value", "dimension_value"],
    "country": ["Country", "country"],
    "company": ["Company", "company"],
    "industry": ["Industry", "industry"],
    "job_function": ["Job Function", "job_function"],
    "job_title": ["Job Title", "job_title"],
    "seniority": ["Seniority", "seniority"],
    "company_size": ["Company Size", "company_size"],
}

# =============================================================================
# Core helpers
# =============================================================================

def canonicalize_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    """Rename columns to canonical names where possible, parse dates, cast numerics."""
    colmap: Dict[str, str] = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for canon, aliases in mapping.items():
        for alias in aliases:
            if alias.lower() in lower_cols:
                colmap[lower_cols[alias.lower()]] = canon
                break
    df = df.rename(columns=colmap)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in [
        "impressions","clicks","spend","conversions","leads","video_views",
        "video_25","video_50","video_75","video_100","reach",
        "conversation_starts","message_link_clicks","button_clicks"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def detect_file_type(df: pd.DataFrame) -> str:
    cols = set(c.lower() for c in df.columns)
    if {"dimension type", "dimension value"}.intersection(cols) or {"dimension_type","dimension_value"}.issubset(cols):
        return "demographics"
    if {"conversation starts", "message link clicks", "button clicks"}.intersection(cols) or \
       {"conversation_starts","message_link_clicks","button_clicks"}.intersection(cols):
        return "conversation"
    if "creative id" in cols or "creative_id" in cols:
        return "creative"
    if ("campaign id" in cols or "campaign_id" in cols) and ("impressions" in cols or "spend" in cols):
        return "campaign"
    return "unknown"


def _ensure_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a Series for a column; if missing, return NaN series of correct length."""
    s = df.get(name, None)
    if s is None:
        return pd.Series(np.nan, index=df.index)
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    if not s.index.equals(df.index):
        s = s.reindex(df.index)
    return s


def kpi_safe_div(n, d):
    if d is None or (isinstance(d, (int, float)) and d == 0) or pd.isna(d):
        return np.nan
    return n / d


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute KPI columns safely even when inputs are missing."""
    df = df.copy()
    imp = _ensure_series(df, "impressions")
    clk = _ensure_series(df, "clicks")
    spd = _ensure_series(df, "spend")
    conv = _ensure_series(df, "conversions")
    leads = _ensure_series(df, "leads")
    reach = _ensure_series(df, "reach")
    v100 = _ensure_series(df, "video_100")
    vviews = _ensure_series(df, "video_views")

    df["ctr"] = np.where((imp > 0), clk / imp, np.nan)
    df["cpc"] = np.where((clk > 0), spd / clk, np.nan)
    df["cpm"] = np.where((imp > 0), (spd / imp) * 1000, np.nan)
    df["cvr"] = np.where((clk > 0) & (~conv.isna()), conv / clk, np.nan)
    df["cpl"] = np.where((leads > 0), spd / leads, np.nan)
    df["frequency"] = np.where((reach > 0), imp / reach, np.nan)
    df["vtr"] = np.where((imp > 0) & (~vviews.isna()), vviews / imp, np.nan)
    df["completion_rate"] = np.where((imp > 0) & (~v100.isna()), v100 / imp, np.nan)
    return df


def dedupe(df: pd.DataFrame, keys: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate by composite keys; numeric cols are summed, others take first."""
    if not keys or not all(k in df.columns for k in keys):
        return df, pd.DataFrame()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = keys
    agg: Dict[str, str] = {c: "sum" for c in numeric_cols if c not in group_cols}
    for c in df.columns:
        if c not in numeric_cols and c not in group_cols:
            agg[c] = "first"
    before = len(df)
    out = df.groupby(group_cols, dropna=False, as_index=False).agg(agg)
    after = len(out)
    report = pd.DataFrame({
        "duplicates_removed": [before - after],
        "rows_before": [before],
        "rows_after": [after],
        "keys": [", ".join(group_cols)],
    })
    return out, report


# =============================================================================
# File ingestion (shared by Streamlit & headless)
# =============================================================================

def read_csv_safely(file_like) -> pd.DataFrame:
    """Robust reader for LinkedIn exports.
    Handles UTF‑16 tab-delimited with 5-line preamble, alternate encodings,
    multiple delimiters, and skips malformed lines.
    """
    import io as _io
    from pandas.errors import ParserError

    def _read_utf16_with_preamble(b: bytes) -> Optional[pd.DataFrame]:
        try:
            text = b.decode("utf-16", errors="ignore")
        except Exception:
            return None
        sio = _io.StringIO(text)
        # Try with standard preamble (5 rows), then without
        for skip in (5, 0):
            try:
                sio.seek(0)
                return pd.read_csv(
                    sio,
                    sep="\t",
                    engine="python",
                    on_bad_lines="skip",
                    skiprows=skip,
                )
            except Exception:
                continue
        return None

    # Path-like input
    if isinstance(file_like, (str, os.PathLike)):
        with open(file_like, "rb") as fh:
            head = fh.read(8)
        if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
            with open(file_like, "rb") as fh:
                b = fh.read()
            out = _read_utf16_with_preamble(b)
            if out is not None:
                return out
        # Fallback attempts
        try:
            return pd.read_csv(file_like, sep=None, engine="python", on_bad_lines="skip")
        except Exception:
            for sep in [",", ";", "\t", "|"]:
                try:
                    return pd.read_csv(file_like, sep=sep, engine="python", on_bad_lines="skip")
                except Exception:
                    continue
            raise

    # File-like (e.g., Streamlit UploadedFile)
    try:
        raw_bytes = file_like.read()
    finally:
        if hasattr(file_like, "seek"):
            try:
                file_like.seek(0)
            except Exception:
                pass

    # UTF-16 detection on bytes (BOM or many NULs)
    if raw_bytes[:2] in (b"\xff\xfe", b"\xfe\xff") or (b"\x00" in raw_bytes[:100]):
        out = _read_utf16_with_preamble(raw_bytes)
        if out is not None:
            return out

    # Generic encoding + delimiter attempts
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    seps = [None, ",", ";", "\t", "|"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            text = raw_bytes.decode(enc, errors="ignore")
        except Exception as e:
            last_err = e
            continue
        sio = _io.StringIO(text)
        for sep in seps:
            try:
                return pd.read_csv(sio, sep=sep, engine="python", on_bad_lines="skip")
            except Exception as e:
                last_err = e
                sio.seek(0)
                continue
    raise ParserError(f"Failed to parse CSV after multiple strategies. Last error: {last_err}")


def load_files_from_paths(paths: List[str], mapping: Dict[str, List[str]]):
    campaign, creative, convo, demo = [], [], [], []
    profiles = []
    for p in paths:
        try:
            raw = read_csv_safely(p)
        except Exception as e:
            profiles.append({"name": os.path.basename(p), "error": str(e)})
            continue
        ftype = detect_file_type(raw)
        df = canonicalize_columns(raw, mapping)
        prof = {
            "name": os.path.basename(p),
            "rows": len(df),
            "cols": len(df.columns),
            "type": ftype,
            "date_min": str(df["date"].min().date()) if "date" in df.columns and df["date"].notna().any() else "—",
            "date_max": str(df["date"].max().date()) if "date" in df.columns and df["date"].notna().any() else "—",
        }
        profiles.append(prof)
        if ftype == "campaign":
            campaign.append(df)
        elif ftype == "creative":
            creative.append(df)
        elif ftype == "conversation":
            convo.append(df)
        elif ftype == "demographics":
            demo.append(df)
        else:
            campaign.append(df)

    def _concat(parts: List[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()

    return _concat(campaign), _concat(creative), _concat(convo), _concat(demo), pd.DataFrame(profiles)


# =============================================================================
# Streamlit UI (only if available)
# =============================================================================

def run_streamlit_app():
    st.set_page_config(page_title="LinkedIn Ads Analyzer", page_icon="📊", layout="wide")

    CSS = """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetricValue"] { font-weight: 800; }
    div[data-testid="stMetricDelta"] { font-weight: 600; }
    .export-btn { margin-top: .25rem; }
    .card {background: #ffffff; border-radius: 16px; padding: 16px; box-shadow: 0 2px 12px rgba(10,31,68,.06);} 
    .small { color:#6b7a8c; font-size: 0.9rem; }
    </style>
    """
    st.markdown(CSS, unsafe_allow_html=True)

    if "mapping" not in st.session_state:
        st.session_state.mapping = DEFAULT_MAPPING.copy()

    with st.sidebar:
        st.title("📥 Upload CSVs")
        files = st.file_uploader(
            "Drop your LinkedIn CSV exports (multiple allowed)",
            type=["csv"], accept_multiple_files=True
        )
        st.markdown("---")
        st.subheader("🔧 Settings")
        show_unknown = st.checkbox("Show unknown file types", value=False)

    @st.cache_data(show_spinner=False)
    def _load(_files, mapping: Dict[str, List[str]]):
        campaign, creative, convo, demo = [], [], [], []
        profiles = []
        for f in _files:
            try:
                raw = read_csv_safely(f)
            except Exception as e:
                st.error(f"Failed to read {getattr(f, 'name', 'uploaded file')}: {e}")
                continue
            ftype = detect_file_type(raw)
            df = canonicalize_columns(raw, mapping)
            prof = {
                "name": f.name,
                "rows": len(df),
                "cols": len(df.columns),
                "type": ftype,
                "date_min": str(df["date"].min().date()) if "date" in df.columns and df["date"].notna().any() else "—",
                "date_max": str(df["date"].max().date()) if "date" in df.columns and df["date"].notna().any() else "—",
            }
            profiles.append(prof)
            if ftype == "campaign":
                campaign.append(df)
            elif ftype == "creative":
                creative.append(df)
            elif ftype == "conversation":
                convo.append(df)
            elif ftype == "demographics":
                demo.append(df)
            else:
                if show_unknown:
                    st.warning(f"Unknown type for {f.name}. Keeping as campaign.")
                campaign.append(df)
        def _concat(parts):
            return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
        return _concat(campaign), _concat(creative), _concat(convo), _concat(demo), pd.DataFrame(profiles)

    if files:
        with st.spinner("Processing files…"):
            df_campaign, df_creative, df_convo, df_demo, df_profiles = _load(files, st.session_state.mapping)
    else:
        df_campaign = df_creative = df_convo = df_demo = pd.DataFrame()
        df_profiles = pd.DataFrame()

    st.header("LinkedIn Ads Analyzer")
    if not files:
        st.info("Upload CSVs in the sidebar to begin. Supported: campaign, creative, conversation ads, demographics.")
    else:
        st.subheader("Ingestion Summary")
        st.dataframe(df_profiles, use_container_width=True)

    # Dedupe + KPIs
    if not df_campaign.empty:
        df_campaign, rep_campaign = dedupe(df_campaign, ["date","account_id","campaign_id"])
        df_campaign = compute_kpis(df_campaign)
    else:
        rep_campaign = pd.DataFrame()

    if not df_creative.empty:
        df_creative, rep_creative = dedupe(df_creative, ["date","campaign_id","creative_id"])
        df_creative = compute_kpis(df_creative)
    else:
        rep_creative = pd.DataFrame()

    if not df_convo.empty:
        df_convo, rep_convo = dedupe(df_convo, ["date","campaign_id","creative_id"])
        df_convo = compute_kpis(df_convo)
    else:
        rep_convo = pd.DataFrame()

    if not df_demo.empty:
        keys = [k for k in ["campaign_id","dimension_type","dimension_value"] if k in df_demo.columns]
        df_demo, rep_demo = dedupe(df_demo, keys if keys else ["dimension_type","dimension_value"])
    else:
        rep_demo = pd.DataFrame()

    if any(len(x) for x in [rep_campaign, rep_creative, rep_convo, rep_demo]):
        st.markdown("### Dedupe Report")
        rep = pd.concat([r.assign(table=t) for r, t in [
            (rep_campaign, "campaign"),(rep_creative,"creative"),(rep_convo,"conversation"),(rep_demo,"demographics")
        ] if not r.empty], ignore_index=True)
        st.dataframe(rep, use_container_width=True)

    if not df_campaign.empty or not df_creative.empty:
        st.markdown("---")
        st.subheader("Filters")
        base = pd.concat([df_campaign, df_creative], ignore_index=True, sort=False)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            date_min = pd.to_datetime(base["date"].min()) if "date" in base.columns else None
            date_max = pd.to_datetime(base["date"].max()) if "date" in base.columns else None
            date_range = st.date_input("Date range", value=(date_min.date() if date_min is not None else datetime(2024,1,1).date(),
                                                             date_max.date() if date_max is not None else datetime.today().date()))
        with col2:
            campaigns = st.multiselect("Campaign", sorted(base.get("campaign_name", pd.Series()).dropna().unique().tolist()))
            objectives = st.multiselect("Objective", sorted(base.get("objective", pd.Series()).dropna().unique().tolist()))
        with col3:
            ad_formats = st.multiselect("Ad Format", sorted(base.get("ad_format", pd.Series()).dropna().unique().tolist()))
            countries = st.multiselect("Country", sorted(base.get("country", pd.Series()).dropna().unique().tolist()))
        with col4:
            has_conv = st.checkbox("Has conversions", value=False)
            has_video = st.checkbox("Has video views", value=False)

        def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            out = df.copy()
            if "date" in out.columns and isinstance(date_range, tuple):
                start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                out = out[(out["date"]>=start) & (out["date"]<=end)]
            if campaigns and "campaign_name" in out.columns:
                out = out[out["campaign_name"].isin(campaigns)]
            if objectives and "objective" in out.columns:
                out = out[out["objective"].isin(objectives)]
            if ad_formats and "ad_format" in out.columns:
                out = out[out["ad_format"].isin(ad_formats)]
            if countries and "country" in out.columns:
                out = out[out["country"].isin(countries)]
            if has_conv and "conversions" in out.columns:
                out = out[out["conversions"].fillna(0) > 0]
            if has_video and "video_views" in out.columns:
                out = out[out["video_views"].fillna(0) > 0]
            return out

        f_campaign = apply_filters(df_campaign)
        f_creative = apply_filters(df_creative)
    else:
        f_campaign = df_campaign
        f_creative = df_creative

    if not df_campaign.empty or not df_creative.empty:
        st.markdown("---")
        st.subheader("Key KPIs")
        def kpi_agg(df: pd.DataFrame) -> Dict[str, float]:
            if df.empty:
                return {k: np.nan for k in ["impressions","clicks","spend","conversions","leads","ctr","cpc","cpm","cvr","cpl","frequency"]}
            sums = df[[c for c in ["impressions","clicks","spend","conversions","leads","reach"] if c in df.columns]].sum(numeric_only=True)
            out: Dict[str, float] = {}
            out.update(sums.to_dict())
            out["ctr"] = kpi_safe_div(out.get("clicks",0), out.get("impressions",0))
            out["cpc"] = kpi_safe_div(out.get("spend",0), out.get("clicks",0))
            out["cpm"] = kpi_safe_div(out.get("spend",0)*1000, out.get("impressions",0))
            out["cvr"] = kpi_safe_div(out.get("conversions",0), out.get("clicks",0))
            out["cpl"] = kpi_safe_div(out.get("spend",0), out.get("leads",0))
            out["frequency"] = kpi_safe_div(out.get("impressions",0), out.get("reach",0))
            return out
        base = pd.concat([f_campaign, f_creative], ignore_index=True, sort=False)
        agg = kpi_agg(base)
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Impressions", f"{agg.get('impressions',0):,.0f}")
        c2.metric("Clicks", f"{agg.get('clicks',0):,.0f}")
        c3.metric("CTR", f"{(agg.get('ctr',np.nan)*100):.2f}%" if not pd.isna(agg.get('ctr',np.nan)) else "—")
        c4.metric("Spend", f"${agg.get('spend',0):,.2f}")
        c5.metric("CPC", f"${agg.get('cpc',np.nan):.2f}" if not pd.isna(agg.get('cpc',np.nan)) else "—")
        c6.metric("CPM", f"${agg.get('cpm',np.nan):.2f}" if not pd.isna(agg.get('cpm',np.nan)) else "—")

    if not f_campaign.empty and PLOTLY_AVAILABLE:
        st.markdown("---")
        st.subheader("Trend Explorer")
        metric = st.selectbox("Metric", [
            "impressions","clicks","spend","conversions","leads","ctr","cpc","cpm","cvr","cpl"
        ], index=0)
        ts = f_campaign.copy()
        if "date" in ts.columns:
            aggfun = "sum" if metric in {"impressions","clicks","spend","conversions","leads"} else "mean"
            ts = ts.groupby("date", as_index=False).agg({metric: aggfun})
            fig = px.line(ts, x="date", y=metric, markers=True)
            fig.update_layout(height=380, margin=dict(l=8,r=8,b=8,t=40))
            st.plotly_chart(fig, use_container_width=True)
            if KALEIDO_AVAILABLE:
                st.download_button("Download chart (PNG)", data=fig.to_image(format="png"), file_name=f"trend_{metric}.png", mime="image/png")

    if not f_campaign.empty and "ad_format" in f_campaign.columns and PLOTLY_AVAILABLE:
        st.subheader("Ad Format Performance")
        grp = f_campaign.groupby("ad_format", as_index=False).agg({
            "impressions":"sum", "clicks":"sum", "spend":"sum", "conversions":"sum"
        })
        grp["ctr"] = grp.apply(lambda r: kpi_safe_div(r["clicks"], r["impressions"]), axis=1)
        grp["cpc"] = grp.apply(lambda r: kpi_safe_div(r["spend"], r["clicks"]), axis=1)
        grp["cpm"] = grp.apply(lambda r: kpi_safe_div(r["spend"]*1000, r["impressions"]), axis=1)
        fig2 = px.bar(grp, x="ad_format", y=["ctr","cpc","cpm"], barmode="group")
        fig2.update_layout(height=420, margin=dict(l=8,r=8,b=8,t=40))
        st.plotly_chart(fig2, use_container_width=True)
        if KALEIDO_AVAILABLE:
            st.download_button("Download chart (PNG)", data=fig2.to_image(format="png"), file_name="ad_format_perf.png", mime="image/png")

    if not f_campaign.empty:
        st.subheader("Campaign Leaderboard")
        board = f_campaign.groupby("campaign_name", as_index=False).agg({
            "impressions":"sum","clicks":"sum","spend":"sum","conversions":"sum","leads":"sum"
        })
        board = compute_kpis(board)
        keep = [c for c in ["campaign_name","impressions","clicks","ctr","spend","cpc","conversions","cvr","cpl"] if c in board.columns]
        st.dataframe(board[keep].sort_values("impressions", ascending=False), use_container_width=True)
        st.download_button("Download table (CSV)", data=board.to_csv(index=False).encode("utf-8"), file_name="campaign_leaderboard.csv", mime="text/csv")

    if not df_creative.empty:
        st.markdown("---")
        st.subheader("Creative Breakdown")
        cr = df_creative.groupby(["creative_name","headline","cta"], as_index=False).agg({
            "impressions":"sum","clicks":"sum","spend":"sum","conversions":"sum","leads":"sum"
        })
        cr = compute_kpis(cr)
        keep = [c for c in ["creative_name","headline","cta","impressions","clicks","ctr","spend","cpc","conversions","cvr","cpl"] if c in cr.columns]
        st.dataframe(cr[keep].sort_values("impressions", ascending=False), use_container_width=True)
        st.download_button("Download table (CSV)", data=cr.to_csv(index=False).encode("utf-8"), file_name="creative_breakdown.csv", mime="text/csv")

    if not df_demo.empty and PLOTLY_AVAILABLE:
        st.markdown("---")
        st.subheader("Demographics")
        demo_kpi = compute_kpis(df_demo.copy())
        dim_candidates = ["dimension_value","company","industry","job_function","job_title","seniority","company_size","country"]
        dim_col = st.selectbox("Dimension", [c for c in dim_candidates if c in demo_kpi.columns])
        metric = st.selectbox("Metric", [c for c in ["impressions","clicks","spend","conversions","ctr","cpc","cpm","cvr","cpl"] if c in demo_kpi.columns], index=0)
        cat_col = "dimension_value" if "dimension_value" in demo_kpi.columns else dim_col
        grp = demo_kpi.groupby(cat_col, as_index=False).agg({
            "impressions":"sum","clicks":"sum","spend":"sum","conversions":"sum"
        })
        grp = compute_kpis(grp)
        top = grp.sort_values(metric, ascending=False).head(20)
        fig3 = px.bar(top, x=cat_col, y=metric)
        fig3.update_layout(height=420, margin=dict(l=8,r=8,b=120,t=40))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Full table")
        st.dataframe(grp.sort_values(metric, ascending=False), use_container_width=True)
        if KALEIDO_AVAILABLE:
            st.download_button("Download chart (PNG)", data=fig3.to_image(format="png"), file_name=f"demo_{metric}.png", mime="image/png")

    with st.expander("⚙️ Column Mapping (advanced)"):
        st.write("Map alternate LinkedIn headers to canonical names.")
        mapping_text = st.text_area("Mapping JSON", value=json.dumps(st.session_state.mapping, indent=2), height=220)
        if st.button("Save Mapping"):
            try:
                st.session_state.mapping = json.loads(mapping_text)
                st.success("Mapping updated. Re-upload files to apply.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    st.markdown("""
    <div class="small">Source: LinkedIn Ads CSV (uploaded). PNG export requires <code>kaleido</code>.<br>
    © 2025 — LinkedIn Ads Analyzer</div>
    """, unsafe_allow_html=True)


# =============================================================================
# Headless report builder (no Streamlit required)
# =============================================================================

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_df(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def build_headless_report(csv_paths: Optional[List[str]] = None, mapping: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
    mapping = mapping or DEFAULT_MAPPING
    outdir = ensure_outdir("/mnt/data/linkedindash_outputs")

    if not csv_paths:
        csv_paths = sorted(glob.glob("/mnt/data/*.csv"))

    df_campaign, df_creative, df_convo, df_demo, df_profiles = load_files_from_paths(csv_paths, mapping)

    write_df(os.path.join(outdir, "ingestion_profile.csv"), df_profiles)

    rep_frames = []
    if not df_campaign.empty:
        df_campaign, rep = dedupe(df_campaign, ["date","account_id","campaign_id"])
        rep["table"] = "campaign"; rep_frames.append(rep)
        df_campaign = compute_kpis(df_campaign)
    if not df_creative.empty:
        df_creative, rep = dedupe(df_creative, ["date","campaign_id","creative_id"])
        rep["table"] = "creative"; rep_frames.append(rep)
        df_creative = compute_kpis(df_creative)
    if not df_convo.empty:
        df_convo, rep = dedupe(df_convo, ["date","campaign_id","creative_id"])
        rep["table"] = "conversation"; rep_frames.append(rep)
        df_convo = compute_kpis(df_convo)
    if not df_demo.empty:
        keys = [k for k in ["campaign_id","dimension_type","dimension_value"] if k in df_demo.columns]
        df_demo, rep = dedupe(df_demo, keys if keys else ["dimension_type","dimension_value"])
        rep["table"] = "demographics"; rep_frames.append(rep)

    if rep_frames:
        write_df(os.path.join(outdir, "dedupe_report.csv"), pd.concat(rep_frames, ignore_index=True))

    base = pd.concat([df_campaign, df_creative], ignore_index=True, sort=False)
    if not base.empty:
        sums = base[[c for c in ["impressions","clicks","spend","conversions","leads","reach"] if c in base.columns]].sum(numeric_only=True)
        kpis = {
            **sums.to_dict(),
            "ctr": kpi_safe_div(sums.get("clicks",0), sums.get("impressions",0)),
            "cpc": kpi_safe_div(sums.get("spend",0), sums.get("clicks",0)),
            "cpm": kpi_safe_div(sums.get("spend",0)*1000, sums.get("impressions",0)),
            "cvr": kpi_safe_div(sums.get("conversions",0), sums.get("clicks",0)),
            "cpl": kpi_safe_div(sums.get("spend",0), sums.get("leads",0)),
            "frequency": kpi_safe_div(sums.get("impressions",0), sums.get("reach",0)),
        }
        write_df(os.path.join(outdir, "kpi_overview.csv"), pd.DataFrame([kpis]))

    if not df_campaign.empty:
        board = df_campaign.groupby("campaign_name", as_index=False).agg({
            "impressions":"sum","clicks":"sum","spend":"sum","conversions":"sum","leads":"sum"
        })
        board = compute_kpis(board)
        write_df(os.path.join(outdir, "campaign_leaderboard.csv"), board)

    if not df_creative.empty:
        cr = df_creative.groupby(["creative_name","headline","cta"], as_index=False).agg({
            "impressions":"sum","clicks":"sum","spend":"sum","conversions":"sum","leads":"sum"
        })
        cr = compute_kpis(cr)
        write_df(os.path.join(outdir, "creative_breakdown.csv"), cr)

    if not df_demo.empty:
        demo_kpi = compute_kpis(df_demo.copy())
        cat_col = "dimension_value" if "dimension_value" in demo_kpi.columns else None
        by_col = cat_col or next((c for c in ["company","industry","job_function","job_title","seniority","company_size","country"] if c in demo_kpi.columns), None)
        if by_col:
            grp = demo_kpi.groupby(by_col, as_index=False).agg({
                "impressions":"sum","clicks":"sum","spend":"sum","conversions":"sum"
            })
            grp = compute_kpis(grp)
            write_df(os.path.join(outdir, "demographics_summary.csv"), grp)

    chart_paths: Dict[str, str] = {}
    if PLOTLY_AVAILABLE and not df_campaign.empty:
        if "date" in df_campaign.columns:
            ts = df_campaign.groupby("date", as_index=False)["impressions"].sum()
            fig = px.line(ts, x="date", y="impressions", markers=True, title="Impressions Over Time")
            trend_path = os.path.join(outdir, "trend_impressions.html")
            fig.write_html(trend_path, include_plotlyjs="cdn")
            chart_paths["trend_impressions_html"] = trend_path
        if "ad_format" in df_campaign.columns:
            grp = df_campaign.groupby("ad_format", as_index=False).agg({"impressions":"sum","clicks":"sum","spend":"sum"})
            grp["ctr"] = grp.apply(lambda r: kpi_safe_div(r["clicks"], r["impressions"]), axis=1)
            fig2 = px.bar(grp, x="ad_format", y="ctr", title="CTR by Ad Format")
            adfmt_path = os.path.join(outdir, "ad_format_ctr.html")
            fig2.write_html(adfmt_path, include_plotlyjs="cdn")
            chart_paths["ad_format_ctr_html"] = adfmt_path

    index = {"outdir": outdir, **chart_paths}
    write_df(os.path.join(outdir, "artifact_index.csv"), pd.DataFrame([index]))
    return index


# =============================================================================
# Unit tests (ALWAYS present; expanded)
# =============================================================================

def run_tests() -> None:
    # 1) KPI math sanity
    df = pd.DataFrame({
        "impressions": [1000, 0],
        "clicks": [100, 0],
        "spend": [200.0, 0.0],
        "conversions": [10, 0],
        "leads": [5, 0],
        "reach": [800, 0],
    })
    out = compute_kpis(df)
    assert round(out.loc[0, "ctr"], 4) == 0.1
    assert round(out.loc[0, "cpc"], 2) == 2.00
    assert round(out.loc[0, "cpm"], 2) == 200.00
    assert round(out.loc[0, "cvr"], 2) == 0.10
    assert round(out.loc[0, "cpl"], 2) == 40.00
    assert round(out.loc[0, "frequency"], 4) == 1.25
    assert pd.isna(out.loc[1, "ctr"])  # division by zero → NaN

    # 2) Dedupe by composite keys
    df2 = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "account_id": [1, 1],
        "campaign_id": [111, 111],
        "impressions": [100, 200],
        "clicks": [10, 20],
    })
    d2, rep = dedupe(df2, ["date","account_id","campaign_id"])
    assert len(d2) == 1 and d2.loc[0, "impressions"] == 300 and d2.loc[0, "clicks"] == 30
    assert rep.loc[0, "duplicates_removed"] == 1

    # 3) Detect file type
    df_demo = pd.DataFrame({"Dimension Type": ["Company"], "Dimension Value": ["Acme"], "Impressions": [10]})
    assert detect_file_type(df_demo) == "demographics"

    # 4) Canonicalize mapping
    df_map = pd.DataFrame({"Date": ["2025-02-01"], "Clicks": [5]})
    mapped = canonicalize_columns(df_map, DEFAULT_MAPPING)
    assert "date" in mapped.columns and "clicks" in mapped.columns

    # 5) Missing video columns should not crash and should yield NaN KPIs
    df_no_video = pd.DataFrame({
        "impressions": [1000, 500],
        "clicks": [50, 10],
        "spend": [100.0, 20.0],
        # no video_views, no video_100, no conversions/leads/reach in second row
    })
    out2 = compute_kpis(df_no_video)
    assert "vtr" in out2.columns and "completion_rate" in out2.columns
    assert out2["vtr"].isna().all()
    assert out2["completion_rate"].isna().all()
    assert out2["cvr"].isna().all()

    # 6) Partial video columns present → compute only where valid
    df_video = pd.DataFrame({
        "impressions": [1000, 2000, 0],
        "video_views": [200, np.nan, 10],
        "video_100": [50, 20, np.nan],
    })
    out3 = compute_kpis(df_video)
    assert round(out3.loc[0, "vtr"], 3) == 0.2
    assert round(out3.loc[0, "completion_rate"], 3) == 0.05
    assert pd.isna(out3.loc[1, "vtr"])  # views NaN
    assert pd.isna(out3.loc[2, "vtr"])  # impressions 0 → NaN

    # 7) UTF-16 + preamble + tab-delimited should parse
    utf16_content = (
        "Conversation Ads Performance Report (in UTC)\n"
        '"Report Start: February 1, 2025, 12:00 AM"\n'
        '"Report End: August 11, 2025, 11:59 PM"\n'
        '"Date Generated: August 11, 2025, 7:02 PM"\n'
        "\n"
        "date\timpressions\tclicks\n"
        "2025-06-01\t100\t10\n"
    ).encode("utf-16")
    import io as _io
    df_utf16 = read_csv_safely(_io.BytesIO(utf16_content))
    assert set(df_utf16.columns) >= {"date","impressions","clicks"}

    # 8) Malformed line is skipped, not fatal
    bad = (
        "impressions,clicks\n"
        "100,10\n"
        "this,is,an,extra,column\n"
        "200,20\n"
    )
    df_bad = read_csv_safely(_io.StringIO(bad))
    assert df_bad.shape[0] == 2 and set(df_bad.columns) == {"impressions","clicks"}


# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    try:
        run_tests()
        print("[tests] ✅ All unit tests passed.")
    except AssertionError as e:
        print("[tests] ❌ A unit test failed:", e)

    if STREAMLIT_AVAILABLE:
        run_streamlit_app()
    else:
        print("[info] Streamlit not available — running headless report builder.")
        artifacts = build_headless_report()
        print("[done] Artifacts written to:", artifacts.get("outdir"))
        for k, v in artifacts.items():
            if k != "outdir":
                print(f"  - {k}: {v}")
