"""
app.py
------
CustomerSegmentationDashboard
DecodeLabs Project 3 -- Unsupervised Learning (Customer Segmentation).

Run with:  streamlit run app.py

Design direction (light theme only -- no black, no red anywhere)
    Soft gradients instead of solid blocks, generous corner radii, tinted
    low-opacity shadows for depth, and short hover transitions.

    #96f7e4  mint        -- gradient start, soft fills
    #53eafd  light aqua  -- highlights
    #00b8db  cyan        -- primary interactive colour
    #009689  teal        -- gradient end, emphasis, threshold markers
    #0092b8  medium blue -- ALL body text

    Title:    hsl(230.97, 99.04%, 59.22%), enlarged, glow + pulse animation
    Tables:   text hsl(216.49, 85.06%, 34.12%), bold + larger header row
    Download: box hsl(186.89, 71.15%, 59.22%), text hsl(172.57, 100%, 20.59%)
    Sidebar:  the << / >> chevron is replaced by a hamburger in
              hsl(180, 100%, 54.71%)

    Streamlit's own chrome is recoloured in .streamlit/config.toml.

TWO CSS BLOCKS, ON PURPOSE
    inject_css()       is an f-string (it interpolates the palette constants),
                       so every CSS brace inside it is doubled: {{ }}.
    inject_overrides() is a plain string, so its braces are written normally.
                       It loads second, which means its rules win on source
                       order without needing extra specificity.

WHY TABLES ARE RENDERED AS HTML
    Recent Streamlit versions draw st.dataframe on an HTML <canvas>
    (glide-data-grid). A canvas has no <table>, <thead> or <th> elements
    inside it, so CSS such as `[data-testid="stDataFrame"] table thead tr th`
    matches nothing and silently does nothing. Every table below therefore
    goes through render_table(), which emits a real HTML <table>.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.clustering import (
    cluster_quality_summary,
    evaluate_k_range,
    fit_kmeans,
    recommend_k,
    silhouette_breakdown,
)
from src.dimensionality import (
    fit_pca,
    loadings_table,
    project_for_plot,
    variance_table,
)
from src.personas import (
    build_personas,
    centroid_table,
    profile_clusters,
    revenue_index,
)
from src.preprocessing import (
    DEFAULT_FEATURES,
    NUMERIC_COLS,
    build_feature_matrix,
    encode_gender,
    handle_missing,
    quality_report,
    scale_features,
)

# --------------------------------------------------------------------------
# Design tokens (kept in one place so charts and CSS never drift apart)
# --------------------------------------------------------------------------
MINT     = "#96f7e4"   # mint        -- gradient start, soft fills
PRIMARY  = "#53eafd"   # light aqua  -- highlights, gradient mid-tones
ACCENT   = "#00b8db"   # cyan        -- primary interactive colour
DEEP     = "#009689"   # teal        -- gradient end, emphasis, thresholds
INK      = "#0092b8"   # medium blue -- ALL body text (never black)
WASH     = "#EDFDFF"   # 6% tint of MINT -- panel backgrounds
CONTRAST = "#009689"   # teal        -- threshold lines and markers (never red)
MUTED    = "#4FB8CE"   # softened INK -- secondary label text

# Table + download-button palette (requested exactly in HSL)
TABLE_INK = "hsl(216.49, 85.06%, 34.12%)"   # all table text
TABLE_LINE = "hsl(180, 100%, 27.25%)"       # table borders
DL_BG = "hsl(186.89, 71.15%, 59.22%)"       # download button box
DL_INK = "hsl(172.57, 100%, 20.59%)"        # download button text

# Header title + hamburger palette
TITLE_INK = "hsl(230.97, 99.04%, 59.22%)"   # app title in the header
BURGER = "hsl(180, 100%, 54.71%)"           # hamburger bars

# Categorical scale for charts: the brand colours plus in-family blends, so a
# K of 6+ never falls back to a Plotly default series colour (which is red).
CHART_COLORS = [ACCENT, DEEP, INK, PRIMARY, "#00786E", MINT, "#0086A8", "#4FC3D9"]

PLOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.6)",
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=CHART_COLORS,
)

# Every section carries a plain-language description so the user always knows
# what a tab contains -- reused by the header nav, the hamburger drawer and the
# sidebar captions from this single definition.
SECTIONS = [
    ("overview",  "Overview",             "Executive summary"),
    ("data",      "Data & Preprocessing", "Cleaning, encoding, scaling"),
    ("pca",       "PCA Compression",      "Dimensionality reduction"),
    ("optimal_k", "Optimal K",            "Elbow + Silhouette diagnostics"),
    ("clusters",  "Cluster Explorer",     "Visualise the clusters"),
    ("personas",  "Business Personas",    "Persona cards + actions"),
]


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def inject_css() -> None:
    """
    Design system for CustomerSegmentationDashboard.

    Everything is built from soft gradients rather than solid blocks, with
    generous corner radii, low-opacity shadows for depth, and short hover
    transitions (lift + colour shift + underline sweep). No black, no red.

    NOTE: this is an f-string, so every literal CSS brace is doubled.
    """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --mint:    #96f7e4;   /* gradient start   */
            --aqua:    #53eafd;   /* light highlight  */
            --cyan:    #00b8db;   /* primary action   */
            --teal:    #009689;   /* gradient end     */
            --ink:     #0092b8;   /* all body text    */
            --muted:   #4FB8CE;   /* secondary labels */
            --wash:    #EDFDFF;   /* panel background */
            --wash-2:  #F5FEFF;   /* lighter panel    */

            /* table + download button palette */
            --table-ink:  {TABLE_INK};
            --table-line: {TABLE_LINE};
            --dl-bg:      {DL_BG};
            --dl-ink:     {DL_INK};

            /* header title + hamburger */
            --title-ink:  {TITLE_INK};
            --burger:     {BURGER};

            /* the signature header / nav gradient */
            --grad-brand: linear-gradient(90deg, #96f7e4 0%, #00b8db 45%, #009689 100%);
            --grad-panel: linear-gradient(150deg, #FFFFFF 0%, #F5FEFF 55%, #EDFDFF 100%);
            --grad-soft:  linear-gradient(135deg, rgba(150,247,228,.40), rgba(83,234,253,.28));
            --grad-btn:   linear-gradient(90deg, #00b8db 0%, #009689 100%);

            /* shadows stay tinted rather than grey, so nothing reads as black */
            --shadow-sm: 0 2px 10px rgba(0, 146, 184, 0.10);
            --shadow-md: 0 10px 26px rgba(0, 150, 137, 0.13);
            --shadow-lg: 0 20px 44px rgba(0, 146, 184, 0.18);
            --radius:    22px;
            --radius-sm: 14px;
        }}

        /* ---------- canvas ---------- */
        .stApp {{
            background:
                radial-gradient(1100px 520px at 8% -10%, rgba(150,247,228,.28) 0%, transparent 62%),
                radial-gradient(900px 480px at 95% 0%, rgba(83,234,253,.22) 0%, transparent 60%),
                linear-gradient(170deg, #FFFFFF 0%, #F7FEFF 50%, #EEFDFF 100%);
            color: var(--ink);
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        /* Streamlit's own toolbar defaults to a dark bar -- neutralise it */
        header[data-testid="stHeader"] {{ background: transparent !important; }}
        header[data-testid="stHeader"] * {{ color: var(--ink) !important; }}
        div[data-testid="stToolbar"] {{ background: transparent !important; }}
        .block-container {{ padding-top: 1.2rem; max-width: 1400px; }}

        /* modern typography: tight, confident headings, airy body text */
        h1, h2, h3, h4, h5, p, span, label, li, div {{ color: var(--ink); }}
        h1, h2, h3, h4 {{
            color: var(--ink) !important;
            letter-spacing: -0.02em; font-weight: 700;
        }}
        p, li, label {{ line-height: 1.6; }}

        /* ---------- motion ---------- */
        @keyframes fadeUp {{
            from {{ opacity: 0; transform: translateY(18px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes drift {{
            0%   {{ background-position: 0% 50%; }}
            50%  {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        .animate {{ animation: fadeUp .55s cubic-bezier(.22,.9,.3,1) both; }}
        .delay-1 {{ animation-delay: .07s; }}
        .delay-2 {{ animation-delay: .14s; }}
        .delay-3 {{ animation-delay: .21s; }}
        .delay-4 {{ animation-delay: .28s; }}

        /* ---------- header + navigation ---------- */
        .app-header {{
            background: linear-gradient(90deg, #96f7e4 0%, #20B2AA 45%, #009689 100%);
            background-size: 200% 200%;
            animation: drift 16s ease infinite;
            border-radius: var(--radius);
            padding: 24px 28px 20px;
            box-shadow: var(--shadow-md);
            margin-bottom: 24px;
            position: relative; overflow: hidden;
        }}
        .app-header::after {{
            content: "";
            position: absolute; inset: 0;
            background: radial-gradient(680px 220px at 88% 0%, rgba(255,255,255,.45), transparent 72%);
            pointer-events: none;
        }}
        .header-top {{
            display: flex; align-items: center; justify-content: space-between;
            gap: 18px; flex-wrap: wrap; position: relative; z-index: 1;
        }}
        .brand {{ display: flex; align-items: center; gap: 14px; }}
        .brand-mark {{
            width: 48px; height: 48px; border-radius: 16px;
            background: linear-gradient(150deg, #FFFFFF, #EDFDFF);
            color: var(--cyan) !important;
            display: grid; place-items: center; font-size: 22px;
            box-shadow: var(--shadow-sm);
            transition: transform .4s cubic-bezier(.22,.9,.3,1);
        }}
        .app-header:hover .brand-mark {{ transform: rotate(-8deg) scale(1.07); }}
        /* base title rule -- colour, size and animation are overridden in
           inject_overrides() so the values live in one obvious place */
        .brand h1 {{
            margin: 0; font-size: 1.42rem; font-weight: 800;
            color: #FFFFFF !important; letter-spacing: -0.02em;
        }}
        .brand p {{
            margin: 3px 0 0; font-size: .82rem; font-weight: 500;
            color: #FFFFFF !important; opacity: .93;
        }}

        .header-status {{
            background: rgba(255,255,255,.95); border-radius: var(--radius-sm);
            padding: 10px 18px; box-shadow: var(--shadow-sm);
            display: flex; flex-direction: column; min-width: 200px;
        }}
        .status-label {{ font-size: .62rem; letter-spacing: .14em; font-weight: 700;
                        text-transform: uppercase; color: var(--muted) !important; }}
        .status-value {{ font-size: 1rem; font-weight: 700; color: var(--ink) !important; }}
        .status-desc  {{ font-size: .72rem; color: var(--teal) !important; font-weight: 500; }}

        /* horizontal nav -- every tab states what it contains */
        .nav-bar {{
            position: relative; z-index: 1;
            margin-top: 18px; display: flex; gap: 10px; flex-wrap: wrap;
        }}
        .nav-item {{
            position: relative; display: flex; flex-direction: column;
            text-decoration: none !important;
            background: linear-gradient(150deg, rgba(255,255,255,.94), rgba(237,253,255,.88));
            border-radius: var(--radius-sm);
            padding: 9px 16px 11px; min-width: 138px;
            box-shadow: 0 2px 8px rgba(0,146,184,.10);
            transition: transform .3s cubic-bezier(.22,.9,.3,1),
                        box-shadow .3s ease, background .3s ease;
        }}
        /* underline sweep on hover */
        .nav-item::after {{
            content: ""; position: absolute; left: 16px; right: 16px; bottom: 6px;
            height: 2px; border-radius: 2px;
            background: var(--grad-btn);
            transform: scaleX(0); transform-origin: left;
            transition: transform .32s cubic-bezier(.22,.9,.3,1);
        }}
        .nav-item:hover {{
            transform: translateY(-3px); box-shadow: var(--shadow-md);
            background: #FFFFFF;
        }}
        .nav-item:hover::after, .nav-item.active::after {{ transform: scaleX(1); }}
        .nav-item:hover .nav-label {{ color: var(--cyan) !important; }}
        .nav-item.active {{
            background: #FFFFFF; box-shadow: var(--shadow-md);
        }}
        .nav-item.active .nav-label {{ color: var(--teal) !important; }}
        .nav-label {{ font-size: .83rem; font-weight: 700; color: var(--ink) !important;
                     line-height: 1.25; transition: color .3s ease; }}
        .nav-desc  {{ font-size: .68rem; font-weight: 500; color: var(--muted) !important;
                     line-height: 1.35; }}

        /* ---------- KPI tiles ---------- */
        .kpi {{
            background: var(--grad-panel);
            border-radius: var(--radius); padding: 18px 22px;
            box-shadow: var(--shadow-sm);
            position: relative; overflow: hidden;
            transition: transform .3s cubic-bezier(.22,.9,.3,1), box-shadow .3s ease;
        }}
        /* gradient accent rail instead of a hard border */
        .kpi::before {{
            content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
            background: var(--grad-brand);
        }}
        .kpi:hover {{ transform: translateY(-5px); box-shadow: var(--shadow-lg); }}
        .kpi .label {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .1em;
                      font-weight: 600; color: var(--muted) !important; }}
        .kpi .value {{ font-size: 1.8rem; font-weight: 800; line-height: 1.15;
                      color: var(--ink) !important; }}
        .kpi .sub   {{ font-size: .76rem; font-weight: 600; color: var(--teal) !important; }}

        /* ---------- persona cards ---------- */
        .persona-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 22px;
        }}
        .persona {{
            background: var(--grad-panel); border-radius: var(--radius);
            overflow: hidden; box-shadow: var(--shadow-sm);
            display: flex; flex-direction: column;
            transition: transform .38s cubic-bezier(.22,.9,.3,1), box-shadow .38s ease;
        }}
        .persona:hover {{ transform: translateY(-8px) scale(1.012); box-shadow: var(--shadow-lg); }}
        .persona-head {{
            padding: 16px 20px; background: var(--grad-brand);
            display: flex; align-items: center; gap: 12px;
        }}
        .persona-head .title {{ font-weight: 700; font-size: 1.02rem; color: #FFFFFF !important; }}
        .persona-head .tag   {{ font-size: .74rem; font-weight: 500;
                               color: #FFFFFF !important; opacity: .95; }}
        .pill {{
            margin-left: auto; font-size: .66rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: .07em;
            background: rgba(255,255,255,.95); color: var(--teal) !important;
            padding: 5px 11px; border-radius: 999px; box-shadow: var(--shadow-sm);
        }}
        .persona-body {{ padding: 16px 20px 20px; flex: 1; }}
        .stat-row {{ display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }}
        .stat {{
            flex: 1; min-width: 64px; text-align: center; padding: 10px 6px;
            border-radius: var(--radius-sm);
            background: var(--grad-soft);
            transition: background .3s ease, transform .3s ease;
        }}
        .persona:hover .stat {{ transform: translateY(-2px); }}
        .stat .s-label {{ font-size: .63rem; letter-spacing: .08em; text-transform: uppercase;
                         font-weight: 600; color: var(--muted) !important; }}
        .stat .s-value {{ font-size: 1.05rem; font-weight: 700; color: var(--ink) !important; }}
        .persona ul {{ margin: 0; padding-left: 18px; }}
        .persona li {{ font-size: .83rem; line-height: 1.5; margin-bottom: 7px;
                      color: var(--muted) !important; }}
        .bar-track {{
            height: 8px; border-radius: 99px; overflow: hidden; margin: 5px 0 14px;
            background: rgba(150,247,228,.35);
        }}
        .bar-fill {{
            height: 100%; border-radius: 99px; background: var(--grad-brand);
            animation: grow 1.1s cubic-bezier(.22,.9,.3,1) both;
        }}
        @keyframes grow {{ from {{ width: 0; }} }}

        /* ---------- callouts ---------- */
        .note {{
            background: var(--grad-soft); border-radius: var(--radius-sm);
            border-left: 4px solid var(--cyan);
            padding: 15px 20px; font-size: .87rem; line-height: 1.6;
            color: var(--ink) !important; box-shadow: var(--shadow-sm);
        }}
        .note.warn {{ border-left-color: var(--teal); }}

        /* ==================================================================
           TABLES -- one rule set used by every table in the app.
           render_table() emits <table class="seg-table">, so these rules
           always bite (unlike the canvas-based st.dataframe).
           ================================================================== */
        .seg-table-wrap {{
            overflow-x: auto;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,146,184,0.15);
            margin: 6px 0 18px;
            background: #FFFFFF;
        }}
        table.seg-table {{
            width: 100%;
            border-collapse: collapse !important;
            border: 2px solid var(--table-line) !important;
            border-radius: 12px;
            overflow: hidden;
            background: #FFFFFF;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }}
        /* header row: bold and one step larger than the body cells */
        table.seg-table thead th {{
            background-color: rgba(150,247,228,0.30) !important;
            color: var(--table-ink) !important;
            font-weight: 800 !important;
            font-size: 0.97rem !important;
            letter-spacing: .01em;
            text-align: left;
            padding: 12px 14px !important;
            border-bottom: 2px solid var(--table-line) !important;
            border-right: 1px solid rgba(0,146,184,0.18) !important;
            white-space: nowrap;
        }}
        table.seg-table tbody td, table.seg-table tbody th {{
            color: var(--table-ink) !important;
            font-size: 0.84rem !important;
            font-weight: 500;
            padding: 9px 14px !important;
            border: 1px solid rgba(0,146,184,0.18) !important;
        }}
        table.seg-table tbody tr:nth-child(even) {{ background: rgba(237,253,255,0.65); }}
        table.seg-table tbody tr:hover {{ background: rgba(150,247,228,0.22); }}
        table.seg-table td.num, table.seg-table th.num {{ text-align: right; }}

        /* Fallback: older Streamlit builds that still render a real <table> */
        [data-testid="stDataFrame"] table {{
            border-collapse: collapse !important;
            border: 2px solid var(--table-line) !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 12px rgba(0,146,184,0.15) !important;
        }}
        [data-testid="stDataFrame"] table thead tr th {{
            background-color: rgba(150,247,228,0.28) !important;
            color: var(--table-ink) !important;
            font-weight: 800 !important;
            font-size: 0.97rem !important;
            border-bottom: 2px solid var(--table-line) !important;
            padding: 10px !important;
        }}
        [data-testid="stDataFrame"] table td,
        [data-testid="stDataFrame"] table th {{
            border: 1px solid rgba(0,146,184,0.18) !important;
            color: var(--table-ink) !important;
        }}
        /* st.table (static) gets the same treatment */
        [data-testid="stTable"] table {{
            border-collapse: collapse !important;
            border: 2px solid var(--table-line) !important;
        }}
        [data-testid="stTable"] th {{
            background-color: rgba(150,247,228,0.28) !important;
            color: var(--table-ink) !important;
            font-weight: 800 !important; font-size: 0.97rem !important;
            border-bottom: 2px solid var(--table-line) !important;
        }}
        [data-testid="stTable"] td {{
            color: var(--table-ink) !important;
            border: 1px solid rgba(0,146,184,0.18) !important;
        }}

        /* ---------- Streamlit widget overrides ---------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #FFFFFF 0%, #F2FEFF 55%, #E9FCFB 100%) !important;
            border-right: 1px solid rgba(0,184,219,.18);
        }}
        section[data-testid="stSidebar"] * {{ color: var(--ink) !important; }}

        /* multiselect chips default to Streamlit red -- force the brand gradient */
        span[data-baseweb="tag"] {{
            background: var(--grad-btn) !important;
            border-radius: 9px !important; border: none !important;
        }}
        span[data-baseweb="tag"] span,
        span[data-baseweb="tag"] svg {{ color: #FFFFFF !important; fill: #FFFFFF !important; }}

        /* sliders default to Streamlit red -- force brand teal */
        div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {{
            background-color: var(--teal) !important;
            border-color: var(--teal) !important;
            box-shadow: var(--shadow-sm) !important;
        }}
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {{
            background: var(--grad-btn) !important;
        }}
        div[data-testid="stSlider"] [data-testid="stTickBarMin"],
        div[data-testid="stSlider"] [data-testid="stTickBarMax"],
        div[data-testid="stSliderThumbValue"] {{ color: var(--ink) !important; }}

        div[data-testid="stRadio"] label span:first-child,
        div[data-testid="stCheckbox"] label span:first-child {{
            background-color: var(--cyan) !important; border-color: var(--cyan) !important;
        }}
        input:checked + div {{ background-color: var(--teal) !important; }}

        div[data-testid="stFileUploader"] section {{
            background: var(--grad-panel) !important;
            border: 1.5px dashed var(--cyan) !important;
            border-radius: var(--radius-sm);
        }}

        div[data-testid="stMetric"] {{
            background: var(--grad-panel); border-radius: var(--radius-sm);
            padding: 15px 18px; box-shadow: var(--shadow-sm);
            transition: transform .28s ease, box-shadow .28s ease;
        }}
        div[data-testid="stMetric"]:hover {{ transform: translateY(-3px); box-shadow: var(--shadow-md); }}
        div[data-testid="stMetric"] * {{ color: var(--ink) !important; }}

        /* ---------- buttons ---------- */
        /* Regular action buttons keep the brand gradient. */
        .stButton > button {{
            background: var(--grad-btn); color: #FFFFFF !important;
            border: none; border-radius: var(--radius-sm);
            padding: .58rem 1.25rem; font-weight: 600; letter-spacing: .01em;
            box-shadow: var(--shadow-sm); transition: all .3s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px); box-shadow: var(--shadow-md); filter: brightness(1.08);
        }}

        /* DOWNLOAD BUTTONS
           An earlier version set `background-color` here, but the shared
           button rule had already set `background: var(--grad-btn)` -- a
           background IMAGE. background-color paints *behind* an image, so the
           gradient stayed on top and the new colour never showed. Using the
           `background` shorthand (plus background-image: none) clears it. The
           label sits inside a nested <p>/<div>, so the colour is forced on
           descendants too. */
        .stDownloadButton > button,
        div[data-testid="stDownloadButton"] > button,
        .stDownloadButton > button:focus,
        .stDownloadButton > button:active,
        .stDownloadButton > button:focus:not(:active) {{
            background: var(--dl-bg) !important;
            background-image: none !important;
            color: var(--dl-ink) !important;
            font-weight: 700 !important;
            border: 2px solid var(--dl-ink) !important;
            border-radius: 10px !important;
            padding: .58rem 1.25rem !important;
            box-shadow: 0 4px 10px rgba(0,146,184,0.15) !important;
            transition: all 0.3s ease !important;
        }}
        .stDownloadButton > button *,
        div[data-testid="stDownloadButton"] > button * {{
            color: var(--dl-ink) !important;
            fill: var(--dl-ink) !important;
            font-weight: 700 !important;
        }}
        .stDownloadButton > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {{
            background: var(--dl-bg) !important;
            background-image: none !important;
            border-color: var(--dl-ink) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 14px rgba(0,146,184,0.25) !important;
            filter: brightness(1.05);
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: var(--radius-sm) var(--radius-sm) 0 0;
            padding: 9px 20px; transition: background .28s ease, color .28s ease;
        }}
        .stTabs [data-baseweb="tab"]:hover {{ background: rgba(150,247,228,.25); }}
        .stTabs [aria-selected="true"] {{ background: var(--grad-soft); }}
        .stTabs [data-baseweb="tab-highlight"] {{ background: var(--grad-btn) !important; }}

        div[data-testid="stExpander"] {{
            border: none; border-radius: var(--radius-sm);
            background: var(--grad-panel); box-shadow: var(--shadow-sm);
        }}
        a {{ color: var(--cyan) !important; transition: color .25s ease; }}
        a:hover {{ color: var(--teal) !important; }}

        /* ---------- responsive ---------- */
        @media (max-width: 900px) {{
            .header-top {{ flex-direction: column; align-items: flex-start; }}
            .header-status {{ width: 100%; }}
            .nav-item {{ min-width: calc(50% - 5px); }}
        }}
        @media (max-width: 640px) {{
            /* the hamburger drawer replaces the nav bar on phones */
            .nav-bar {{ display: none; }}
            .persona-grid {{ grid-template-columns: 1fr; }}
            .kpi .value {{ font-size: 1.45rem; }}
            .block-container {{ padding-left: .7rem; padding-right: .7rem; }}
            table.seg-table thead th {{ font-size: .88rem !important; padding: 10px !important; }}
            table.seg-table tbody td {{ font-size: .78rem !important; padding: 8px 10px !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_overrides() -> None:
    """
    Late-loading overrides: header title colour/animation and the hamburger
    sidebar toggle.

    Kept in its own NON-f-string block so the CSS braces don't need doubling,
    and injected after inject_css() so these rules win on source order without
    needing extra specificity hacks.
    """
    st.markdown(
        """
        <style>
        /* ============================================================
           1. HEADER TITLE
           The base rule sets `color: #FFFFFF !important`, so this needs
           !important too -- equal specificity, later in source order wins.
           ============================================================ */
        .app-header .brand h1 {
            color: hsl(230.97, 99.04%, 59.22%) !important;
            font-size: 1.95rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.025em;
            line-height: 1.15;
            /* white halo first so the blue stays legible on the mint end of
               the gradient, then a soft coloured glow underneath */
            text-shadow:
                0 1px 0 rgba(255,255,255,.85),
                0 0 12px rgba(255,255,255,.55),
                0 6px 18px rgba(30, 80, 255, .28);
            animation: titlePulse 3.6s ease-in-out infinite;
            transition: transform .35s cubic-bezier(.22,.9,.3,1);
        }
        .app-header:hover .brand h1 { transform: translateY(-2px) scale(1.02); }

        @keyframes titlePulse {
            0%, 100% {
                text-shadow:
                    0 1px 0 rgba(255,255,255,.85),
                    0 0 12px rgba(255,255,255,.55),
                    0 6px 18px rgba(30, 80, 255, .28);
            }
            50% {
                text-shadow:
                    0 1px 0 rgba(255,255,255,.95),
                    0 0 22px rgba(255,255,255,.85),
                    0 8px 26px rgba(30, 80, 255, .42);
            }
        }
        /* respect reduced-motion preferences */
        @media (prefers-reduced-motion: reduce) {
            .app-header .brand h1 { animation: none; }
        }
        @media (max-width: 640px) {
            .app-header .brand h1 { font-size: 1.35rem !important; }
        }

        /* ============================================================
           2. HAMBURGER SIDEBAR TOGGLE

           The << / >> arrow is NOT always an <svg>. Current Streamlit
           renders it as Material-Symbols LIGATURE TEXT inside
           <span data-testid="stIconMaterial"> -- the literal string
           "keyboard_double_arrow_left" turned into a glyph by the font.
           That is why `svg { display: none }` alone left the arrow on
           screen.

           Three things kill it for good:
             font-size: 0   -- a ligature font renders nothing at zero size
             > *            -- hides every child whatever its tag
             z-index: 2     -- keeps the bars above anything left behind

           The three bars are drawn on the button itself: one ::before bar
           plus two more via box-shadow offset above and below. No extra
           markup needed. The selector list covers several Streamlit
           versions -- entries that match nothing are harmless.
           ============================================================ */
        [data-testid="stSidebarCollapseButton"] button,
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="collapsedControl"] button,
        [data-testid="stExpandSidebarButton"],
        button[kind="headerNoPadding"],
        button[data-testid="baseButton-headerNoPadding"] {
            position: relative !important;
            width: 42px !important;
            height: 42px !important;
            min-width: 42px !important;
            padding: 0 !important;
            border-radius: 13px !important;
            background: rgba(255,255,255,.92) !important;
            background-image: none !important;
            border: 2px solid hsl(180, 100%, 44%) !important;
            box-shadow: 0 4px 12px rgba(0,146,184,.18) !important;
            overflow: hidden !important;
            /* kills the Material-Symbols ligature glyph */
            font-size: 0 !important;
            line-height: 0 !important;
            color: transparent !important;
            transition: transform .3s cubic-bezier(.22,.9,.3,1),
                        box-shadow .3s ease, background .3s ease !important;
        }

        /* hide EVERY child: svg, icon span, text-node wrapper, anything */
        [data-testid="stSidebarCollapseButton"] button > *,
        [data-testid="stSidebarCollapsedControl"] button > *,
        [data-testid="collapsedControl"] button > *,
        [data-testid="stExpandSidebarButton"] > *,
        button[kind="headerNoPadding"] > *,
        button[data-testid="baseButton-headerNoPadding"] > *,
        [data-testid="stSidebarCollapseButton"] button svg,
        [data-testid="stSidebarCollapsedControl"] button svg,
        [data-testid="collapsedControl"] button svg,
        [data-testid="stIconMaterial"] {
            display: none !important;
            visibility: hidden !important;
            font-size: 0 !important;
        }

        /* the three bars */
        [data-testid="stSidebarCollapseButton"] button::before,
        [data-testid="stSidebarCollapsedControl"] button::before,
        [data-testid="collapsedControl"] button::before,
        [data-testid="stExpandSidebarButton"]::before,
        button[kind="headerNoPadding"]::before,
        button[data-testid="baseButton-headerNoPadding"]::before {
            content: "" !important;
            position: absolute !important;
            left: 50%; top: 50%;
            z-index: 2;
            width: 20px; height: 2.6px;
            margin-left: -10px; margin-top: -1.3px;
            border-radius: 3px;
            background: hsl(180, 100%, 54.71%) !important;
            box-shadow:
                0  6.5px 0 hsl(180, 100%, 54.71%),
                0 -6.5px 0 hsl(180, 100%, 54.71%) !important;
            transition: background .3s ease, box-shadow .3s ease;
        }

        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="stSidebarCollapsedControl"] button:hover,
        [data-testid="collapsedControl"] button:hover,
        [data-testid="stExpandSidebarButton"]:hover,
        button[kind="headerNoPadding"]:hover,
        button[data-testid="baseButton-headerNoPadding"]:hover {
            transform: translateY(-2px) scale(1.05) !important;
            background: #FFFFFF !important;
            box-shadow: 0 6px 16px rgba(0,146,184,.28) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def html(markup: str) -> None:
    """
    Render raw HTML safely inside Streamlit.

    Streamlit runs every st.markdown string through a Markdown parser first,
    and Markdown treats any line indented by four or more spaces as a CODE
    BLOCK. Multi-line HTML written inside an indented f-string therefore gets
    printed as literal text instead of rendering. Collapsing the markup to a
    single unindented line removes the ambiguity entirely.
    """
    flattened = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flattened, unsafe_allow_html=True)


def render_table(df: pd.DataFrame, index: bool = False, decimals: int = 2) -> None:
    """
    Render a DataFrame as a REAL HTML table so the .seg-table CSS applies.

    st.dataframe draws onto a <canvas> in current Streamlit versions, which no
    stylesheet can reach -- that is why the table rules appeared to be ignored.
    pandas.to_html gives back genuine <table>/<thead>/<th> markup instead.

    The output is passed through html() because pandas indents its rows, and
    indented lines would otherwise be swallowed by Streamlit's Markdown parser
    as a code block.

    Parameters
    ----------
    df       : the frame to display
    index    : keep the DataFrame index as a first column
    decimals : rounding applied to float columns before rendering
    """
    frame = df.copy()
    float_cols = frame.select_dtypes(include=["float", "float64", "float32"]).columns
    frame[float_cols] = frame[float_cols].round(decimals)

    markup = frame.to_html(
        index=index,
        border=0,
        classes="seg-table",
        escape=False,
        na_rep="—",
        float_format=lambda v: f"{v:,.{decimals}f}",
    )
    html(f'<div class="seg-table-wrap">{markup}</div>')


def render_header(active: str) -> None:
    nav = "".join(
        f'<a class="nav-item{" active" if label == active else ""}" href="#{key}">'
        f'<span class="nav-label">{label}</span>'
        f'<span class="nav-desc">{desc}</span></a>'
        for key, label, desc in SECTIONS
    )
    current = next((d for _, l, d in SECTIONS if l == active), "")
    html(
        f"""
        <div class="app-header animate">
          <div class="header-top">
            <div class="brand">
              <div class="brand-mark">&#9673;</div>
              <div>
                <h1>CustomerSegmentationDashboard</h1>
                <p>DecodeLabs Project 3 · unsupervised segmentation with PCA + K-Means</p>
              </div>
            </div>
            <div class="header-status">
              <span class="status-label">Currently viewing</span>
              <span class="status-value">{active}</span>
              <span class="status-desc">{current}</span>
            </div>
          </div>
          <nav class="nav-bar">{nav}</nav>
        </div>
        """
    )


def kpi(label: str, value: str, sub: str = "", delay: int = 1) -> str:
    """Return the HTML for a single animated KPI tile."""
    return (
        f'<div class="kpi animate delay-{delay}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def styled(fig: go.Figure, height: int = 400) -> go.Figure:
    """Apply the shared design tokens to any Plotly figure."""
    fig.update_layout(**PLOT_LAYOUT, height=height)
    return fig


# --------------------------------------------------------------------------
# Cached analysis -- clustering logic is unchanged, it all lives in src/
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_analysis(
    csv_bytes: bytes,
    features: tuple[str, ...],
    variance: float,
    k_min: int,
    k_max: int,
    k_override: int | None,
) -> dict:
    """
    Full SCALE -> COMPRESS -> CLUSTER -> TRANSLATE run.

    Cached on its inputs so moving a slider re-runs the maths only when a
    parameter actually changed, keeping the UI instant.
    """
    raw = pd.read_csv(io.BytesIO(csv_bytes))
    raw.columns = [c.strip() for c in raw.columns]

    # Phase 1 -- SCALE
    report = quality_report(raw)
    clean, missing_log = handle_missing(raw)
    clean = encode_gender(clean)

    X = build_feature_matrix(clean, list(features))
    X_scaled, scaler = scale_features(X)

    # Phase 2 -- COMPRESS
    Z, pca = fit_pca(X_scaled, variance)
    var_table = variance_table(X_scaled)
    loadings = loadings_table(pca, list(features))
    Z2, _ = project_for_plot(X_scaled, 2)
    Z3, _ = project_for_plot(X_scaled, min(3, X_scaled.shape[1]))

    # Phase 3 -- CLUSTER
    scores = evaluate_k_range(Z, k_min, k_max)
    rec = recommend_k(scores)
    k = int(k_override or rec["recommended_k"])

    kmeans = fit_kmeans(Z, k)
    labels = kmeans.labels_
    quality = cluster_quality_summary(Z, labels)
    sil_samples = silhouette_breakdown(Z, labels)

    # Phase 4 -- TRANSLATE
    profile = profile_clusters(clean, labels, list(features))
    centroids = centroid_table(kmeans, pca, scaler, list(features))
    personas = revenue_index(build_personas(clean, labels, profile, quality))

    segmented = clean.copy()
    segmented["Cluster"] = labels
    segmented["Persona"] = segmented["Cluster"].map(
        dict(zip(personas["cluster"], personas["persona"]))
    )

    return {
        "raw": raw, "clean": clean, "report": report, "missing_log": missing_log,
        "features": list(features), "X_scaled": X_scaled,
        "Z": Z, "Z2": Z2, "Z3": Z3, "n_components": int(pca.n_components_),
        "variance_kept": float(pca.explained_variance_ratio_.sum()),
        "var_table": var_table, "loadings": loadings,
        "scores": scores, "rec": rec, "k": k,
        "labels": labels, "quality": quality, "sil_samples": sil_samples,
        "profile": profile, "centroids": centroids, "personas": personas,
        "segmented": segmented,
    }


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
def section_overview(R: dict) -> None:
    """Overview -- executive summary."""
    st.markdown('<div id="overview"></div>', unsafe_allow_html=True)
    st.subheader("Executive summary")

    top = R["personas"].iloc[0]
    cols = st.columns(4)
    tiles = [
        kpi("Customers analysed", f"{len(R['clean']):,}", "after cleaning", 1),
        kpi("Segments discovered", f"{R['k']}",
            f"elbow {R['rec']['elbow_k']} / silhouette {R['rec']['silhouette_k']}", 2),
        kpi("Silhouette score",
            f"{R['scores'].loc[R['scores']['k'] == R['k'], 'silhouette'].iloc[0]:.3f}",
            "cohesion vs separation", 3),
        kpi("Priority segment", top["persona"].split()[-1],
            f"{top['share_pct']}% of base", 4),
    ]
    for col, tile in zip(cols, tiles):
        col.markdown(tile, unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([3, 2])

    with left:
        fig = px.scatter(
            R["clean"].assign(Persona=R["segmented"]["Persona"]),
            x="Annual Income (k$)", y="Spending Score (1-100)",
            color="Persona", size="Age", size_max=16, opacity=0.85,
            title="The market in one picture: income vs spending propensity",
        )
        fig.update_traces(marker=dict(line=dict(width=0.6, color="white")))
        st.plotly_chart(styled(fig, 460), use_container_width=True)

    with right:
        share = R["personas"].sort_values("share_pct", ascending=True)
        fig = px.bar(
            share, x="share_pct", y="persona", orientation="h",
            text=share["share_pct"].map(lambda v: f"{v:.1f}%"),
            title="Share of customer base",
        )
        fig.update_traces(marker_color=ACCENT, textposition="outside",
                          hovertemplate="%{y}: %{x:.1f}%<extra></extra>")
        fig.update_layout(yaxis_title="", xaxis_title="% of customers")
        st.plotly_chart(styled(fig, 460), use_container_width=True)

    html(
        f"""
        <div class="note animate delay-2">
        <b>How to read this:</b> every customer was standardised, projected onto
        {R['n_components']} principal component(s) holding
        {R['variance_kept']:.1%} of the original variance, then grouped by
        K-Means into {R['k']} segments. The largest commercial opportunity sits with
        <b>{top['persona']}</b> ({top['size']} customers, average income
        ${top['avg_income']:.0f}k and spending score {top['avg_spend']:.0f}).
        </div>
        """
    )


def section_data(R: dict) -> None:
    """Data & Preprocessing -- cleaning, encoding, scaling."""
    st.markdown('<div id="data"></div>', unsafe_allow_html=True)
    st.subheader("Phase 1 — Scale: cleaning, encoding, standardising")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows loaded", f"{len(R['raw']):,}")
    c2.metric("Missing values", int(R["raw"].isna().sum().sum()))
    c3.metric("Duplicates removed", len(R["raw"]) - len(R["clean"]))

    tab1, tab2, tab3 = st.tabs(["Quality audit", "Distributions", "Correlation"])

    with tab1:
        render_table(R["report"])
        if R["missing_log"]:
            st.markdown(
                '<div class="note warn">Imputation applied: '
                + ", ".join(
                    f"<b>{k}</b> &rarr; {v['strategy']} ({v['filled']} cells)"
                    for k, v in R["missing_log"].items()
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="note">No missing values detected — the imputation '
                "step ran and found nothing to fill, which is the audit trail "
                "a reviewer wants to see.</div>",
                unsafe_allow_html=True,
            )
        with st.expander("Preview cleaned + encoded data"):
            render_table(R["clean"].head(20))

    with tab2:
        cols = st.columns(len(NUMERIC_COLS))
        for col, feature in zip(cols, NUMERIC_COLS):
            with col:
                fig = px.histogram(R["clean"], x=feature, nbins=20, title=feature)
                fig.update_traces(marker_color=ACCENT, marker_line_color="white",
                                  marker_line_width=1)
                fig.update_layout(showlegend=False, yaxis_title="customers")
                st.plotly_chart(styled(fig, 290), use_container_width=True)

        gender_counts = R["clean"]["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "count"]
        fig = px.pie(gender_counts, names="Gender", values="count", hole=0.62,
                     title="Gender split (encoded Male=0 / Female=1)")
        fig.update_traces(marker=dict(colors=[MINT, ACCENT],
                                      line=dict(color="white", width=3)))
        st.plotly_chart(styled(fig, 340), use_container_width=True)

    with tab3:
        corr = R["clean"][NUMERIC_COLS + ["Gender_Encoded"]].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale=[[0, "#FFFFFF"], [0.5, MINT], [1, DEEP]],
                        title="Feature correlation matrix")
        st.plotly_chart(styled(fig, 420), use_container_width=True)
        st.markdown(
            '<div class="note">Near-zero correlations mean each feature carries '
            "independent information — which is exactly why PCA cannot compress "
            "this particular dataset much, and why the same code compresses a "
            "20-column enterprise table dramatically.</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Standardised feature matrix (mean 0, sd 1)"):
        render_table(R["X_scaled"].head(10), decimals=3)


def section_pca(R: dict) -> None:
    """PCA Compression -- dimensionality reduction."""
    st.markdown('<div id="pca"></div>', unsafe_allow_html=True)
    st.subheader("Phase 2 — Compress: principal component analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Input features", len(R["features"]))
    c2.metric("Components retained", R["n_components"])
    c3.metric("Variance retained", f"{R['variance_kept']:.1%}")

    left, right = st.columns([3, 2])

    with left:
        vt = R["var_table"]
        fig = go.Figure()
        fig.add_bar(x=vt["component"], y=vt["explained_variance"],
                    name="Individual", marker_color=MINT)
        fig.add_scatter(x=vt["component"], y=vt["cumulative_variance"],
                        name="Cumulative", mode="lines+markers",
                        line=dict(color=ACCENT, width=3), marker=dict(size=9))
        fig.add_hline(y=0.95, line_dash="dash", line_color=CONTRAST,
                      annotation_text="95% threshold",
                      annotation_position="bottom right")
        fig.update_layout(title="Scree plot and cumulative explained variance",
                          yaxis_tickformat=".0%", yaxis_title="variance explained")
        st.plotly_chart(styled(fig, 430), use_container_width=True)

    with right:
        fig = px.imshow(R["loadings"], text_auto=True, aspect="auto",
                        color_continuous_scale=[[0, ACCENT], [0.5, "#FFFFFF"], [1, DEEP]],
                        title="Component loadings (what each axis means)")
        st.plotly_chart(styled(fig, 430), use_container_width=True)

    fig = px.scatter(R["Z2"], x="PC1", y="PC2", opacity=0.8,
                     title="Customers projected into 2-D PCA space (pre-clustering)")
    fig.update_traces(marker=dict(size=9, color=ACCENT,
                                  line=dict(width=0.6, color="white")))
    st.plotly_chart(styled(fig, 400), use_container_width=True)

    with st.expander("Explained variance table"):
        render_table(R["var_table"], decimals=4)

    html(
        f"""
        <div class="note animate">
        <b>Reading the loadings:</b> a component is just a weighted recipe of the
        original columns. Whichever feature has the largest absolute weight on a
        component is what that axis effectively measures — that is how an abstract
        "PC1 = 1.42" becomes a sentence management can act on.
        With a 95% threshold this dataset keeps {R['n_components']} of
        {len(R['features'])} components; lower the threshold in the sidebar to
        force harder compression and watch the trade-off.
        </div>
        """
    )


def section_optimal_k(R: dict) -> None:
    """Optimal K -- Elbow + Silhouette diagnostics."""
    st.markdown('<div id="optimal_k"></div>', unsafe_allow_html=True)
    st.subheader("Phase 3 — Cluster: proving the optimal K")

    rec, scores = R["rec"], R["scores"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Elbow method", f"K = {rec['elbow_k']}")
    c2.metric("Silhouette peak", f"K = {rec['silhouette_k']}",
              f"{rec['best_silhouette']:.3f}")
    c3.metric("Recommended", f"K = {rec['recommended_k']}")
    c4.metric("Agreement", rec["confidence"].title())

    left, right = st.columns(2)

    with left:
        fig = go.Figure()
        fig.add_scatter(x=scores["k"], y=scores["wcss"], mode="lines+markers",
                        line=dict(color=ACCENT, width=3),
                        marker=dict(size=9), name="WCSS")
        elbow_row = scores.loc[scores["k"] == rec["elbow_k"]]
        fig.add_scatter(x=elbow_row["k"], y=elbow_row["wcss"], mode="markers",
                        marker=dict(size=20, color="rgba(0,146,184,0.22)",
                                    line=dict(color=CONTRAST, width=3)),
                        name=f"Elbow (K={rec['elbow_k']})")
        fig.update_layout(title="Gatekeeper 1 — Elbow method (WCSS)",
                          xaxis_title="number of clusters (K)", yaxis_title="WCSS")
        st.plotly_chart(styled(fig, 400), use_container_width=True)

    with right:
        fig = go.Figure()
        fig.add_scatter(x=scores["k"], y=scores["silhouette"], mode="lines+markers",
                        line=dict(color=DEEP, width=3), marker=dict(size=9),
                        name="Silhouette")
        sil_row = scores.loc[scores["k"] == rec["silhouette_k"]]
        fig.add_scatter(x=sil_row["k"], y=sil_row["silhouette"], mode="markers",
                        marker=dict(size=20, color="rgba(0,150,137,0.22)",
                                    line=dict(color=ACCENT, width=3)),
                        name=f"Peak (K={rec['silhouette_k']})")
        fig.update_layout(title="Gatekeeper 2 — Silhouette score",
                          xaxis_title="number of clusters (K)",
                          yaxis_title="mean silhouette")
        st.plotly_chart(styled(fig, 400), use_container_width=True)

    # Per-sample silhouette plot for the chosen K.
    sil = R["sil_samples"].copy()
    sil["order"] = range(len(sil))
    sil["Cluster"] = sil["cluster"].astype(str)
    fig = px.bar(sil, x="order", y="silhouette", color="Cluster",
                 title=f"Per-customer silhouette at K = {R['k']}",
                 labels={"order": "customers (sorted)"})
    fig.add_hline(y=sil["silhouette"].mean(), line_dash="dash",
                  line_color=CONTRAST, annotation_text="average")
    fig.update_layout(xaxis_showticklabels=False, legend_title="cluster")
    st.plotly_chart(styled(fig, 380), use_container_width=True)

    with st.expander("Full diagnostic table (all K values)"):
        # The chosen K is flagged in a column instead of a Styler highlight:
        # Styler output cannot carry the .seg-table class the CSS targets.
        table = scores.copy()
        table.insert(1, "chosen", np.where(table["k"] == R["k"], "◄ selected", ""))
        render_table(table, decimals=4)

    verdict = {
        "high": "Both gatekeepers agree — this K is well supported.",
        "medium": "The two gatekeepers land one step apart; the silhouette wins "
                  "because WCSS always falls as K grows.",
        "review": "The gatekeepers disagree by more than one step. The silhouette "
                  "is used, but compare both K values before presenting.",
    }[rec["confidence"]]
    st.markdown(f'<div class="note">{verdict}</div>', unsafe_allow_html=True)


def section_clusters(R: dict) -> None:
    """Cluster Explorer -- visualise the clusters."""
    st.markdown('<div id="clusters"></div>', unsafe_allow_html=True)
    st.subheader("Cluster explorer")

    plot_df = R["Z2"].copy()
    plot_df["Persona"] = R["segmented"]["Persona"].values
    plot_df["Cluster"] = R["labels"].astype(str)

    tab2d, tab3d, tabreal = st.tabs(["2-D PCA space", "3-D PCA space",
                                     "Original business units"])

    with tab2d:
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Persona", opacity=0.88,
                         title="Segments in principal component space")
        fig.update_traces(marker=dict(size=11, line=dict(width=0.8, color="white")))
        st.plotly_chart(styled(fig, 520), use_container_width=True)

    with tab3d:
        if R["Z3"].shape[1] >= 3:
            df3 = R["Z3"].copy()
            df3["Persona"] = R["segmented"]["Persona"].values
            fig = px.scatter_3d(df3, x="PC1", y="PC2", z="PC3", color="Persona",
                                opacity=0.85,
                                title="Segments in 3-D principal component space")
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(styled(fig, 620), use_container_width=True)
        else:
            st.info("A 3-D view needs at least three features. Add another "
                    "feature in the sidebar to enable it.")

    with tabreal:
        real = R["clean"].copy()
        real["Persona"] = R["segmented"]["Persona"].values
        fig = px.scatter(real, x="Annual Income (k$)", y="Spending Score (1-100)",
                         color="Persona", symbol="Gender", opacity=0.88,
                         hover_data=["Age", "CustomerID"],
                         title="Segments plotted on the axes the business uses")
        fig.update_traces(marker=dict(size=11, line=dict(width=0.7, color="white")))
        centroids = R["centroids"]
        fig.add_scatter(
            x=centroids["Annual Income (k$)"], y=centroids["Spending Score (1-100)"],
            mode="markers+text", name="Centroids",
            text=[f"C{int(c)}" for c in centroids["cluster"]],
            textposition="top center",
            marker=dict(symbol="x", size=16, color=DEEP, line=dict(width=2)),
        )
        st.plotly_chart(styled(fig, 540), use_container_width=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        q = R["quality"].copy()
        q["cluster"] = q["cluster"].astype(str)
        fig = px.bar(q, x="cluster", y="mean_silhouette",
                     text=q["mean_silhouette"].round(2),
                     title="Cohesion by cluster")
        fig.update_traces(marker_color=ACCENT, textposition="outside")
        st.plotly_chart(styled(fig, 360), use_container_width=True)
    with c2:
        st.markdown("**Reconstructed centroids in real-world units**")
        render_table(R["centroids"], decimals=1)
        st.markdown(
            '<div class="note">These rows are the mathematical centre of each '
            "cluster mapped back through the inverse PCA and inverse scaler — "
            "the Phase 4 translation step.</div>",
            unsafe_allow_html=True,
        )


def persona_card(row: pd.Series, max_index: float) -> str:
    """
    Build the HTML for one persona card as a SINGLE flat line.

    Multi-line, indented HTML is parsed by Markdown as a code block, which is
    why an indented card renders as visible source text instead of a card.
    Every fragment below is joined without newlines to avoid that entirely.

    Optional columns (emoji, pct_female) are guarded with `in row.index` rather
    than accessed directly: `row["emoji"]` raises KeyError the moment that
    column is dropped upstream in build_personas(), whereas this version simply
    renders the card without the badge.
    """
    actions = "".join(f"<li>{a}</li>" for a in row["actions"])
    female = (
        '<div class="stat"><div class="s-label">Female</div>'
        f'<div class="s-value">{row["pct_female"]:.0f}%</div></div>'
        if "pct_female" in row.index and pd.notna(row.get("pct_female")) else ""
    )
    width = row["opportunity_index"] / max_index * 100

    # The emoji badge was removed from the persona schema -- render it only if a
    # future build_personas() puts the column back.
    emoji = (
        f'<div class="emoji">{row["emoji"]}</div>'
        if "emoji" in row.index and pd.notna(row.get("emoji")) else ""
    )

    head = (
        '<div class="persona-head">'
        f"{emoji}"
        f'<div><div class="title">{row["persona"]}</div>'
        f'<div class="tag">{row["tagline"]}</div></div>'
        f'<div class="pill">{row["priority"]}</div>'
        "</div>"
    )
    stats = (
        '<div class="stat-row">'
        f'<div class="stat"><div class="s-label">Size</div>'
        f'<div class="s-value">{row["size"]}</div></div>'
        f'<div class="stat"><div class="s-label">Age</div>'
        f'<div class="s-value">{row["avg_age"]:.0f}</div></div>'
        f'<div class="stat"><div class="s-label">Income</div>'
        f'<div class="s-value">${row["avg_income"]:.0f}k</div></div>'
        f'<div class="stat"><div class="s-label">Spend</div>'
        f'<div class="s-value">{row["avg_spend"]:.0f}</div></div>'
        f"{female}</div>"
    )
    meter = (
        '<div class="s-label" style="font-size:.66rem;letter-spacing:.07em;'
        'text-transform:uppercase;font-weight:600;">'
        f'Opportunity index {row["opportunity_index"]:.0f}/100</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>'
    )
    return (
        '<div class="persona animate">'
        f'{head}<div class="persona-body">{stats}{meter}<ul>{actions}</ul></div>'
        "</div>"
    )


def section_personas(R: dict) -> None:
    """Business Personas -- persona cards + recommended actions."""
    st.markdown('<div id="personas"></div>', unsafe_allow_html=True)
    st.subheader("Phase 4 — Translate: the strategic persona matrix")

    personas = R["personas"]
    cards = "".join(persona_card(row, personas["opportunity_index"].max())
                    for _, row in personas.iterrows())
    html(f'<div class="persona-grid">{cards}</div>')

    st.write("")
    left, right = st.columns(2)

    with left:
        fig = px.bar(personas.sort_values("opportunity_index"),
                     x="opportunity_index", y="persona", orientation="h",
                     title="Commercial priority (size x income x spend, rebased to 100)")
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(yaxis_title="", xaxis_title="opportunity index")
        st.plotly_chart(styled(fig, 420), use_container_width=True)

    with right:
        # Radar comparison on a common 0-100 scale.
        radar = personas.copy()
        axes = ["avg_age", "avg_income", "avg_spend"]
        labels = ["Age", "Income", "Spending"]
        for col in axes:
            span = radar[col].max() - radar[col].min()
            radar[col + "_n"] = (
                50.0 if span == 0
                else (radar[col] - radar[col].min()) / span * 100
            )
        fig = go.Figure()
        for _, row in radar.iterrows():
            values = [row[c + "_n"] for c in axes]
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]], theta=labels + [labels[0]],
                fill="toself", name=row["persona"], opacity=0.55,
            ))
        fig.update_layout(title="Persona fingerprints (relative scale)",
                          polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        st.plotly_chart(styled(fig, 420), use_container_width=True)

    st.markdown("**Full persona table**")
    # Only ask for columns that actually exist, so dropping a field upstream can
    # never break this table the way it broke the cards.
    display_cols = ["cluster", "persona", "priority", "size", "share_pct",
                    "avg_age", "avg_income", "avg_spend", "opportunity_index"]
    if "pct_female" in personas.columns:
        display_cols.insert(-1, "pct_female")
    display_cols = [c for c in display_cols if c in personas.columns]
    render_table(personas[display_cols], decimals=1)

    c1, c2 = st.columns(2)
    c1.download_button(
        "Download segmented customers (CSV)",
        R["segmented"].to_csv(index=False).encode("utf-8"),
        file_name="segmented_customers.csv", mime="text/csv",
    )
    c2.download_button(
        "Download persona matrix (CSV)",
        personas.drop(columns=["actions"], errors="ignore")
        .to_csv(index=False).encode("utf-8"),
        file_name="personas.csv", mime="text/csv",
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="CustomerSegmentationDashboard",
                       page_icon="\u25C9", layout="wide",
                       initial_sidebar_state="expanded")
    inject_css()
    inject_overrides()   # header title + hamburger toggle

    # ---------------- sidebar: navigation + model controls ----------------
    with st.sidebar:
        st.markdown("### Customer Segmentation")
        st.caption("PCA + K-Means segmentation dashboard")

        labels = [label for _, label, _ in SECTIONS]
        captions = [desc for _, _, desc in SECTIONS]
        # captions put the purpose of each tab directly under its name
        active = st.radio("Navigate", labels, captions=captions, index=0,
                          label_visibility="collapsed")

        st.divider()
        st.markdown("#### Data source")
        uploaded = st.file_uploader("Customer CSV", type=["csv"])
        if uploaded is not None:
            csv_bytes = uploaded.getvalue()
        else:
            try:
                with open("data/Mall_Customers.csv", "rb") as fh:
                    csv_bytes = fh.read()
            except FileNotFoundError:
                st.error("Upload a CSV to begin — no default dataset found at "
                         "data/Mall_Customers.csv.")
                st.stop()

        st.divider()
        st.markdown("#### Model controls")
        available = NUMERIC_COLS + ["Gender_Encoded"]
        features = st.multiselect("Clustering features", available,
                                  default=list(DEFAULT_FEATURES))
        if len(features) < 2:
            st.warning("Select at least two features.")
            st.stop()

        variance = st.slider("PCA variance to retain", 0.60, 0.99, 0.95, 0.01)
        k_min, k_max = st.slider("K search range", 2, 12, (2, 10))
        auto_k = st.checkbox("Auto-select K from diagnostics", value=True)
        k_override = None
        if not auto_k:
            k_override = st.slider("Manual K", k_min, k_max, min(5, k_max))

        st.divider()
        st.caption("DecodeLabs Industrial Training Kit — Project 3")

    render_header(active)

    with st.spinner("Running pipeline: scale -> compress -> cluster -> translate"):
        R = run_analysis(csv_bytes, tuple(features), variance, k_min, k_max,
                         k_override)

    # Router keeps navigation and rendering decoupled -- adding a section means
    # adding one entry to SECTIONS and one here.
    router = {
        "Overview": section_overview,
        "Data & Preprocessing": section_data,
        "PCA Compression": section_pca,
        "Optimal K": section_optimal_k,
        "Cluster Explorer": section_clusters,
        "Business Personas": section_personas,
    }
    router[active](R)


if __name__ == "__main__":
    main()