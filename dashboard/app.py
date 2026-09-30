"""
dashboard/app.py
================
ApexBank | Customer Segmentation BI Platform
Academic DWDM Project: Customer Segmentation in Banking Using Clustering and Data Mining
Dataset: Default of Credit Card Clients (UCI Machine Learning Repository, ID: 350)
Official Citation: Yeh, I. (2009). DOI: 10.24432/C55S3H
Author: DWDM Project Team
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Safe Plotly Import with full styling capabilities
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Streamlit Page Configuration
st.set_page_config(
    page_title="ApexBank | Customer Segmentation BI Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# THEME ENGINE SETUP (Dark Luxury, Light Professional, System Default)
# ============================================================================
st.sidebar.markdown("""
<div style="padding: 0.2rem 0 0.5rem 0;">
    <h2 style="font-weight: 800; margin: 0; font-size: 1.45rem;">🏛️ ApexBank BI</h2>
    <p style="color: #3b82f6; font-size: 0.8rem; margin: 0; font-weight: 600;">Customer Segmentation BI Platform</p>
    <p style="font-size: 0.72rem; margin: 2px 0 0 0; opacity: 0.75;">UCI Credit Card Clients (Yeh, 2009)</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🎨 System Appearance")
theme_mode = st.sidebar.selectbox(
    "Interface Theme:",
    ["🌓 System Default (Auto-Adapt)", "🌙 Dark Luxury", "☀️ Light Professional"],
    index=0
)

# Determine Plotly chart palette
if theme_mode == "☀️ Light Professional":
    is_dark = False
elif theme_mode == "🌙 Dark Luxury":
    is_dark = True
else:
    # In System Default mode, allow chart palette toggle
    chart_light = st.sidebar.toggle("☀️ Chart Palette: Light Mode", value=False)
    is_dark = not chart_light

# Dynamic Theme CSS Variables Injection
if theme_mode == "☀️ Light Professional":
    theme_vars = """
    :root {
        --bg-app: #f8fafc;
        --bg-card: #ffffff;
        --bg-card-hover: #f1f5f9;
        --text-main: #0f172a;
        --text-muted: #475569;
        --border-color: #e2e8f0;
        --table-bg: #ffffff;
        --table-th-bg: #f1f5f9;
        --table-th-text: #0f172a;
        --table-td-text: #1e293b;
        --table-border: #e2e8f0;
        --table-hover: #f8fafc;
        --status-bar-bg: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
        --hero-title-gradient: linear-gradient(135deg, #0f172a 0%, #1e40af 50%, #2563eb 100%);
        --card-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
        --alert-bg: #eff6ff;
        --alert-text: #1e3a8a;
        --alert-warning-bg: #fffbeb;
        --alert-warning-text: #92400e;
        --status-border: #cbd5e1;
        --takeaway-bg: #f0fdf4;
        --takeaway-border: #86efac;
        --takeaway-text: #166534;
    }
    """
elif theme_mode == "🌙 Dark Luxury":
    theme_vars = """
    :root {
        --bg-app: #090d16;
        --bg-card: #0f172a;
        --bg-card-hover: #1e293b;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --border-color: #1e293b;
        --table-bg: #0b1120;
        --table-th-bg: #0f172a;
        --table-th-text: #93c5fd;
        --table-td-text: #cbd5e1;
        --table-border: #1e293b;
        --table-hover: #1e293b;
        --status-bar-bg: linear-gradient(90deg, #090d16 0%, #0f172a 50%, #1e1b4b 100%);
        --hero-title-gradient: linear-gradient(135deg, #f8fafc 0%, #93c5fd 50%, #3b82f6 100%);
        --card-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        --alert-bg: rgba(30, 41, 59, 0.7);
        --alert-text: #cbd5e1;
        --alert-warning-bg: rgba(41, 37, 36, 0.7);
        --alert-warning-text: #fef3c7;
        --status-border: #1e293b;
        --takeaway-bg: rgba(6, 78, 59, 0.25);
        --takeaway-border: #059669;
        --takeaway-text: #6ee7b7;
    }
    """
else:
    # System Default (Auto-Adapt): Dark baseline with @media (prefers-color-scheme: light) override
    theme_vars = """
    :root {
        --bg-app: #090d16;
        --bg-card: #0f172a;
        --bg-card-hover: #1e293b;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --border-color: #1e293b;
        --table-bg: #0b1120;
        --table-th-bg: #0f172a;
        --table-th-text: #93c5fd;
        --table-td-text: #cbd5e1;
        --table-border: #1e293b;
        --table-hover: #1e293b;
        --status-bar-bg: linear-gradient(90deg, #090d16 0%, #0f172a 50%, #1e1b4b 100%);
        --hero-title-gradient: linear-gradient(135deg, #f8fafc 0%, #93c5fd 50%, #3b82f6 100%);
        --card-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        --alert-bg: rgba(30, 41, 59, 0.7);
        --alert-text: #cbd5e1;
        --alert-warning-bg: rgba(41, 37, 36, 0.7);
        --alert-warning-text: #fef3c7;
        --status-border: #1e293b;
        --takeaway-bg: rgba(6, 78, 59, 0.25);
        --takeaway-border: #059669;
        --takeaway-text: #6ee7b7;
    }
    @media (prefers-color-scheme: light) {
        :root {
            --bg-app: #f8fafc;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --text-main: #0f172a;
            --text-muted: #475569;
            --border-color: #e2e8f0;
            --table-bg: #ffffff;
            --table-th-bg: #f1f5f9;
            --table-th-text: #0f172a;
            --table-td-text: #1e293b;
            --table-border: #e2e8f0;
            --table-hover: #f8fafc;
            --status-bar-bg: linear-gradient(90deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
            --hero-title-gradient: linear-gradient(135deg, #0f172a 0%, #1e40af 50%, #2563eb 100%);
            --card-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
            --alert-bg: #eff6ff;
            --alert-text: #1e3a8a;
            --alert-warning-bg: #fffbeb;
            --alert-warning-text: #92400e;
            --status-border: #cbd5e1;
            --takeaway-bg: #f0fdf4;
            --takeaway-border: #86efac;
            --takeaway-text: #166534;
        }
    }
    """

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    {theme_vars}
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Layout Containers */
    [data-testid="stAppViewContainer"], .main {{
        background-color: var(--bg-app) !important;
        color: var(--text-main) !important;
    }}
    [data-testid="stHeader"] {{
        background-color: transparent !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border-color) !important;
    }}
    [data-testid="stSidebar"] * {{
        color: var(--text-main);
    }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{
        color: var(--text-muted) !important;
    }}
    
    /* Status Bar */
    .status-bar {{
        background: var(--status-bar-bg) !important;
        color: var(--text-muted) !important;
        padding: 0.75rem 1.4rem;
        border-radius: 12px;
        font-size: 0.84rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        border: 1px solid var(--status-border) !important;
        box-shadow: var(--card-shadow) !important;
    }}
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #10b981;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.12);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    
    /* Hero Headers */
    .hero-title {{
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: var(--hero-title-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0.2rem;
    }}
    .hero-sub {{
        font-size: 0.95rem;
        color: var(--text-muted) !important;
        margin-bottom: 1.2rem;
        font-weight: 500;
    }}
    
    /* KPI Cards */
    .kpi-card {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px;
        padding: 1.15rem 1.3rem;
        box-shadow: var(--card-shadow) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 0.8rem;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.15);
    }}
    .kpi-blue {{ border-left: 4px solid #3b82f6 !important; }}
    .kpi-green {{ border-left: 4px solid #10b981 !important; }}
    .kpi-amber {{ border-left: 4px solid #f59e0b !important; }}
    .kpi-purple {{ border-left: 4px solid #8b5cf6 !important; }}
    .kpi-rose {{ border-left: 4px solid #ec4899 !important; }}
    .kpi-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }}
    .kpi-label {{
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted) !important;
    }}
    .kpi-num {{
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--text-main) !important;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }}
    .kpi-delta {{
        font-size: 0.76rem;
        font-weight: 600;
        margin-top: 0.35rem;
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    .delta-up {{ color: #10b981; }}
    .delta-neutral {{ color: #3b82f6; }}
    
    /* Content Containers */
    .dark-card {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 14px;
        padding: 1.3rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--card-shadow) !important;
    }}
    .dark-card h3, .dark-card h4 {{
        color: var(--text-main) !important;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 0.6rem;
    }}
    .dark-card p, .dark-card li, .dark-card strong {{
        color: var(--text-muted) !important;
    }}
    
    /* Styled HTML Tables */
    .styled-table-wrap {{
        border-radius: 12px;
        border: 1px solid var(--border-color) !important;
        overflow-x: auto;
        box-shadow: var(--card-shadow) !important;
        margin: 1rem 0;
        background: var(--table-bg) !important;
    }}
    .fintech-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        background-color: var(--table-bg) !important;
        color: var(--table-td-text) !important;
    }}
    .fintech-table th {{
        background: var(--table-th-bg) !important;
        color: var(--table-th-text) !important;
        font-weight: 600;
        padding: 11px 15px;
        text-align: left;
        letter-spacing: 0.03em;
        white-space: nowrap;
        border-bottom: 2px solid var(--border-color) !important;
    }}
    .fintech-table td {{
        padding: 10px 15px;
        border-bottom: 1px solid var(--table-border) !important;
        color: var(--table-td-text) !important;
        white-space: nowrap;
    }}
    .fintech-table tr:hover {{
        background-color: var(--table-hover) !important;
    }}
    
    /* Custom Badges & Alert Boxes */
    .alert-box {{
        background: var(--alert-bg) !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
        font-size: 0.86rem;
        color: var(--alert-text) !important;
    }}
    .alert-box-warning {{
        background: var(--alert-warning-bg) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0;
        font-size: 0.86rem;
        color: var(--alert-warning-text) !important;
    }}
    .plain-takeaway {{
        background: var(--takeaway-bg) !important;
        border-left: 4px solid var(--takeaway-border) !important;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.6rem 0 1rem 0;
        font-size: 0.85rem;
        color: var(--takeaway-text) !important;
        font-weight: 500;
    }}
</style>
""", unsafe_allow_html=True)


# Pure Pandas HTML Table Renderer (Guarantees zero Windows PyArrow DLL crashes)
def render_fintech_table(df, max_rows=15, format_cols=None):
    if df is None or df.empty:
        st.info("No records available to display.")
        return
        
    display_df = df.head(max_rows).copy()
    if format_cols:
        for col, fmt in format_cols.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else "")
                
    html = display_df.to_html(classes='fintech-table', escape=False, index=False)
    st.markdown(f'<div class="styled-table-wrap">{html}</div>', unsafe_allow_html=True)
    if len(df) > max_rows:
        st.caption(f"Displaying first {max_rows} of {len(df):,} total records.")


# Dynamic Theme Plotly Layout Applier
def apply_plotly_layout(fig, height=430):
    if is_dark:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#090d16",
            plot_bgcolor="#0f172a",
            font=dict(family="Plus Jakarta Sans", color="#e2e8f0", size=11),
            height=height,
            margin=dict(l=40, r=40, t=50, b=80),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(15, 23, 42, 0.7)",
                bordercolor="#1e293b",
                borderwidth=1
            )
        )
    else:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            font=dict(family="Plus Jakarta Sans", color="#0f172a", size=11),
            height=height,
            margin=dict(l=40, r=40, t=50, b=80),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="#e2e8f0",
                borderwidth=1
            )
        )
    return fig

apply_dark_plotly_layout = apply_plotly_layout


# Cached Data Loading (Strictly from actual computed artifacts)
@st.cache_data
def load_all_artifacts():
    clustered_path = 'data/processed/credit_card_clustered.csv'
    df_clustered = pd.read_csv(clustered_path) if os.path.exists(clustered_path) else pd.DataFrame()
    
    raw_path = 'data/raw/default_of_credit_card_clients.csv'
    df_raw = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame()
    
    dq_path = 'outputs/tables/data_quality_summary.csv'
    df_dq = pd.read_csv(dq_path) if os.path.exists(dq_path) else pd.DataFrame()
    
    fe_path = 'outputs/tables/feature_engineering_summary.csv'
    df_fe = pd.read_csv(fe_path) if os.path.exists(fe_path) else pd.DataFrame()
    
    k_eval_path = 'outputs/tables/k_evaluation_metrics.csv'
    df_keval = pd.read_csv(k_eval_path) if os.path.exists(k_eval_path) else pd.DataFrame()
    
    profiles_path = 'outputs/tables/cluster_profiles.csv'
    df_profiles = pd.read_csv(profiles_path) if os.path.exists(profiles_path) else pd.DataFrame()
    
    post_y_path = 'outputs/tables/post_clustering_default_analysis.csv'
    df_post_y = pd.read_csv(post_y_path) if os.path.exists(post_y_path) else pd.DataFrame()
    
    algo_path = 'outputs/tables/algorithm_comparison.csv'
    df_algo = pd.read_csv(algo_path) if os.path.exists(algo_path) else pd.DataFrame()
    
    meta_path = 'outputs/tables/clustering_metadata.csv'
    df_meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else pd.DataFrame()
    
    pca_loadings_path = 'outputs/tables/pca_loadings.csv'
    df_pca_loadings = pd.read_csv(pca_loadings_path) if os.path.exists(pca_loadings_path) else pd.DataFrame()
    
    centers_path = 'outputs/tables/cluster_centers.csv'
    df_centers = pd.read_csv(centers_path) if os.path.exists(centers_path) else pd.DataFrame()
    
    return {
        'clustered': df_clustered,
        'raw': df_raw,
        'dq': df_dq,
        'fe': df_fe,
        'keval': df_keval,
        'profiles': df_profiles,
        'post_y': df_post_y,
        'algo': df_algo,
        'meta': df_meta,
        'pca_loadings': df_pca_loadings,
        'centers': df_centers
    }


@st.cache_resource
def load_ml_models():
    km = joblib.load('outputs/models/kmeans_model.joblib') if os.path.exists('outputs/models/kmeans_model.joblib') else None
    sc = joblib.load('outputs/models/scaler.joblib') if os.path.exists('outputs/models/scaler.joblib') else None
    pca = joblib.load('outputs/models/pca_model.joblib') if os.path.exists('outputs/models/pca_model.joblib') else None
    return km, sc, pca


data = load_all_artifacts()
df_clustered = data['clustered']
df_raw = data['raw']
df_dq = data['dq']
df_fe = data['fe']
df_keval = data['keval']
df_profiles = data['profiles']
df_post_y = data['post_y']
df_algo = data['algo']
df_meta = data['meta']
df_pca_loadings = data['pca_loadings']
df_centers = data['centers']
kmeans_model, scaler_model, pca_model = load_ml_models()


# ============================================================================
# USER-FRIENDLY GROUPED SIDEBAR NAVIGATION (5 High-Level Hubs + Predictor)
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📑 Platform Modules")

nav_selection = st.sidebar.radio(
    "Choose Module:",
    [
        "⚡ 1. Executive Dashboard & Personas",
        "📊 2. Data Studio & Preprocessing",
        "🎯 3. Machine Learning & Diagnostics",
        "🏛️ 4. DWDM Architecture & Warehouse",
        "🎓 5. Academic Research & Viva Center",
        "🔮 6. Real-Time Customer Predictor"
    ],
    index=0
)

st.sidebar.markdown("---")

# Global Filter Slicer
if not df_clustered.empty:
    st.sidebar.markdown("### 🎛️ Interactive Data Slicer")
    
    available_personas = ["All Segments"] + sorted(list(df_clustered['Cluster_Persona'].unique()))
    selected_persona = st.sidebar.selectbox("Filter by Persona:", available_personas)
    
    min_lim = int(df_clustered['LIMIT_BAL'].min())
    max_lim = int(df_clustered['LIMIT_BAL'].max())
    credit_range = st.sidebar.slider(
        "Credit Limit Range (NT$):",
        min_value=min_lim,
        max_value=max_lim,
        value=(min_lim, max_lim),
        step=50000,
        format="NT$ %d"
    )
    
    min_age = int(df_clustered['AGE'].min())
    max_age = int(df_clustered['AGE'].max())
    age_range = st.sidebar.slider(
        "Age Range (Years):",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        step=1
    )
    
    max_delay_options = ["Any Delay", "Prompt Payers (<= 0 mos)", "Delayed Accounts (> 0 mos)", "Chronic Delinquency (>= 2 mos)"]
    delay_choice = st.sidebar.selectbox("Repayment Delay Filter:", max_delay_options)
    
    default_options = ["All Customers", "Non-Default Only (Y=0)", "Defaulted Accounts Only (Y=1)"]
    default_choice = st.sidebar.selectbox("Post-Hoc Default Filter (Y):", default_options)
    
    # Apply filtering
    active_df = df_clustered.copy()
    if selected_persona != "All Segments":
        active_df = active_df[active_df['Cluster_Persona'] == selected_persona]
    active_df = active_df[(active_df['LIMIT_BAL'] >= credit_range[0]) & (active_df['LIMIT_BAL'] <= credit_range[1])]
    active_df = active_df[(active_df['AGE'] >= age_range[0]) & (active_df['AGE'] <= age_range[1])]
    
    if delay_choice == "Prompt Payers (<= 0 mos)":
        active_df = active_df[active_df['Max_Repayment_Delay'] <= 0]
    elif delay_choice == "Delayed Accounts (> 0 mos)":
        active_df = active_df[active_df['Max_Repayment_Delay'] > 0]
    elif delay_choice == "Chronic Delinquency (>= 2 mos)":
        active_df = active_df[active_df['Max_Repayment_Delay'] >= 2]
        
    if default_choice == "Non-Default Only (Y=0)":
        active_df = active_df[active_df['Y'] == 0]
    elif default_choice == "Defaulted Accounts Only (Y=1)":
        active_df = active_df[active_df['Y'] == 1]
        
    # Demographic / Financial filtered population (independent of single persona selection)
    pop_filter_df = df_clustered.copy()
    pop_filter_df = pop_filter_df[(pop_filter_df['LIMIT_BAL'] >= credit_range[0]) & (pop_filter_df['LIMIT_BAL'] <= credit_range[1])]
    pop_filter_df = pop_filter_df[(pop_filter_df['AGE'] >= age_range[0]) & (pop_filter_df['AGE'] <= age_range[1])]
    if delay_choice == "Prompt Payers (<= 0 mos)":
        pop_filter_df = pop_filter_df[pop_filter_df['Max_Repayment_Delay'] <= 0]
    elif delay_choice == "Delayed Accounts (> 0 mos)":
        pop_filter_df = pop_filter_df[pop_filter_df['Max_Repayment_Delay'] > 0]
    elif delay_choice == "Chronic Delinquency (>= 2 mos)":
        pop_filter_df = pop_filter_df[pop_filter_df['Max_Repayment_Delay'] >= 2]
    if default_choice == "Non-Default Only (Y=0)":
        pop_filter_df = pop_filter_df[pop_filter_df['Y'] == 0]
    elif default_choice == "Defaulted Accounts Only (Y=1)":
        pop_filter_df = pop_filter_df[pop_filter_df['Y'] == 1]
else:
    active_df = pd.DataFrame()
    pop_filter_df = pd.DataFrame()


# ============================================================================
# TOP LIVE STATUS BAR
# ============================================================================
total_records = len(df_clustered) if not df_clustered.empty else 30000
opt_k = int(df_meta['Optimal_K'].values[0]) if not df_meta.empty else 4
opt_sil = float(df_meta['Silhouette_Score'].values[0]) if not df_meta.empty else 0.2443

st.markdown(f"""
<div class="status-bar">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span class="status-pill">● LIVE UCI DATASET (ID: 350)</span>
        <span>Repository: <strong>Default of Credit Card Clients (Yeh, 2009)</strong></span>
    </div>
    <div>
        <span>Active Records Filtered: <strong style="color: #3b82f6;">{len(active_df):,}</strong> / {total_records:,}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# HUB 1: EXECUTIVE DASHBOARD & PERSONAS
# ============================================================================
if "1. Executive" in nav_selection:
    tab_exec, tab_personas, tab_compare, tab_explorer, tab_pca = st.tabs([
        "⚡ Executive KPIs & Brief",
        "👥 Discovered Customer Personas",
        "⚖️ Compare Clusters",
        "🔍 Customer Account Explorer",
        "🌐 2D PCA Cluster Map"
    ])
    
    with tab_exec:
        st.markdown('<div class="hero-title">Executive Intelligence Center</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Translating 30,000 Multi-Month Credit Histories into Strategic Banking Segments</div>', unsafe_allow_html=True)
        
        # Determine metrics dynamically
        curr_active_count = len(active_df) if not active_df.empty else 0
        overall_default_rate = float(df_meta['Overall_Default_Rate'].values[0]) if not df_meta.empty and 'Overall_Default_Rate' in df_meta.columns else (float(df_raw['Y'].mean() * 100) if not df_raw.empty else 22.12)
        active_default_rate = float(active_df['Y'].mean() * 100) if not active_df.empty else 0.0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-header"><span class="kpi-label">Active / Total Customers</span><span>👥</span></div>
            <div class="kpi-num">{curr_active_count:,}</div>
            <div class="kpi-delta delta-up">Total Population: {total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-header"><span class="kpi-label">Features Analyzed</span><span>📊</span></div>
            <div class="kpi-num">{df_raw.shape[1] if not df_raw.empty else 25}</div>
            <div class="kpi-delta delta-neutral">8 Clustering Inputs + Target (Y)</div>
        </div>
        """, unsafe_allow_html=True)
        c3.markdown(f"""
        <div class="kpi-card kpi-amber">
            <div class="kpi-header"><span class="kpi-label">Observed Default Rate</span><span>🛡️</span></div>
            <div class="kpi-num">{active_default_rate:.2f}%</div>
            <div class="kpi-delta delta-neutral">Population Baseline: {overall_default_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        c4.markdown(f"""
        <div class="kpi-card kpi-purple">
            <div class="kpi-header"><span class="kpi-label">Optimal Segments (K)</span><span>🎯</span></div>
            <div class="kpi-num">{opt_k}</div>
            <div class="kpi-delta delta-neutral">Silhouette: {opt_sil:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Dynamic Takeaway summary
        if not df_profiles.empty:
            largest_row = df_profiles.loc[df_profiles['Customers'].idxmax()]
            highest_risk_row = df_profiles.loc[df_profiles['Post_Default_Rate'].idxmax()]
            lowest_risk_row = df_profiles.loc[df_profiles['Post_Default_Rate'].idxmin()]
            
            st.markdown(f"""
            <div class="plain-takeaway">
                💡 <strong>Executive Takeaway:</strong> Analyzed <strong>{total_records:,} accounts</strong> across 6 months of billing and repayment data. 
                The mathematical K-Means optimization determined <strong>{opt_k} distinct behavioral segments</strong>.
                The largest segment is <strong>{largest_row['Cluster_Persona']}</strong> ({largest_row['Percentage']:.1f}% share),
                the safest cohort is <strong>{lowest_risk_row['Cluster_Persona']}</strong> ({lowest_risk_row['Post_Default_Rate']:.2f}% default),
                and the highest credit risk cohort is <strong>{highest_risk_row['Cluster_Persona']}</strong> with an alarming 
                <strong>{highest_risk_row['Post_Default_Rate']:.2f}% default rate</strong>.
            </div>
            """, unsafe_allow_html=True)
            
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("### 📊 Customer Volume per Segment")
            if not df_profiles.empty and HAS_PLOTLY:
                all_personas = list(df_profiles.sort_values(by='Cluster')['Cluster_Persona'].unique())
                source_df = pop_filter_df if not pop_filter_df.empty else df_clustered
                counts_dict = source_df['Cluster_Persona'].value_counts().to_dict() if not source_df.empty else {}
                total_in_pop = sum(counts_dict.values())
                
                chart_data = []
                for p in all_personas:
                    c_count = counts_dict.get(p, 0)
                    c_pct = (c_count / total_in_pop * 100) if total_in_pop > 0 else 0.0
                    chart_data.append({
                        'Cluster_Persona': p,
                        'Customers': c_count,
                        'Percentage': round(c_pct, 2)
                    })
                chart_df = pd.DataFrame(chart_data)
                
                fig_bar = px.bar(
                    chart_df,
                    x='Cluster_Persona',
                    y='Customers',
                    color='Cluster_Persona',
                    text='Percentage',
                    color_discrete_sequence=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#f59e0b']
                )
                fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_bar.update_layout(xaxis_title="Customer Segment", yaxis_title="Number of Customers")
                st.plotly_chart(apply_plotly_layout(fig_bar, height=360), width='stretch')
        with col_c2:
            st.markdown("### 🥧 Proportional Customer Base Breakdown")
            if not df_profiles.empty and HAS_PLOTLY:
                fig_donut = px.pie(
                    chart_df,
                    names='Cluster_Persona',
                    values='Customers',
                    hole=0.45,
                    color='Cluster_Persona',
                    color_discrete_sequence=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#f59e0b']
                )
                fig_donut.update_traces(textinfo='percent+label', textposition='inside')
                st.plotly_chart(apply_plotly_layout(fig_donut, height=360), width='stretch')

    with tab_personas:
        st.markdown('<div class="hero-title">Discovered Customer Segment Profiles</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Empirical Statistical Signatures of Discovered Behavioral Segments</div>', unsafe_allow_html=True)
        
        # Interactive Sorting Controls
        st.markdown("##### 🔀 Sort Segment Profiles Table:")
        sort_col1, sort_col2 = st.columns([3, 1])
        with sort_col1:
            sort_metric = st.selectbox(
                "Sort Clusters By Metric:",
                options=[
                    "Cluster Index (Default)",
                    "Customer Count (Size)",
                    "Average Credit Limit",
                    "Average Monthly Payment",
                    "Average Monthly Bill",
                    "Maximum Overdue Delay",
                    "Post-Hoc Default Rate"
                ]
            )
        with sort_col2:
            sort_ascending = st.checkbox("Ascending Order", value=False)
            
        display_profiles = df_profiles.copy()
        if sort_metric == "Customer Count (Size)":
            display_profiles = display_profiles.sort_values(by='Customers', ascending=sort_ascending)
        elif sort_metric == "Average Credit Limit":
            display_profiles = display_profiles.sort_values(by='Avg_Credit_Limit', ascending=sort_ascending)
        elif sort_metric == "Average Monthly Payment":
            display_profiles = display_profiles.sort_values(by='Avg_Payment', ascending=sort_ascending)
        elif sort_metric == "Average Monthly Bill":
            display_profiles = display_profiles.sort_values(by='Avg_Bill', ascending=sort_ascending)
        elif sort_metric == "Maximum Overdue Delay":
            display_profiles = display_profiles.sort_values(by='Avg_Max_Delay', ascending=sort_ascending)
        elif sort_metric == "Post-Hoc Default Rate":
            display_profiles = display_profiles.sort_values(by='Post_Default_Rate', ascending=sort_ascending)
        else:
            display_profiles = display_profiles.sort_values(by='Cluster', ascending=True)

        # Standard Cluster Profiles Table
        render_fintech_table(display_profiles, max_rows=10, format_cols={
            'Avg_Credit_Limit': "NT$ {:,.0f}",
            'Avg_Bill': "NT$ {:,.0f}",
            'Avg_Payment': "NT$ {:,.0f}",
            'Percentage': "{:.2f}%",
            'Post_Default_Rate': "{:.2f}%",
            'Avg_Max_Delay': "{:+.2f} mos",
            'Avg_Overdue_Months': "{:.2f} mos",
            'Avg_Pay_to_Bill_Ratio': "{:.2f}"
        })

        # Side-by-Side Transposed Comparison Table (Metrics as rows, Clusters as columns)
        st.markdown("### 📋 Side-by-Side Segment Profile Matrix (Transposed)")
        transposed_data = []
        metric_definitions = [
            ('Customers (Count)', 'Customers', '{:,.0f}'),
            ('Population Share (%)', 'Percentage', '{:.2f}%'),
            ('Average Credit Limit (NT$)', 'Avg_Credit_Limit', 'NT$ {:,.0f}'),
            ('Average Age (Years)', 'Avg_Age', '{:.1f} yrs'),
            ('Average Monthly Bill (NT$)', 'Avg_Bill', 'NT$ {:,.0f}'),
            ('Average Monthly Payment (NT$)', 'Avg_Payment', 'NT$ {:,.0f}'),
            ('Payment-to-Bill Ratio', 'Avg_Pay_to_Bill_Ratio', '{:.2f}'),
            ('Peak Repayment Delay', 'Avg_Max_Delay', '{:+.2f} mos'),
            ('Count of Overdue Months', 'Avg_Overdue_Months', '{:.2f} mos'),
            ('Post-Hoc Default Rate (%)', 'Post_Default_Rate', '{:.2f}%')
        ]
        
        matrix_df = pd.DataFrame({'Metric': [m[0] for m in metric_definitions]})
        for _, row in df_profiles.sort_values(by='Cluster').iterrows():
            cl_id = int(row['Cluster'])
            cl_name = f"Cluster {cl_id}: {row['Cluster_Persona'][:18]}..."
            col_vals = []
            for label, key, fmt in metric_definitions:
                val = row.get(key, np.nan)
                col_vals.append(fmt.format(val) if pd.notnull(val) else "-")
            matrix_df[cl_name] = col_vals
            
        render_fintech_table(matrix_df, max_rows=15)

        # Dynamic Persona Cards
        st.markdown("### 🏷️ Discovered Banking Persona Archetypes")
        card_colors = ['#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#f59e0b']
        
        p_cols = st.columns(min(len(df_profiles), 2))
        for idx, (_, prof) in enumerate(df_profiles.sort_values(by='Cluster').iterrows()):
            c_target = p_cols[idx % 2]
            border_c = card_colors[idx % len(card_colors)]
            
            # Formulate strategic advice dynamically
            def_rate = prof['Post_Default_Rate']
            if def_rate > 40.0:
                advice = "High Delinquency Alert: Enforce strict credit limits, require security collateral, and initiate proactive debt restructuring."
            elif prof['Avg_Payment'] > 12000 and prof['Avg_Credit_Limit'] > 250000:
                advice = "Premier Wealth Potential: Affluent, high-spending accounts. Offer premium black/infinite cards, concierge services, and investment products."
            elif prof['Avg_Max_Delay'] <= 0.0:
                advice = "Prime Solvency Segment: Reliable, advance payers with negligible default history. Target with rewards cards, annual fee waivers, and credit line expansions."
            else:
                advice = "Core Retail Revolvers: Stable cardholders who maintain balances and generate steady interest spread with low default. Cross-sell installment loans."
                
            with c_target:
                st.markdown(f"""
                <div class="dark-card" style="border-left: 5px solid {border_c};">
                    <h4 style="color:{border_c}; margin:0 0 8px 0;">Cluster {int(prof['Cluster'])}: {prof['Cluster_Persona']}</h4>
                    <p style="margin: 4px 0;"><strong>Customer Base:</strong> {int(prof['Customers']):,} accounts ({prof['Percentage']:.2f}% of population)</p>
                    <p style="margin: 4px 0;"><strong>Sanctioned Credit Facility:</strong> NT$ {prof['Avg_Credit_Limit']:,.0f} | <strong>Age:</strong> {prof['Avg_Age']:.1f} yrs</p>
                    <p style="margin: 4px 0;"><strong>Monthly Financials:</strong> Bill: NT$ {prof['Avg_Bill']:,.0f} | Payment: NT$ {prof['Avg_Payment']:,.0f}</p>
                    <p style="margin: 4px 0;"><strong>Delinquency Signature:</strong> Max Delay: {prof['Avg_Max_Delay']:+.2f} mos | Overdue Count: {prof.get('Avg_Overdue_Months', 0):.2f} mos</p>
                    <p style="margin: 4px 0;"><strong>Post-Hoc Default Rate:</strong> <strong style="color: {'#ef4444' if def_rate > 30 else '#10b981'};">{def_rate:.2f}%</strong></p>
                    <div class="plain-takeaway" style="margin-top: 10px;">
                        💡 <strong>Banking Strategy:</strong> {advice}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_compare:
        st.markdown('<div class="hero-title">Side-by-Side Cluster Comparison</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Deep-Dive Comparative Analysis Between Any Two Discovered Customer Segments</div>', unsafe_allow_html=True)
        
        if len(df_profiles) >= 2:
            cluster_list = list(df_profiles['Cluster'].unique())
            cluster_names = {c: f"Cluster {c}: {df_profiles.loc[df_profiles['Cluster']==c, 'Cluster_Persona'].values[0]}" for c in cluster_list}
            
            comp_c1, comp_c2 = st.columns(2)
            with comp_c1:
                cl_a = st.selectbox("Select Baseline Segment (Cluster A):", options=cluster_list, format_func=lambda x: cluster_names[x], index=0)
            with comp_c2:
                default_b_idx = 1 if len(cluster_list) > 1 else 0
                cl_b = st.selectbox("Select Benchmark Segment (Cluster B):", options=cluster_list, format_func=lambda x: cluster_names[x], index=default_b_idx)
                
            prof_a = df_profiles[df_profiles['Cluster'] == cl_a].iloc[0]
            prof_b = df_profiles[df_profiles['Cluster'] == cl_b].iloc[0]
            
            # Comparison metrics table
            comp_metrics = [
                ("Customers Count", prof_a['Customers'], prof_b['Customers'], "{:,.0f}"),
                ("Share of Accounts (%)", prof_a['Percentage'], prof_b['Percentage'], "{:.2f}%"),
                ("Avg Credit Limit (NT$)", prof_a['Avg_Credit_Limit'], prof_b['Avg_Credit_Limit'], "NT$ {:,.0f}"),
                ("Avg Monthly Bill (NT$)", prof_a['Avg_Bill'], prof_b['Avg_Bill'], "NT$ {:,.0f}"),
                ("Avg Monthly Payment (NT$)", prof_a['Avg_Payment'], prof_b['Avg_Payment'], "NT$ {:,.0f}"),
                ("Payment-to-Bill Ratio", prof_a['Avg_Pay_to_Bill_Ratio'], prof_b['Avg_Pay_to_Bill_Ratio'], "{:.2f}"),
                ("Avg Max Delay (Months)", prof_a['Avg_Max_Delay'], prof_b['Avg_Max_Delay'], "{:+.2f}"),
                ("Avg Overdue Months Count", prof_a.get('Avg_Overdue_Months', 0), prof_b.get('Avg_Overdue_Months', 0), "{:.2f}"),
                ("Post-Hoc Default Rate (%)", prof_a['Post_Default_Rate'], prof_b['Post_Default_Rate'], "{:.2f}%")
            ]
            
            comp_rows = []
            for name, val_a, val_b, fmt in comp_metrics:
                diff = val_b - val_a
                ratio = (val_b / val_a) if val_a != 0 else np.nan
                diff_str = f"{diff:+,.1f}" if isinstance(val_a, (int, float)) else "-"
                ratio_str = f"{ratio:.2f}x" if pd.notnull(ratio) else "-"
                comp_rows.append({
                    'Comparison Metric': name,
                    f'Cluster {cl_a} (A)': fmt.format(val_a),
                    f'Cluster {cl_b} (B)': fmt.format(val_b),
                    'Delta (B - A)': diff_str,
                    'Ratio (B / A)': ratio_str
                })
            comp_df = pd.DataFrame(comp_rows)
            render_fintech_table(comp_df, max_rows=15)
            
            # Comparative Charts
            if HAS_PLOTLY:
                comp_chart_data = pd.DataFrame({
                    'Metric': ['Credit Limit (k NT$)', 'Monthly Bill (k NT$)', 'Monthly Payment (k NT$)', 'Default Rate (%)'],
                    f'Cluster {cl_a}': [
                        prof_a['Avg_Credit_Limit'] / 1000,
                        prof_a['Avg_Bill'] / 1000,
                        prof_a['Avg_Payment'] / 1000,
                        prof_a['Post_Default_Rate']
                    ],
                    f'Cluster {cl_b}': [
                        prof_b['Avg_Credit_Limit'] / 1000,
                        prof_b['Avg_Bill'] / 1000,
                        prof_b['Avg_Payment'] / 1000,
                        prof_b['Post_Default_Rate']
                    ]
                })
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    x=comp_chart_data['Metric'],
                    y=comp_chart_data[f'Cluster {cl_a}'],
                    name=f"Cluster {cl_a}: {prof_a['Cluster_Persona'][:20]}",
                    marker_color='#3b82f6'
                ))
                fig_comp.add_trace(go.Bar(
                    x=comp_chart_data['Metric'],
                    y=comp_chart_data[f'Cluster {cl_b}'],
                    name=f"Cluster {cl_b}: {prof_b['Cluster_Persona'][:20]}",
                    marker_color='#ef4444'
                ))
                fig_comp.update_layout(
                    title=f"Comparative Metrics: Cluster {cl_a} vs. Cluster {cl_b}",
                    barmode='group',
                    yaxis_title="Metric Value",
                    height=380
                )
                st.plotly_chart(apply_plotly_layout(fig_comp, height=380), width='stretch')
        else:
            st.info("Insufficient clusters available to compare.")

    with tab_explorer:
        st.markdown('<div class="hero-title">Customer Account Explorer</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Individual Account Trajectory Inspection and Benchmark Profiling</div>', unsafe_allow_html=True)
        
        if not df_clustered.empty:
            exp_c1, exp_c2 = st.columns([2, 1])
            with exp_c1:
                cust_id_input = st.number_input(
                    "Enter Customer Account ID (1 to 30,000):",
                    min_value=1,
                    max_value=len(df_clustered),
                    value=1,
                    step=1
                )
            with exp_c2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🎲 Random Customer Account"):
                    cust_id_input = int(np.random.randint(1, len(df_clustered) + 1))
                    
            # Fetch single customer record
            cust_record = df_clustered[df_clustered['ID'] == cust_id_input]
            if not cust_record.empty:
                cust = cust_record.iloc[0]
                cust_cluster = int(cust['Cluster'])
                cust_persona = cust['Cluster_Persona']
                cluster_bench = df_profiles[df_profiles['Cluster'] == cust_cluster].iloc[0]
                
                # Customer High-level Cards
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Sanctioned Credit Limit", f"NT$ {cust['LIMIT_BAL']:,.0f}")
                k2.metric("Customer Age / Sex", f"{cust['AGE']} yrs ({cust.get('SEX_DESC', 'N/A')})")
                k3.metric("Assigned Segment", f"Cluster {cust_cluster}")
                k4.metric("Post-Hoc Default (Y)", "YES (Defaulted)" if cust['Y'] == 1 else "NO (Non-Default)")
                
                st.markdown(f"**Customer Archetype:** `{cust_persona}` | **Education:** `{cust.get('EDUCATION_DESC', 'N/A')}` | **Marital Status:** `{cust.get('MARRIAGE_DESC', 'N/A')}`")
                
                # 6-Month Trajectory Chart (Bill vs Payment)
                months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
                bill_vals = [cust[f'BILL_AMT{i}'] for i in range(6, 0, -1)]
                pay_vals = [cust[f'PAY_AMT{i}'] for i in range(6, 0, -1)]
                
                if HAS_PLOTLY:
                    fig_traj = go.Figure()
                    fig_traj.add_trace(go.Scatter(
                        x=months, y=bill_vals, mode='lines+markers', name='Statement Billed (NT$)',
                        line=dict(color='#3b82f6', width=3), marker=dict(size=8)
                    ))
                    fig_traj.add_trace(go.Scatter(
                        x=months, y=pay_vals, mode='lines+markers', name='Remitted Payment (NT$)',
                        line=dict(color='#10b981', width=3), marker=dict(size=8)
                    ))
                    fig_traj.update_layout(
                        title=f"Customer #{cust_id_input} - 6-Month Billing vs. Payment Remittance Trajectory",
                        xaxis_title="Month (2005)",
                        yaxis_title="Amount (NT$)",
                        height=340
                    )
                    st.plotly_chart(apply_plotly_layout(fig_traj, height=340), width='stretch')
                    
                # Repayment Delay Status Badges
                st.markdown("##### ⏱️ 6-Month Repayment Promptness Status Codes:")
                st_cols = st.columns(6)
                status_keys = ['PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2', 'PAY_0']
                for m_idx, (m_label, s_key) in enumerate(zip(months, status_keys)):
                    s_val = int(cust[s_key])
                    s_color = "#10b981" if s_val <= 0 else "#ef4444"
                    s_desc = "On-Time/Revolving" if s_val <= 0 else f"{s_val} Mo Delay"
                    st_cols[m_idx].markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.5); padding: 8px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                        <span style="font-size: 0.75rem; color: #94a3b8;">{m_label}</span><br>
                        <strong style="color: {s_color}; font-size: 1.1rem;">{s_val}</strong><br>
                        <span style="font-size: 0.7rem; color: {s_color};">{s_desc}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Benchmark deviation table
                st.markdown("##### 📊 Benchmark vs. Assigned Cluster Average:")
                bench_comp = pd.DataFrame([
                    {
                        'Metric': 'Credit Limit (NT$)',
                        'Customer Value': f"NT$ {cust['LIMIT_BAL']:,.0f}",
                        f'Cluster {cust_cluster} Mean': f"NT$ {cluster_bench['Avg_Credit_Limit']:,.0f}",
                        'Status': 'Above Average' if cust['LIMIT_BAL'] > cluster_bench['Avg_Credit_Limit'] else 'Below Average'
                    },
                    {
                        'Metric': 'Average Monthly Bill (NT$)',
                        'Customer Value': f"NT$ {cust['Avg_Bill_Amt']:,.0f}",
                        f'Cluster {cust_cluster} Mean': f"NT$ {cluster_bench['Avg_Bill']:,.0f}",
                        'Status': 'Above Average' if cust['Avg_Bill_Amt'] > cluster_bench['Avg_Bill'] else 'Below Average'
                    },
                    {
                        'Metric': 'Average Monthly Payment (NT$)',
                        'Customer Value': f"NT$ {cust['Avg_Pay_Amt']:,.0f}",
                        f'Cluster {cust_cluster} Mean': f"NT$ {cluster_bench['Avg_Payment']:,.0f}",
                        'Status': 'Above Average' if cust['Avg_Pay_Amt'] > cluster_bench['Avg_Payment'] else 'Below Average'
                    },
                    {
                        'Metric': 'Max Repayment Delay (Months)',
                        'Customer Value': f"{cust['Max_Repayment_Delay']} mos",
                        f'Cluster {cust_cluster} Mean': f"{cluster_bench['Avg_Max_Delay']:+.2f} mos",
                        'Status': 'More Delinquent' if cust['Max_Repayment_Delay'] > cluster_bench['Avg_Max_Delay'] else 'Better than Average'
                    }
                ])
                render_fintech_table(bench_comp, max_rows=5)
            else:
                st.warning(f"Customer #{cust_id_input} not found in database.")

    with tab_pca:
        st.markdown('<div class="hero-title">2D PCA Cluster Projection</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Principal Component Dimensionality Reduction for Geometric Verification</div>', unsafe_allow_html=True)
        
        pca1_var = float(df_meta['PCA1_Var'].values[0]) if not df_meta.empty and 'PCA1_Var' in df_meta.columns else 35.44
        pca2_var = float(df_meta['PCA2_Var'].values[0]) if not df_meta.empty and 'PCA2_Var' in df_meta.columns else 20.72
        total_pca_var = pca1_var + pca2_var
        
        st.markdown(f"""
        <div class="plain-takeaway">
            💡 <strong>Why PCA?</strong> The customer feature space has 8 behavioral dimensions. Principal Component Analysis (PCA)
            mathematically compresses these 8 dimensions into 2 orthogonal axes (PC1 and PC2) while preserving <strong>{total_pca_var:.2f}%</strong> of total data variance
            (PC1: {pca1_var:.2f}%, PC2: {pca2_var:.2f}%). Note that PCA is utilized strictly for visual projection rather than creating the cluster boundaries.
        </div>
        """, unsafe_allow_html=True)
        
        if not active_df.empty and HAS_PLOTLY:
            sample_scatter = active_df.sample(n=min(3000, len(active_df)), random_state=42)
            fig_pca = px.scatter(
                sample_scatter,
                x='PCA1',
                y='PCA2',
                color='Cluster_Persona',
                hover_data=['ID', 'LIMIT_BAL', 'Avg_Bill_Amt', 'Avg_Pay_Amt', 'Max_Repayment_Delay'],
                color_discrete_sequence=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#f59e0b'],
                opacity=0.75
            )
            fig_pca.update_layout(
                xaxis_title=f"PC1: Spending & Remittance Magnitude ({pca1_var:.2f}% Variance)",
                yaxis_title=f"PC2: Overdue Delay & Delinquency Index ({pca2_var:.2f}% Variance)"
            )
            st.plotly_chart(apply_plotly_layout(fig_pca, height=450), width='stretch')


# ============================================================================
# HUB 2: DATA STUDIO & PREPROCESSING
# ============================================================================
elif "2. Data Studio" in nav_selection:
    tab_ds, tab_dq, tab_prep, tab_eda, tab_fe = st.tabs([
        "📄 Dataset Verification",
        "🛡️ Data Quality Audit",
        "🔄 Preprocessing Pipeline",
        "📈 Exploratory Analysis (EDA)",
        "⚙️ Feature Engineering"
    ])
    
    with tab_ds:
        st.markdown('<div class="hero-title">Official UCI Dataset Verification</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-box">
            <strong>Official Dataset Citation:</strong><br>
            Yeh, I. (2009). <em>Default of Credit Card Clients</em>. UCI Machine Learning Repository.<br>
            DOI: <code>10.24432/C55S3H</code> | URL: <a href="https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2B" target="_blank" style="color:#3b82f6;">https://archive.ics.uci.edu/dataset/350/default+of+credit+card+</a>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df_raw):,}")
        c2.metric("Total Features", df_raw.shape[1])
        c3.metric("Missing Values", int(df_raw.isnull().sum().sum()))
        c4.metric("Exact Duplicates", int(df_raw.duplicated().sum()))
        
        st.markdown("### 📋 Staging Columns Specification")
        render_fintech_table(df_raw, max_rows=10)

    with tab_dq:
        st.markdown('<div class="hero-title">Data Quality Audit (Before vs. After)</div>', unsafe_allow_html=True)
        render_fintech_table(df_dq, max_rows=10)
        
        st.markdown("""
        <div class="plain-takeaway">
            💡 <strong>Key Cleaning Summary:</strong> The raw UCI dataset had zero missing rows and zero duplicates.
            However, <code>EDUCATION</code> contained rare undocumented codes (0, 5, 6; 345 accounts) and <code>MARRIAGE</code> had code 0 (54 accounts).
            Our ETL consolidated these into category 4 ('Others') and category 3 ('Others') respectively, ensuring 100% data preservation.
        </div>
        """, unsafe_allow_html=True)

    with tab_prep:
        st.markdown('<div class="hero-title">7-Stage Preprocessing Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        ```mermaid
        flowchart LR
            A["1. Raw Data - 30,000 x 25"] --> B["2. Data Validation - 0 Nulls"]
            B --> C["3. Cleaning - Anomaly Codes"]
            C --> D["4. Feature Selection - Drop ID and Y"]
            D --> E["5. Feature Engineering - Ratios and Delays"]
            E --> F["6. Scaling - StandardScaler"]
            F --> G["7. Clustering Dataset - 8 Features"]
        ```
        """)
        
        # Native styled HTML pipeline representation ensuring 100% fail-safe visual rendering
        st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: center; margin: 15px 0 20px 0;">
            <div style="background: rgba(59, 130, 246, 0.15); border: 1px solid #3b82f6; border-radius: 8px; padding: 8px 14px; font-weight: 600; font-size: 0.82rem; color: #93c5fd;">1. Raw Data (30,000 x 25)</div>
            <span style="color: #64748b; font-weight: bold;">➔</span>
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 8px; padding: 8px 14px; font-weight: 600; font-size: 0.82rem; color: #6ee7b7;">2. Data Validation (0 Nulls)</div>
            <span style="color: #64748b; font-weight: bold;">➔</span>
            <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; border-radius: 8px; padding: 8px 14px; font-weight: 600; font-size: 0.82rem; color: #fde68a;">3. Code Harmonization</div>
            <span style="color: #64748b; font-weight: bold;">➔</span>
            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px; padding: 8px 14px; font-weight: 600; font-size: 0.82rem; color: #fca5a5;">4. Drop ID & Isolate Target Y</div>
            <span style="color: #64748b; font-weight: bold;">➔</span>
            <div style="background: rgba(139, 92, 246, 0.15); border: 1px solid #8b5cf6; border-radius: 8px; padding: 8px 14px; font-weight: 600; font-size: 0.82rem; color: #c4b5fd;">5. Feature Engineering</div>
            <span style="color: #64748b; font-weight: bold;">➔</span>
            <div style="background: rgba(6, 182, 212, 0.15); border: 1px solid #06b6d4; border-radius: 8px; padding: 8px 14px; font-weight: 600; font-size: 0.82rem; color: #67e8f9;">6. StandardScaler Scaling</div>
            <span style="color: #64748b; font-weight: bold;">➔</span>
            <div style="background: rgba(16, 185, 129, 0.2); border: 2px solid #10b981; border-radius: 8px; padding: 8px 14px; font-weight: 700; font-size: 0.82rem; color: #10b981;">7. Clustering Dataset (8 Inputs)</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="dark-card">
            <h4>Decisions Taken During Preprocessing:</h4>
            <ul>
                <li><strong>Removal of Customer ID:</strong> ID is an arbitrary database index with zero financial meaning; including it would corrupt cluster distance math.</li>
                <li><strong>Exclusion of Target Y:</strong> Y is whether the customer defaulted next month. Including it would cause data leakage in an unsupervised task.</li>
                <li><strong>StandardScaler Scaling:</strong> Balances credit limits (up to NT$ 1M) with monthly delay counts (0 to 6 months) so every feature contributes fairly.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab_eda:
        st.markdown('<div class="hero-title">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
        eda_c1, eda_c2 = st.columns(2)
        with eda_c1:
            if not df_clustered.empty and HAS_PLOTLY:
                fig_age = px.histogram(df_clustered, x='AGE', nbins=30, title="Age Distribution of Customers", color_discrete_sequence=['#3b82f6'])
                fig_age.add_vline(x=df_clustered['AGE'].median(), line_dash="dash", line_color="#ef4444", annotation_text="Median Age: 34 yrs")
                st.plotly_chart(apply_plotly_layout(fig_age, height=340), width='stretch')
                st.markdown('<div class="plain-takeaway">💡 Age spans 21 to 79 years, unimodally concentrated around 28–35 years.</div>', unsafe_allow_html=True)
                
            if not df_clustered.empty and HAS_PLOTLY:
                fig_bills = px.histogram(df_clustered, x='Avg_Bill_Amt', nbins=35, title="Average Monthly Bill Statements (NT$)", color_discrete_sequence=['#06b6d4'])
                st.plotly_chart(apply_plotly_layout(fig_bills, height=340), width='stretch')
                st.markdown('<div class="plain-takeaway">💡 Heavily right-skewed; most cardholders incur bills below NT$ 100,000.</div>', unsafe_allow_html=True)
                
        with eda_c2:
            if not df_clustered.empty and HAS_PLOTLY:
                fig_lim = px.histogram(df_clustered, x='LIMIT_BAL', nbins=35, title="Sanctioned Credit Limit Distribution (NT$)", color_discrete_sequence=['#10b981'])
                fig_lim.add_vline(x=df_clustered['LIMIT_BAL'].median(), line_dash="dash", line_color="#ef4444", annotation_text="Median Limit: NT$ 140k")
                st.plotly_chart(apply_plotly_layout(fig_lim, height=340), width='stretch')
                st.markdown('<div class="plain-takeaway">💡 Credit limits peak at NT$ 20k, 50k, 100k, and 200k standard banking tiers.</div>', unsafe_allow_html=True)
                
            if not df_clustered.empty and HAS_PLOTLY:
                corr_cols = ['LIMIT_BAL', 'AGE', 'Avg_Bill_Amt', 'Avg_Pay_Amt', 'Payment_to_Bill_Ratio', 'Avg_Repayment_Status', 'Max_Repayment_Delay', 'Y']
                fig_corr = px.imshow(df_clustered[corr_cols].corr(), text_auto=".2f", color_continuous_scale="Blues", title="Correlation Heatmap")
                st.plotly_chart(apply_plotly_layout(fig_corr, height=340), width='stretch')
                st.markdown('<div class="plain-takeaway">💡 Overdue repayment status correlates strongly (+0.32) with future default (Y).</div>', unsafe_allow_html=True)

    with tab_fe:
        st.markdown('<div class="hero-title">Feature Engineering Architecture</div>', unsafe_allow_html=True)
        render_fintech_table(df_fe, max_rows=15)


# ============================================================================
# HUB 3: MACHINE LEARNING & DIAGNOSTICS
# ============================================================================
elif "3. Machine Learning" in nav_selection:
    tab_km, tab_opt, tab_eval = st.tabs([
        "🎯 K-Means Pipeline & Benchmarking",
        "📊 Optimal K Evaluation (K=2..10)",
        "📐 Model Evaluation & Internal Metrics"
    ])
    
    with tab_km:
        st.markdown('<div class="hero-title">Unsupervised K-Means Clustering</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Algorithm", "K-Means (Lloyd's)")
        c2.metric("Input Features", "8 Standardized")
        c3.metric("Population (N)", f"{total_records:,}")
        c4.metric("Convergence Inits", "10 (n_init=10)")
        
        st.markdown("### 🏆 Algorithmic Benchmark: K-Means vs. Hierarchical Clustering")
        render_fintech_table(df_algo, max_rows=5)
        st.markdown(r"""
        <div class="plain-takeaway">
            💡 <strong>Why K-Means over Hierarchical?</strong> K-Means has linear time complexity $\mathcal{O}(K \cdot N \cdot d)$,
            converging in seconds on 30,000 records. Hierarchical clustering requires cubic time $\mathcal{O}(N^3)$ and $\mathcal{O}(N^2)$ memory,
            which locks up when scaling to millions of banking accounts.
        </div>
        """, unsafe_allow_html=True)

    with tab_opt:
        st.markdown('<div class="hero-title">Optimal Number of Clusters (K = 2 to 10)</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Systematic Multi-Metric Testing Across Inertia, Silhouette, Davies-Bouldin, and Calinski-Harabasz</div>', unsafe_allow_html=True)
        
        render_fintech_table(df_keval, max_rows=12)
        
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            if not df_keval.empty and HAS_PLOTLY:
                fig_el = px.line(df_keval, x='K', y='Inertia_WCSS', markers=True, title="1. Elbow Plot (Inertia/WCSS - Lower is Better)")
                fig_el.add_vline(x=4, line_dash="dash", line_color="#ef4444", annotation_text="Selected K=4")
                st.plotly_chart(apply_plotly_layout(fig_el, height=330), width='stretch')
                
                fig_db = px.line(df_keval, x='K', y='Davies_Bouldin_Index', markers=True, title="3. Davies-Bouldin Index (Lower is Better)")
                fig_db.add_vline(x=4, line_dash="dash", line_color="#ef4444", annotation_text="Selected K=4")
                st.plotly_chart(apply_plotly_layout(fig_db, height=330), width='stretch')
        with col_k2:
            if not df_keval.empty and HAS_PLOTLY:
                fig_si = px.line(df_keval, x='K', y='Silhouette_Score', markers=True, title="2. Silhouette Score (Higher is Better)")
                fig_si.add_vline(x=4, line_dash="dash", line_color="#ef4444", annotation_text="Selected K=4")
                st.plotly_chart(apply_plotly_layout(fig_si, height=330), width='stretch')
                
                fig_ch = px.line(df_keval, x='K', y='Calinski_Harabasz_Index', markers=True, title="4. Calinski-Harabasz Index (Higher is Better)")
                fig_ch.add_vline(x=4, line_dash="dash", line_color="#ef4444", annotation_text="Selected K=4")
                st.plotly_chart(apply_plotly_layout(fig_ch, height=330), width='stretch')
                
        st.markdown("""
        <div class="plain-takeaway">
            💡 <strong>Why K=4 was Selected:</strong> At K=4, the rate of reduction in Within-Cluster Sum of Squares (Inertia) levels off,
            the Calinski-Harabasz score hits a massive peak of <strong>7,596.6</strong>, and the Silhouette Score maintains solid cohesion
            while producing 4 clean, distinct, and actionable business segments.
        </div>
        """, unsafe_allow_html=True)

    with tab_eval:
        st.markdown('<div class="hero-title">Model Evaluation Metrics</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-box-warning">
            <strong>⚠️ Academic Requirement:</strong> This is an <strong>unsupervised clustering problem</strong>.
            Therefore, quality must be evaluated using internal geometric indices (Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz)
            rather than classification accuracy, precision, or recall (which require target ground truth).
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-header"><span class="kpi-label">Inertia / WCSS</span><span>📉</span></div>
            <div class="kpi-num">{float(df_meta['Inertia'].values[0]):,.1f}</div>
            <div class="kpi-delta delta-up">Within-Cluster Sum of Squares</div>
        </div>
        """, unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-header"><span class="kpi-label">Silhouette Score</span><span>💎</span></div>
            <div class="kpi-num">{float(df_meta['Silhouette_Score'].values[0]):.4f}</div>
            <div class="kpi-delta delta-up">Positive Cohesion Ratio</div>
        </div>
        """, unsafe_allow_html=True)
        c3.markdown(f"""
        <div class="kpi-card kpi-amber">
            <div class="kpi-header"><span class="kpi-label">Davies-Bouldin</span><span>📐</span></div>
            <div class="kpi-num">{float(df_meta['Davies_Bouldin'].values[0]):.4f}</div>
            <div class="kpi-delta delta-neutral">Lower Denotes Better Separation</div>
        </div>
        """, unsafe_allow_html=True)
        c4.markdown(f"""
        <div class="kpi-card kpi-purple">
            <div class="kpi-header"><span class="kpi-label">Calinski-Harabasz</span><span>🏆</span></div>
            <div class="kpi-num">{float(df_meta['Calinski_Harabasz'].values[0]):,.1f}</div>
            <div class="kpi-delta delta-up">Variance Ratio Criterion</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# HUB 4: DWDM ARCHITECTURE & WAREHOUSE
# ============================================================================
elif "4. DWDM" in nav_selection:
    tab_star, tab_etl, tab_sql = st.tabs([
        "🏛️ Star Schema Architecture",
        "⚙️ ETL Staging & Loading",
        "💾 SQL Schema Scripts"
    ])
    
    with tab_star:
        st.markdown('<div class="hero-title">Dimensional Star Schema Model</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="plain-takeaway">
            💡 <strong>Analytical Notice:</strong> The raw UCI dataset is provided as a flat file. This Star Schema is a
            project-designed dimensional warehouse architecture engineered to enable Online Analytical Processing (OLAP)
            and feed the analytical data mart.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ```mermaid
        erDiagram
            DIM_CUSTOMER ||--o{ FACT_CUSTOMER_CREDIT_PROFILE : has
            DIM_CREDIT ||--o{ FACT_CUSTOMER_CREDIT_PROFILE : assigns
            DIM_PAYMENT_HISTORY ||--o{ FACT_CUSTOMER_CREDIT_PROFILE : tracks
            DIM_BILLING ||--o{ FACT_CUSTOMER_CREDIT_PROFILE : incurs
            DIM_PAYMENT ||--o{ FACT_CUSTOMER_CREDIT_PROFILE : remits

            DIM_CUSTOMER {
                int Customer_Key PK
                int Customer_ID
                int Age
                varchar Gender_Desc
                varchar Education_Desc
                varchar Marital_Status_Desc
            }
            DIM_CREDIT {
                int Credit_Key PK
                numeric Credit_Limit
                varchar Credit_Tier
            }
            DIM_PAYMENT_HISTORY {
                int Payment_History_Key PK
                int PAY_0
                int PAY_2
                int Max_Delay_Months
            }
            DIM_BILLING {
                int Billing_Key PK
                numeric Total_Bill_Amount
                numeric Avg_Bill_Amount
            }
            DIM_PAYMENT {
                int Payment_Key PK
                numeric Total_Payment_Amount
                numeric Avg_Payment_Amount
            }
            FACT_CUSTOMER_CREDIT_PROFILE {
                int Profile_Fact_Key PK
                numeric Credit_Limit
                numeric Total_Billed_NTD
                numeric Total_Paid_NTD
                numeric Payment_to_Bill_Ratio
                int Max_Consecutive_Delay
                int Target_Default_Next_Month
            }
        ```
        """)

    with tab_etl:
        st.markdown('<div class="hero-title">ETL Staging & Transformation Stages</div>', unsafe_allow_html=True)
        st.markdown("""
        ```mermaid
        flowchart LR
            A["Extract: Raw UCI CSV"] --> B["Validate: Schema and Null Checks"]
            B --> C["Transform: Harmonize Categories and Calculate Ratios"]
            C --> D["Load: Populate Dimensions and Fact Table"]
            D --> E["Data Mart: Feed Agg_Customer_Segment_Mart"]
        ```
        """)

    with tab_sql:
        st.markdown('<div class="hero-title">Production SQL Scripts</div>', unsafe_allow_html=True)
        sql_choice = st.radio("Choose Script to Inspect:", ["sql/create_tables.sql (DDL)", "sql/etl.sql (Transformation Queries)"], horizontal=True)
        target_path = 'sql/create_tables.sql' if "create_tables" in sql_choice else 'sql/etl.sql'
        if os.path.exists(target_path):
            with open(target_path, 'r', encoding='utf-8') as f:
                st.code(f.read(), language='sql')


# ============================================================================
# HUB 5: ACADEMIC RESEARCH & VIVA CENTER
# ============================================================================
elif "5. Academic" in nav_selection:
    tab_res, tab_posty, tab_biz, tab_viva = st.tabs([
        "🔬 Research Questions & Findings",
        "📊 Post-Hoc Default (Y) Analysis",
        "💼 Banking Applications & Limitations",
        "🎓 Interactive Viva Preparation (20 Q&As)"
    ])
    
    with tab_res:
        st.markdown('<div class="hero-title">Research Methodology & Core Questions</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="dark-card">
            <h3>Central Research Question</h3>
            <p style="font-size:1.1rem; color:#3b82f6; font-weight:600;">
                "Can unsupervised clustering identify meaningful customer segments based on demographic,
                credit limit, billing trajectory, and payment behavior?"
            </p>
            <p><strong>Empirical Answer:</strong> Yes. By withholding target Y and letting K-Means discover natural geometric clusters,
            we identified 4 distinct behavioral segments that correlate strongly with financial solvency and risk, proving that
            transactional behavior contains natural credit patterns.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab_posty:
        st.markdown('<div class="hero-title">Post-Clustering Target Variable (Y) Analysis</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-box">
            <strong>Critical Concept:</strong> Target <code>Y</code> (default next month) was <strong>NEVER used to train the clusters</strong>.
            It is evaluated strictly after clustering to see how natural behavioral groupings align with credit default.
        </div>
        """, unsafe_allow_html=True)
        
        render_fintech_table(df_post_y, max_rows=10, format_cols={
            'Default_0_Pct': "{:.2f}% (Non-Default)",
            'Default_1_Pct': "{:.2f}% (Default)"
        })
        
        overall_default_pct = float(df_meta['Overall_Default_Rate'].values[0]) if not df_meta.empty and 'Overall_Default_Rate' in df_meta.columns else (float(df_raw['Y'].mean() * 100) if not df_raw.empty else 22.12)
        if not df_post_y.empty and HAS_PLOTLY:
            fig_y = px.bar(
                df_post_y, x='Cluster_Persona', y='Default_1_Pct', color='Cluster_Persona', text='Default_1_Pct',
                title="Empirical Default Rate by Segment (Target Variable Y - %)",
                color_discrete_sequence=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#f59e0b']
            )
            fig_y.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_y.add_hline(y=overall_default_pct, line_dash="dash", line_color="#f59e0b", annotation_text=f"Overall Bank Baseline ({overall_default_pct:.2f}%)")
            st.plotly_chart(apply_plotly_layout(fig_y, height=360), width='stretch')
            
        if not df_post_y.empty:
            max_risk_p = df_post_y.loc[df_post_y['Default_1_Pct'].idxmax()]
            min_risk_p = df_post_y.loc[df_post_y['Default_1_Pct'].idxmin()]
            st.markdown(f"""
            <div class="plain-takeaway">
                💡 <strong>Observation:</strong> <strong>Cluster {int(max_risk_p['Cluster'])} ({max_risk_p['Cluster_Persona']})</strong> exhibited an empirical 
                default rate of <strong>{max_risk_p['Default_1_Pct']:.2f}%</strong> (more than double the bank baseline of {overall_default_pct:.2f}%), 
                whereas <strong>Cluster {int(min_risk_p['Cluster'])} ({min_risk_p['Cluster_Persona']})</strong> exhibited only <strong>{min_risk_p['Default_1_Pct']:.2f}% default</strong>. 
                This demonstrates that unsupervised behavioral clustering naturally uncovers high-risk delinquency profiles without any supervised outcome guidance.
            </div>
            """, unsafe_allow_html=True)

    with tab_biz:
        st.markdown('<div class="hero-title">Banking Applications & Limitations</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="dark-card">
                <h4 style="color:#3b82f6;">Strategic Banking Applications</h4>
                <ul>
                    <li><strong>Tailored Credit Limit Adjustments:</strong> Safely increase limits for Segment 0 and 3; freeze limits for Segment 2.</li>
                    <li><strong>Proactive Delinquency Warnings:</strong> Automated SMS reminders for accounts shifting toward Segment 2 behaviors.</li>
                    <li><strong>Loyalty Incentives:</strong> Fee waivers and merchant cashbacks for Segment 1 to drive daily card usage.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="dark-card">
                <h4 style="color:#ef4444;">Documented Project Limitations</h4>
                <ul>
                    <li><strong>Domain Scope:</strong> Reflects unsecured revolving credit cards; does not cover checking deposits or mortgages.</li>
                    <li><strong>Historical Geography:</strong> Data from Taiwan in 2005; macroeconomic factors may differ in contemporary markets.</li>
                    <li><strong>Correlation vs. Causation:</strong> Clustering demonstrates behavioral correlation, not causal drivers of default.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab_viva:
        st.markdown('<div class="hero-title">Interactive Viva & Exam Prep Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Instant Model Answers for 20 Common Professor / Examiner Questions</div>', unsafe_allow_html=True)
        
        viva_qas = [
            ("1. Why did you choose this dataset?",
             "We selected the official UCI 'Default of Credit Card Clients' dataset (Yeh, 2009; ID: 350) because it provides authentic, multi-month credit limit, billing, and repayment trajectories for 30,000 customers. Unlike single transaction logs, it captures multi-dimensional debt dynamics necessary for DWDM dimensional modeling and clustering."),
            ("2. What is customer segmentation?",
             "Customer segmentation is partitioning a diverse customer base into distinct, homogeneous subgroups based on shared demographic and financial characteristics, enabling personalized marketing, risk mitigation, and product differentiation."),
            ("3. Why clustering?",
             "Clustering is the primary unsupervised data mining technique that discovers natural groupings in unlabelled data based on geometric distance, eliminating subjective human bias."),
            ("4. Why K-Means?",
             "K-Means offers linear computational efficiency O(K * N * d) and fast convergence, making it highly scalable for production banking data warehouses compared to cubic-time Hierarchical clustering."),
            ("5. Why is this unsupervised learning?",
             "Because no target labels or ground truth were provided during model training. The algorithm grouped accounts solely based on similarities in their demographic, billing, and payment behavior."),
            ("6. Why did you remove ID?",
             "Customer ID is an arbitrary surrogate index with zero behavioral signal. Leaving it in would distort Euclidean distance calculations."),
            ("7. Why did you exclude Y from clustering?",
             "Y is the future default outcome. Including Y would cause supervised data leakage and destroy the unsupervised nature of customer segmentation. Y was strictly reserved for post-clustering validation."),
            ("8. How did you choose K=4?",
             "We tested K=2 to 10 across 4 internal metrics. K=4 marks the primary elbow inflection, achieves a Calinski-Harabasz score peak of 7,596.6, and produces 4 strategically distinct, actionable banking personas."),
            ("9. What is the Elbow Method?",
             "It plots Within-Cluster Sum of Squares (Inertia) against K. The optimal K is the elbow point where adding another cluster produces diminishing returns in inertia reduction."),
            ("10. What is Silhouette Score?",
             "It measures how cohesive a point is to its own cluster compared to neighboring clusters, ranging from -1 to +1. Our K=4 model achieves a solid positive score of 0.2443."),
            ("11. What is Davies-Bouldin Index?",
             "It evaluates average similarity between each cluster and its most similar counterpart. Lower scores denote superior cluster separation. Our model achieved a favorable 1.3970."),
            ("12. What is Calinski-Harabasz Index?",
             "Also known as the Variance Ratio Criterion, it is the ratio of between-cluster dispersion to within-cluster dispersion. Higher values denote better clustering; our model achieved 7,596.6."),
            ("13. Why isn't classification accuracy used?",
             "Customer segmentation is unsupervised; there are no predefined 'correct' classes. Accuracy requires supervised ground-truth labels and is mathematically invalid for clustering."),
            ("14. What preprocessing was performed?",
             "Validated 30,000 records, cleaned undocumented categories in EDUCATION and MARRIAGE, engineered solvency ratios and delay metrics, removed ID and Y, and applied StandardScaler."),
            ("15. Why did you scale the features?",
             "Credit limits (NT$ 1M) and monthly delays (0-6 months) have wildly different numerical ranges. Without StandardScaler, credit limits would dominate Euclidean distances by 1,000x."),
            ("16. What does each cluster represent?",
             "Cluster 0: Advance & Prompt Payers (24.2%); Cluster 1: Moderate Credit Revolvers (47.4%); Cluster 2: High Delay Risk Cohort (16.3%); Cluster 3: Premium High Spenders (12.1%)."),
            ("17. What is ETL?",
             "Extract, Transform, Load: Ingesting raw staged data, cleaning anomalies and engineering behavioral metrics, and loading structured dimensional tables and fact profiles."),
            ("18. What is the DWDM component?",
             "Data Warehousing organizes records into a dimensional Star Schema optimized for OLAP querying; Data Mining applies K-Means to extract non-trivial customer segments."),
            ("19. What is the data warehouse design?",
             "A Star Schema with 5 dimensions (Dim_Customer, Dim_Credit, Dim_Payment_History, Dim_Billing, Dim_Payment) surrounding Fact_Customer_Credit_Profile."),
            ("20. What are the limitations?",
             "Covers revolving credit cards only; based on historical Taiwan 2005 context; shows behavioral correlation rather than causation.")
        ]
        
        for q, a in viva_qas:
            with st.expander(f"❓ {q}"):
                st.markdown(f"**Answer:** {a}")


# ============================================================================
# HUB 6: REAL-TIME CUSTOMER PREDICTOR
# ============================================================================
elif "6. Real-Time" in nav_selection:
    st.markdown('<div class="hero-title">Real-Time Customer Segment Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Score and Classify Live Credit Applicants into Discovered Personas</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="plain-takeaway">
        💡 <strong>Interactive Tool:</strong> Enter credit parameters for a prospective or existing customer below.
        The trained K-Means model will standardize the inputs and immediately classify them into one of the 4 discovered segments!
    </div>
    """, unsafe_allow_html=True)
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        inp_limit = st.number_input("Sanctioned Credit Limit (NT$):", min_value=10000, max_value=1000000, value=150000, step=10000)
        inp_age = st.slider("Customer Age:", min_value=18, max_value=85, value=35)
        inp_avg_bill = st.number_input("Average Monthly Bill (NT$):", min_value=0, max_value=500000, value=45000, step=5000)
    with col_in2:
        inp_avg_pay = st.number_input("Average Monthly Payment (NT$):", min_value=0, max_value=500000, value=5000, step=1000)
        inp_ratio = st.slider("Payment-to-Bill Solvency Ratio:", min_value=0.0, max_value=5.0, value=0.15, step=0.05)
        inp_avg_status = st.slider("Average Repayment Status (-2 to +8):", min_value=-2.0, max_value=8.0, value=0.2, step=0.1)
    with col_in3:
        inp_max_delay = st.slider("Maximum Overdue Delay (Months):", min_value=-2, max_value=8, value=1)
        inp_delayed_months = st.slider("Count of Overdue Months (0 to 6):", min_value=0, max_value=6, value=1)
        
    if st.button("🚀 Classify Customer Account", type="primary"):
        if kmeans_model is not None and scaler_model is not None:
            # Input validations and sanity checks
            warnings = []
            if inp_avg_bill == 0 and inp_max_delay > 0:
                warnings.append("Note: Cardholder has NT$ 0 statement bill but indicates active overdue delay. Model will classify based on aggregate repayment patterns.")
            if inp_avg_pay > 10 * (inp_avg_bill + 1) and inp_avg_bill > 10000:
                warnings.append("Note: Payment remittance is exceptionally large relative to monthly statement balance (solvency ratio > 10x).")
            if inp_delayed_months > 0 and inp_max_delay <= 0:
                warnings.append("Note: Overdue month count is positive, but maximum observed delay is <= 0 months. Standardizing delay metrics.")
                
            for w in warnings:
                st.info(f"ℹ️ {w}")
                
            feature_cols = [
                'LIMIT_BAL', 'AGE', 'Avg_Bill_Amt', 'Avg_Pay_Amt',
                'Payment_to_Bill_Ratio', 'Avg_Repayment_Status',
                'Max_Repayment_Delay', 'Delayed_Months_Count'
            ]
            raw_input_df = pd.DataFrame([[
                inp_limit, inp_age, inp_avg_bill, inp_avg_pay,
                inp_ratio, inp_avg_status, inp_max_delay, inp_delayed_months
            ]], columns=feature_cols)
            
            scaled_input = scaler_model.transform(raw_input_df)
            pred_cluster = int(kmeans_model.predict(scaled_input)[0])
            
            # Calculate Euclidean distances to all cluster centroids
            distances = kmeans_model.transform(scaled_input)[0]
            sorted_dist_indices = np.argsort(distances)
            closest_dist = distances[sorted_dist_indices[0]]
            runner_up_cluster = sorted_dist_indices[1]
            runner_up_dist = distances[runner_up_cluster]
            confidence_margin = runner_up_dist - closest_dist
            
            # Fetch cluster profile dynamically
            cluster_prof_row = df_profiles[df_profiles['Cluster'] == pred_cluster]
            if not cluster_prof_row.empty:
                pred_persona = cluster_prof_row.iloc[0]['Cluster_Persona']
                exp_default = cluster_prof_row.iloc[0]['Post_Default_Rate']
                seg_share = cluster_prof_row.iloc[0]['Percentage']
                seg_size = cluster_prof_row.iloc[0]['Customers']
                avg_seg_lim = cluster_prof_row.iloc[0]['Avg_Credit_Limit']
            else:
                pred_persona = f"Cluster {pred_cluster}"
                exp_default = 22.12
                seg_share = 25.0
                seg_size = 7500
                avg_seg_lim = 150000
                
            st.success(f"🎯 Assigned Segment: **Cluster {pred_cluster} — {pred_persona}** (Centroid Distance: {closest_dist:.3f})")
            
            # Actionable banking recommendation
            if exp_default > 40.0:
                rec_title = "🛑 High Credit Risk Delinquency Warning"
                rec_desc = "Account matches characteristics of chronic delayers with high default incidence. Recommend capping credit limits, restricting cash advances, and routing to early collections."
                rec_badge = "CRITICAL RISK"
                badge_bg = "#ef4444"
            elif inp_avg_pay > 12000 and inp_limit > 250000:
                rec_title = "🌟 Premier High-Volume Affluent Account"
                rec_desc = "Account exhibits superior purchasing capacity and remittance volume. Offer premium Visa Infinite / World Elite tier, lifestyle concierge, and elevated credit facilities."
                rec_badge = "HIGH VALUE"
                badge_bg = "#8b5cf6"
            elif inp_max_delay <= 0:
                rec_title = "🛡️ Prime Advance & Prompt Solvency Profile"
                rec_desc = "Account consistently settles on time or in advance with strong liquidity. Ideal candidate for credit limit upgrades, fee waivers, and loyalty cross-selling."
                rec_badge = "PRIME CUSTOMER"
                badge_bg = "#3b82f6"
            else:
                rec_title = "💳 Standard Retail Revolver Profile"
                rec_desc = "Account regularly utilizes revolving credit lines with moderate interest spread and low default. Maintain ongoing automated limit monitoring."
                rec_badge = "CORE REVOLVER"
                badge_bg = "#10b981"
                
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.markdown(f"""
                <div class="dark-card" style="border-left: 5px solid {badge_bg};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; color:{badge_bg};">{rec_title}</h4>
                        <span style="background:{badge_bg}; color:#fff; font-size:0.7rem; font-weight:700; padding:2px 8px; border-radius:12px;">{rec_badge}</span>
                    </div>
                    <p style="margin: 8px 0 4px 0;"><strong>Predicted Cluster:</strong> Cluster {pred_cluster} ({pred_persona})</p>
                    <p style="margin: 4px 0;"><strong>Historical Segment Default Rate:</strong> <strong>{exp_default:.2f}%</strong></p>
                    <p style="margin: 4px 0;"><strong>Segment Population Base:</strong> {int(seg_size):,} accounts ({seg_share:.1f}% share)</p>
                    <p style="margin: 4px 0;"><strong>Segment Benchmark Credit Limit:</strong> NT$ {avg_seg_lim:,.0f} (Applicant: NT$ {inp_limit:,})</p>
                    <div class="plain-takeaway" style="margin-top:10px;">
                        💡 <strong>Banking Directive:</strong> {rec_desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with c_res2:
                # Centroid Proximity Bar Chart
                dist_data = []
                for cl_idx in range(len(distances)):
                    p_name = df_profiles.loc[df_profiles['Cluster']==cl_idx, 'Cluster_Persona'].values[0] if not df_profiles.empty else f"Cluster {cl_idx}"
                    dist_data.append({
                        'Cluster': f"Cluster {cl_idx}",
                        'Persona': p_name,
                        'Distance': round(distances[cl_idx], 3),
                        'Status': 'Assigned Centroid' if cl_idx == pred_cluster else 'Alternative'
                    })
                dist_df = pd.DataFrame(dist_data)
                
                if HAS_PLOTLY:
                    fig_dist = px.bar(
                        dist_df,
                        x='Distance',
                        y='Cluster',
                        orientation='h',
                        color='Status',
                        text='Distance',
                        title=f"Centroid Proximity (Separation Margin: {confidence_margin:.3f})",
                        color_discrete_map={'Assigned Centroid': '#10b981', 'Alternative': '#64748b'}
                    )
                    fig_dist.update_traces(textposition='outside')
                    fig_dist.update_layout(xaxis_title="Euclidean Distance in Scaled Space (Lower is Closer)", yaxis_title="Cluster Centroid")
                    st.plotly_chart(apply_plotly_layout(fig_dist, height=260), width='stretch')

            # 2D PCA Projection Plot with applicant overlay
            if pca_model is not None and HAS_PLOTLY and not active_df.empty:
                st.markdown("### 🌐 Visualizing Applicant on 2D PCA Manifold")
                pca_point = pca_model.transform(scaled_input)[0]
                
                sample_pca_pts = active_df.sample(n=min(1500, len(active_df)), random_state=42)
                fig_overlay = px.scatter(
                    sample_pca_pts,
                    x='PCA1',
                    y='PCA2',
                    color='Cluster_Persona',
                    color_discrete_sequence=['#3b82f6', '#10b981', '#ef4444', '#8b5cf6', '#f59e0b'],
                    opacity=0.45,
                    title="2D PCA Position of Real-Time Applicant (⭐ Red Star)"
                )
                
                # Add applicant marker
                fig_overlay.add_trace(go.Scatter(
                    x=[pca_point[0]],
                    y=[pca_point[1]],
                    mode='markers+text',
                    marker=dict(symbol='star', size=20, color='#ff0055', line=dict(color='#ffffff', width=2)),
                    name='Current Applicant',
                    text=['⭐ Live Applicant'],
                    textposition='top center'
                ))
                fig_overlay.update_layout(
                    xaxis_title="PC1: Spending & Remittance Magnitude",
                    yaxis_title="PC2: Overdue Delay & Delinquency Index"
                )
                st.plotly_chart(apply_plotly_layout(fig_overlay, height=420), width='stretch')
        else:
            st.error("Model artifacts not loaded. Please ensure src/main.py has executed.")
