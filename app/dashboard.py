"""
dashboard.py — Sanket
Unsupervised anomaly detection dashboard for a refinery centrifugal compressor.

Run:  python -m streamlit run app/dashboard.py --server.headless=true --browser.gatherUsageStats=false
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="Sanket · Compressor Monitoring",
                   layout="wide",
                   initial_sidebar_state="collapsed")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,300..900&family=Inter:wght@300..800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #323232;
        -webkit-font-smoothing: antialiased;
    }

    /* === App background: clean white with subtle dot grid === */
    .stApp {
        background-color: #f7f7f5;
        background-image: radial-gradient(rgba(50,50,50,0.035) 1px, transparent 1px);
        background-size: 26px 26px;
    }

    /* === Floating background orbs === */
    .bg-orbs {
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }
    .orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(80px);
    }
    .orb-red {
        top: -14%;
        right: -8%;
        width: 520px; height: 520px;
        background: rgba(238,49,36,0.045);
    }
    .orb-blue {
        bottom: -16%;
        left: -12%;
        width: 580px; height: 580px;
        background: rgba(0,113,179,0.05);
    }
    .orb-bone {
        top: 42%;
        right: -4%;
        width: 380px; height: 380px;
        background: rgba(224,210,183,0.10);
    }

    .block-container {
        padding-top: 0;
        padding-bottom: 4rem;
        max-width: 1280px;
        position: relative;
        z-index: 1;
    }
    header, footer, #MainMenu { visibility: hidden; }

    /* === Nav === */
    .nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.4rem 0;
        border-bottom: 1px solid #e8e5de;
        margin-bottom: 4.5rem;
        position: relative;
        z-index: 2;
    }
    .nav-mark {
        font-family: 'Source Serif 4', serif;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: #323232;
    }
    .nav-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #8a8578;
    }
    .nav-live {
        display: flex; align-items: center; gap: 0.55rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #0071B3;
    }
    .live-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #0071B3;
        box-shadow: 0 0 0 4px rgba(0,113,179,0.14);
    }

    /* === Hero === */
    .hero {
        text-align: center;
        padding: 1rem 0 5rem 0;
        position: relative;
        z-index: 2;
    }
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        color: #EE3124;
        margin-bottom: 1.9rem;
    }
    .hero-name {
        font-family: 'Source Serif 4', serif;
        font-weight: 700;
        font-size: 8rem;
        line-height: 0.86;
        letter-spacing: -0.05em;
        color: #EE3124;
        margin: 0;
    }
    .hero-italic {
        font-family: 'Source Serif 4', serif;
        font-style: italic;
        font-weight: 300;
        font-size: 1.3rem;
        color: #4a4438;
        margin-top: 1.6rem;
        letter-spacing: -0.005em;
    }
    .hero-rule {
        width: 48px;
        height: 2px;
        background: #0071B3;
        margin: 2.2rem auto 2.2rem auto;
    }
    .hero-body {
        max-width: 680px;
        margin: 0 auto;
        font-size: 1rem;
        line-height: 1.75;
        color: #5c5648;
    }

    /* === Section header === */
    .sect {
        display: flex;
        align-items: baseline;
        gap: 1.4rem;
        margin: 0 0 2rem 0;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e8e5de;
        position: relative;
        z-index: 2;
    }
    .sect-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.16em;
        color: #EE3124;
    }
    .sect-title {
        font-family: 'Source Serif 4', serif;
        font-weight: 500;
        font-size: 1.5rem;
        letter-spacing: -0.015em;
        color: #323232;
        margin: 0;
    }

    /* === Cards === */
    .card {
        background: #ffffff;
        border: 1px solid #e8e5de;
        border-radius: 2px;
        padding: 1.6rem 1.5rem 1.5rem 1.5rem;
        height: 100%;
        position: relative;
        box-shadow: 0 1px 2px rgba(50,50,50,0.03);
    }
    .card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: #EE3124;
    }
    .card.blue::before {
        background: #0071B3;
    }
    .card-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.64rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #EE3124;
        margin-bottom: 1.1rem;
    }
    .card.blue .card-label {
        color: #0071B3;
    }
    .card-value {
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        font-size: 2.6rem;
        line-height: 0.95;
        letter-spacing: -0.03em;
        color: #323232;
        font-variant-numeric: tabular-nums lining-nums;
    }
    .card-value.small {
        font-size: 1.4rem;
        line-height: 1.2;
    }
    .card-note {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        color: #8a8578;
        margin-top: 0.95rem;
        font-weight: 500;
    }

    /* === Status blocks === */
    .status {
        padding: 1.3rem 1.5rem;
        font-size: 0.94rem;
        line-height: 1.7;
        border-left: 3px solid;
        background: #ffffff;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 2px rgba(50,50,50,0.03);
    }
    .status b {
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        font-style: italic;
        letter-spacing: -0.005em;
    }
    .status-ok    { border-color: #0071B3; color: #003a5c; }
    .status-warn  { border-color: #EE3124; color: #7a1610; }
    .status-info  { border-color: #c9b48f; color: #5a4a30; }
    .status-bone  { border-color: #E0D2B7; color: #4a4438; }

    /* === Tabs === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e8e5de;
        margin-bottom: 2.5rem;
        position: relative;
        z-index: 2;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        padding: 15px 28px 15px 0;
        margin-right: 28px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #a39a86;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #EE3124 !important;
        border-bottom: 3px solid #EE3124 !important;
        font-weight: 600;
    }

    /* === Captions === */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #8a8578 !important;
        font-size: 0.76rem !important;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.03em;
    }

    /* === Footer === */
    .foot {
        margin-top: 5rem;
        padding-top: 2rem;
        border-top: 1px solid #e8e5de;
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #a39a86;
        position: relative;
        z-index: 2;
    }

    .stDataFrame { font-size: 0.88rem; }

    /* === Process step list === */
    .step-list {
        counter-reset: step;
    }
    .step {
        display: flex;
        gap: 1.2rem;
        padding: 1.1rem 0;
        border-bottom: 1px solid #f0ede6;
        align-items: flex-start;
    }
    .step:last-child {
        border-bottom: none;
    }
    .step-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        color: #0071B3;
        flex-shrink: 0;
        padding-top: 0.15rem;
    }
    .step-body {
        font-size: 0.94rem;
        line-height: 1.65;
        color: #4a4438;
    }
    .step-body b {
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        color: #323232;
    }
</style>

<div class="bg-orbs">
    <div class="orb orb-red"></div>
    <div class="orb orb-blue"></div>
    <div class="orb orb-bone"></div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data
def load_full_year():
    df = pd.read_csv("data/processed/full_year.csv", parse_dates=["Timestamp"])
    scores = pd.read_csv("results/tables/anomaly_scores.csv", parse_dates=["Timestamp"])
    return df.merge(scores, on="Timestamp", how="left")


@st.cache_data
def load_periods_with_actions():
    return pd.read_csv("results/tables/periods_with_actions.csv",
                       parse_dates=["start", "end"])


@st.cache_data
def load_contributors():
    return pd.read_csv("results/tables/contributors_filtered.csv")


full_df = load_full_year()
periods_df = load_periods_with_actions()
contrib_df = load_contributors()


# ============================================================
# NAV
# ============================================================
st.markdown('''
<div class="nav">
    <div class="nav-mark">Sanket</div>
    <div class="nav-meta">Compressor · Motor Oil Hellas · 2022</div>
    <div class="nav-live"><span class="live-dot"></span>Model live</div>
</div>
''', unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================
st.markdown('''
<div class="hero">
    <div class="hero-eyebrow">Unsupervised Monitoring · Refinery Compressor</div>
    <h1 class="hero-name">SANKET</h1>
    <div class="hero-italic">abnormal operation, between the noise.</div>
    <div class="hero-rule"></div>
    <p class="hero-body">
        Learns the joint behaviour of 25 compressor sensors during steady-state
        operation, flags periods that deviate, and ranks the variables that
        changed most. Trained on one full year of DCS data from a BCL 509/A
        centrifugal compressor. No fault labels were used during training.
    </p>
</div>
''', unsafe_allow_html=True)


# ============================================================
# HEADLINE STRIP
# ============================================================
st.markdown('''
<div class="sect">
    <div class="sect-num">01</div>
    <div class="sect-title">Model at a glance</div>
</div>
''', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown('''
        <div class="card">
            <div class="card-label">Algorithm</div>
            <div class="card-value small">Isolation Forest</div>
            <div class="card-note">Unsupervised · no labels</div>
        </div>
    ''', unsafe_allow_html=True)

with m2:
    st.markdown('''
        <div class="card blue">
            <div class="card-label">Sensors monitored</div>
            <div class="card-value">25</div>
            <div class="card-note">Axial, vibration, T, P, speed</div>
        </div>
    ''', unsafe_allow_html=True)

with m3:
    high_count = int((periods_df["priority"] == "High").sum())
    st.markdown(f'''
        <div class="card">
            <div class="card-label">High-priority periods</div>
            <div class="card-value">{high_count}</div>
            <div class="card-note">Investigate first</div>
        </div>
    ''', unsafe_allow_html=True)

with m4:
    st.markdown(f'''
        <div class="card blue">
            <div class="card-label">Anomaly periods</div>
            <div class="card-value">{len(periods_df)}</div>
            <div class="card-note">Lasting ≥ 1 hour</div>
        </div>
    ''', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab_overview, tab_trends, tab_periods, tab_support = st.tabs(
    ["Overview", "Process Trends", "Anomaly Periods & Actions", "Operator Decision Support"]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab_overview:

    # ---- THE INDUSTRIAL PROBLEM ----
    st.markdown('''
        <div class="sect">
            <div class="sect-num">02</div>
            <div class="sect-title">The industrial problem</div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-warn">
        <b>What was happening.</b> A refinery centrifugal compressor in hydrogen
        service runs 24 hours a day, 365 days a year. It is monitored by 25 sensors —
        axial displacement, vibration, temperatures, pressures, and speed.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-info">
        <b>Why it matters.</b> An operator cannot watch 25 signals at once.
        Subtle drift — like a thrust bearing wearing out over weeks — is invisible
        until it becomes a failure. An unplanned compressor shutdown in a refinery
        can cost millions of dollars in lost production and emergency maintenance.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-ok">
        <b>The gap.</b> The plant had years of historical data but no automated
        system to learn from it. Operators relied on individual alarm limits —
        which only trigger after a variable has already crossed a fixed threshold.
        By then, the damage is often already done.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---- WHAT SANKET DOES ----
    st.markdown('''
        <div class="sect">
            <div class="sect-num">03</div>
            <div class="sect-title">What Sanket does</div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="step-list">
            <div class="step">
                <div class="step-num">01</div>
                <div class="step-body">
                    <b>Learns what normal looks like.</b> Takes one year of real
                    DCS data from the refinery compressor and learns the joint
                    behaviour of all 25 sensors during steady-state operation.
                    Not the range of each sensor alone — how they move together.
                    This is where the ML lives: an Isolation Forest model trained
                    on 34,014 steady-state readings. No fault labels used. Entirely
                    unsupervised.
                </div>
            </div>
            <div class="step">
                <div class="step-num">02</div>
                <div class="step-body">
                    <b>Flags abnormal periods.</b> When applied to the full year,
                    the model produces an anomaly score for every timestamp. High
                    score means "these 25 readings do not look like the combinations
                    I learned." Periods lasting more than 1 hour are kept as
                    anomaly events.
                </div>
            </div>
            <div class="step">
                <div class="step-num">03</div>
                <div class="step-body">
                    <b>Ranks the contributors.</b> For each flagged period, it
                    computes a z-score for every sensor — how many standard
                    deviations from normal. It groups the 25 sensors into 5 physical
                    categories (axial, vibration, temperature, pressure, speed) and
                    shows which group dominated the anomaly.
                </div>
            </div>
            <div class="step">
                <div class="step-num">04</div>
                <div class="step-body">
                    <b>Classifies the event.</b> Rule-based decision logic built
                    on chemical engineering reasoning labels each period: Shutdown,
                    Restart transient, Possible thrust bearing wear, Pressure system
                    event, Thermal event, Cold start / low load, or Unclassified.
                </div>
            </div>
            <div class="step">
                <div class="step-num">05</div>
                <div class="step-body">
                    <b>Suggests what to do.</b> Every category has a suggested
                    action. For "Possible thrust bearing wear," the action is:
                    <i>Investigate thrust bearing. Check axial displacement trend,
                    verify lube oil flow, and schedule inspection at next
                    opportunity.</i>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---- ANOMALY SCORE PLOT ----
    st.markdown('''
        <div class="sect">
            <div class="sect-num">04</div>
            <div class="sect-title">Anomaly score over the year</div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-info">
        The Isolation Forest was trained on 34,014 steady-state readings only.
        When applied to the full year, it independently identified both planned
        shutdowns (June, September) and the restart transients that followed them —
        without ever having seen a shutdown during training.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(13, 3.6))
    fig.patch.set_facecolor("#f7f7f5")
    ax.set_facecolor("#ffffff")
    ax.plot(full_df["Timestamp"], full_df["anomaly_score"],
            linewidth=0.6, color="#323232")
    for _, row in periods_df.iterrows():
        ax.axvspan(row["start"], row["end"], color="#EE3124", alpha=0.10)
    threshold = full_df.loc[full_df["is_anomaly"], "anomaly_score"].min()
    ax.axhline(threshold, color="#0071B3", linestyle="--", linewidth=1)
    ax.set_ylabel("Anomaly score", fontsize=9, color="#4a4438")
    ax.tick_params(colors="#8a8578", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#e8e5de")
    ax.grid(color="#e8e5de", alpha=0.5, linewidth=0.6)
    st.pyplot(fig)

    st.caption("Red bands = detected anomaly periods (≥ 1 hour). Blue dashed line = detection threshold.")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---- WHAT THE MODEL FOUND ----
    st.markdown('''
        <div class="sect">
            <div class="sect-num">05</div>
            <div class="sect-title">What the model found</div>
        </div>
    ''', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('''
            <div class="status status-warn">
            <b>Most important finding.</b> 17 June 2022 — approximately 24 hours
            before the first shutdown — axial displacement reached +8 standard
            deviations above normal. This is a classic pre-failure signature of
            thrust bearing wear.
            </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown('''
            <div class="status status-info">
            <b>Second finding.</b> 28 April 2022 — pressure_1 dropped -36 standard
            deviations for one hour. Not gradual drift. A discrete event,
            consistent with a valve action or filter change.
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-ok">
        <b>What this is not.</b> The system does not prove causality. It ranks
        the variables that deviated most and categorises the pattern. The engineer
        interprets the pattern using chemical engineering knowledge.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---- LIMITATIONS ----
    st.markdown('''
        <div class="sect">
            <div class="sect-num">06</div>
            <div class="sect-title">Limitations</div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-bone">
        <b>No labelled fault data.</b> The dataset contains no marked failures.
        Precision, recall, and F1 cannot be computed. Evaluation is qualitative
        and physical — every flagged period must be explained by an engineer.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-bone">
        <b>Sensor identities are partly inferred.</b> The P&ID was not released
        with the dataset, so the exact physical meaning of each temperature and
        pressure tag is not confirmed. Tag type (ZI, PI, TI, XI, SI) is known;
        specific process connection is inferred.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-bone">
        <b>No causal claims.</b> The system identifies leading contributors,
        not causes. A high z-score means "this variable changed more than
        expected" — not "this variable caused the anomaly."
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-bone">
        <b>Model retraining required over time.</b> As the machine ages, the
        definition of "normal" shifts. The model should be retrained periodically
        on recent steady-state data to stay calibrated.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-bone">
        <b>Not a replacement for the operator.</b> SANKET is a decision-support
        tool. It gives the engineer evidence to investigate. The engineer makes
        the decision.
        </div>
    ''', unsafe_allow_html=True)


# ============================================================
# TAB 2 — PROCESS TRENDS
# ============================================================
with tab_trends:
    st.markdown('''
        <div class="sect">
            <div class="sect-num">07</div>
            <div class="sect-title">Sensor behaviour over the year</div>
        </div>
    ''', unsafe_allow_html=True)

    st.write("Select a sensor group to view its full-year trend. Red bands mark detected anomaly periods.")

    st.markdown("<br>", unsafe_allow_html=True)

    group = st.selectbox(
        "Sensor group",
        ["Axial Displacement", "Vibration", "Temperature", "Pressure", "Speed"],
        label_visibility="collapsed",
        key="trends_sensor_group",
    )

    sensor_lookup = {
        "Axial Displacement": ["axial_1", "axial_2", "axial_3", "axial_4"],
        "Vibration": ["vibration_1", "vibration_2", "vibration_3", "vibration_4"],
        "Temperature": ["temp_1", "temp_2", "temp_3", "temp_4", "temp_5",
                        "temp_6", "temp_7", "temp_8", "temp_9", "temp_10", "temp_11"],
        "Pressure": ["pressure_1", "pressure_2", "pressure_3", "pressure_4",
                     "pressure_diff"],
        "Speed": ["speed"],
    }

    for col in sensor_lookup[group]:
        fig, ax = plt.subplots(figsize=(13, 1.9))
        fig.patch.set_facecolor("#f7f7f5")
        ax.set_facecolor("#ffffff")
        ax.plot(full_df["Timestamp"], full_df[col], linewidth=0.55, color="#323232")
        for _, row in periods_df.iterrows():
            ax.axvspan(row["start"], row["end"], color="#EE3124", alpha=0.10)
        ax.set_ylabel(col, fontsize=8, color="#4a4438")
        ax.tick_params(colors="#8a8578", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#e8e5de")
        ax.grid(color="#e8e5de", alpha=0.4, linewidth=0.5)
        st.pyplot(fig)


# ============================================================
# TAB 3 — ANOMALY PERIODS & ACTIONS
# ============================================================
with tab_periods:
    st.markdown('''
        <div class="sect">
            <div class="sect-num">08</div>
            <div class="sect-title">Detected anomaly periods & suggested actions</div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-info">
        Each period below was flagged as anomalous and lasted more than 1 hour.
        Category and action come from rule-based decision support, built on
        chemical engineering reasoning — not from a black-box model.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    high = int((periods_df["priority"] == "High").sum())
    med = int((periods_df["priority"] == "Medium").sum())
    low = int((periods_df["priority"] == "Low").sum())

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(f'''
            <div class="card">
                <div class="card-label">High priority</div>
                <div class="card-value">{high}</div>
                <div class="card-note">Investigate immediately</div>
            </div>
        ''', unsafe_allow_html=True)
    with p2:
        st.markdown(f'''
            <div class="card blue">
                <div class="card-label">Medium priority</div>
                <div class="card-value">{med}</div>
                <div class="card-note">Review when convenient</div>
            </div>
        ''', unsafe_allow_html=True)
    with p3:
        st.markdown(f'''
            <div class="card">
                <div class="card-label">Low priority</div>
                <div class="card-value">{low}</div>
                <div class="card-note">Informational</div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    if high > 0:
        st.markdown('''<div class="sect" style="border:none; padding:0; margin-bottom:1rem;">
                       <div class="sect-title" style="font-size:1.1rem;">High-priority periods</div>
                       </div>''', unsafe_allow_html=True)
        high_df = periods_df[periods_df["priority"] == "High"]
        for _, row in high_df.iterrows():
            st.markdown(f'''
                <div class="status status-warn">
                <b>Period {int(row['period_id'])} — {row['category']}.</b>
                {row['start'].strftime('%d %b %Y, %H:%M')} to {row['end'].strftime('%d %b %Y, %H:%M')}
                · {row['duration_hours']:.1f} hours.<br>
                {row['suggested_action']}
                </div>
            ''', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('''<div class="sect" style="border:none; padding:0; margin-bottom:1rem;">
                   <div class="sect-title" style="font-size:1.1rem;">All anomaly periods</div>
                   </div>''', unsafe_allow_html=True)

    table = periods_df[[
        "period_id", "start", "end", "duration_hours",
        "category", "priority",
        "group_axial", "group_vibration", "group_temperature",
        "group_pressure", "group_speed",
    ]].copy()
    table.columns = [
        "ID", "Start", "End", "Duration (h)",
        "Category", "Priority",
        "Axial z", "Vibration z", "Temperature z",
        "Pressure z", "Speed z",
    ]
    table = table.round(2)
    st.dataframe(table, use_container_width=True, hide_index=True)


# ============================================================
# TAB 4 — OPERATOR DECISION SUPPORT
# ============================================================
with tab_support:
    st.markdown('''
        <div class="sect">
            <div class="sect-num">09</div>
            <div class="sect-title">Operator decision support</div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="status status-info">
        Select a detected period to view its physical category, suggested action,
        group-level z-scores, contributing sensors, and the sensor traces around
        the event. Everything here is meant to help an engineer decide what to
        investigate first.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    labels = [
        f"P{int(row['period_id'])} · {row['start'].strftime('%d %b %H:%M')} · {row['category']}"
        for _, row in periods_df.iterrows()
    ]
    selected_index = st.selectbox(
        "Select a period",
        range(len(periods_df)),
        format_func=lambda i: labels[i],
        label_visibility="collapsed",
        key="support_period_selector",
    )
    selected_row = periods_df.iloc[selected_index]

    priority_class = {
        "High": "status-warn",
        "Medium": "status-info",
        "Low": "status-ok",
    }.get(selected_row["priority"], "status-info")

    st.markdown(f'''
        <div class="status {priority_class}">
        <b>{selected_row['category']} · {selected_row['priority']} priority.</b><br>
        {selected_row['start'].strftime('%d %b %Y, %H:%M')} to
        {selected_row['end'].strftime('%d %b %Y, %H:%M')} ·
        {selected_row['duration_hours']:.1f} hours<br><br>
        {selected_row['suggested_action']}
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Group z-scores
    st.markdown('''<div class="sect" style="border:none; padding:0; margin-bottom:1rem;">
                   <div class="sect-title" style="font-size:1.1rem;">Group z-scores</div>
                   </div>''', unsafe_allow_html=True)

    groups = ["axial", "vibration", "temperature", "pressure", "speed"]
    g_cols = st.columns(5)
    for i, g in enumerate(groups):
        with g_cols[i]:
            val = selected_row[f"group_{g}"]
            sign = "+" if val >= 0 else ""
            card_class = "card" if i % 2 == 0 else "card blue"
            st.markdown(f'''
                <div class="{card_class}">
                    <div class="card-label">{g}</div>
                    <div class="card-value small">{sign}{val:.2f}</div>
                    <div class="card-note">σ from normal</div>
                </div>
            ''', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Top contributors
    st.markdown('''<div class="sect" style="border:none; padding:0; margin-bottom:1rem;">
                   <div class="sect-title" style="font-size:1.1rem;">Top contributing sensors</div>
                   </div>''', unsafe_allow_html=True)

    c_row = contrib_df[contrib_df["period_id"] == selected_row["period_id"]]
    if len(c_row) > 0:
        c_row = c_row.iloc[0]
        contrib_rows = []
        for rank in [1, 2, 3, 4, 5]:
            contrib_rows.append({
                "Sensor": c_row[f"top_{rank}"],
                "z-score": round(c_row[f"top_{rank}_z"], 2),
            })
        st.dataframe(pd.DataFrame(contrib_rows), hide_index=True,
                     use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Sensor traces
    st.markdown('''<div class="sect" style="border:none; padding:0; margin-bottom:1rem;">
                   <div class="sect-title" style="font-size:1.1rem;">Sensor traces (±2 days)</div>
                   </div>''', unsafe_allow_html=True)

    start = selected_row["start"]
    end = selected_row["end"]
    window_start = start - pd.Timedelta("2 days")
    window_end = end + pd.Timedelta("2 days")
    mask = (full_df["Timestamp"] >= window_start) & (full_df["Timestamp"] <= window_end)
    window_df = full_df[mask]

    group_plot = st.selectbox(
        "Sensor group",
        ["Axial Displacement", "Vibration", "Temperature", "Pressure", "Speed"],
        label_visibility="collapsed",
        key="support_sensor_group",
    )

    plot_cols = sensor_lookup[group_plot]
    for col in plot_cols:
        fig, ax = plt.subplots(figsize=(12, 1.9))
        fig.patch.set_facecolor("#f7f7f5")
        ax.set_facecolor("#ffffff")
        ax.plot(window_df["Timestamp"], window_df[col], linewidth=0.7, color="#323232")
        ax.axvspan(start, end, color="#EE3124", alpha=0.15)
        ax.set_ylabel(col, fontsize=8, color="#4a4438")
        ax.tick_params(colors="#8a8578", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#e8e5de")
        ax.grid(color="#e8e5de", alpha=0.4, linewidth=0.5)
        st.pyplot(fig)

    st.caption("Red band = the selected anomaly period.")


# ============================================================
# FOOTER
# ============================================================
st.markdown('''
<div class="foot">
    <div>Sanket · Compressor Anomaly Detection</div>
    <div>Isolation Forest · Motor Oil Hellas · 2022</div>
</div>
''', unsafe_allow_html=True)