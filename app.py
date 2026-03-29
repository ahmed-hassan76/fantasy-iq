from __future__ import annotations

import time
import streamlit as st
import pandas as pd

from src.predict import build_predictions_table
from src.optimizer import build_optimized_squad_from_predictions, summarize_squad
from src.gw1_builder import build_gw1_hybrid_outputs
from src.transfer_logic import (
    recommend_best_one_transfer,
    recommend_best_two_transfers,
    validate_full_squad,
    validate_starting_xi,
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Fantasy IQ",
    page_icon="⚽",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #2b0038 0%, #43004d 28%, #24002d 100%);
            color: white;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #24002d 0%, #3c0048 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        h1, h2, h3, h4 {
            color: white !important;
            font-weight: 800 !important;
        }

        p, li, div, label, span {
            color: #f7ecff;
        }

        [data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.02);
        }

        [data-testid="stDataFrame"] > div {
            background: rgba(255,255,255,0.03) !important;
            border-radius: 18px !important;
        }

        [data-testid="stDataFrame"] [role="grid"] {
            background: rgba(255,255,255,0.02) !important;
            color: white !important;
        }

        [data-testid="stDataFrame"] div[role="columnheader"] {
            background: rgba(255,255,255,0.08) !important;
            color: white !important;
            font-weight: 800 !important;
            border-bottom: 1px solid rgba(255,255,255,0.08) !important;
        }

        [data-testid="stDataFrame"] div[role="gridcell"] {
            background: transparent !important;
            color: #f7ecff !important;
            border-bottom: 1px solid rgba(255,255,255,0.04) !important;
        }

        [data-testid="stDataFrame"] div[role="row"]:hover div[role="gridcell"] {
            background: rgba(255,255,255,0.05) !important;
        }

        [data-testid="stTable"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="popover"] > div {
            background: rgba(255,255,255,0.06) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            color: white !important;
        }

        .stButton > button {
            width: 100%;
            border-radius: 18px;
            border: none;
            color: white;
            font-weight: 700;
            padding: 0.8rem 1rem;
            background: linear-gradient(90deg, #176baf 0%, #5d2ca5 100%);
            box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #1a83d0 0%, #7a39d1 100%);
            color: white;
        }

        div[role="radiogroup"] {
            background: rgba(255,255,255,0.04);
            padding: 10px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
        }

        [data-testid="metric-container"] {
            background: linear-gradient(135deg, rgba(0,196,255,0.14) 0%, rgba(147,51,234,0.16) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 16px;
            border-radius: 18px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        }

        [data-testid="metric-container"] label {
            color: #f7ecff !important;
            font-weight: 700 !important;
        }

        .hero-card {
            background: linear-gradient(120deg, #19c3df 0%, #4b5de6 45%, #8f2cff 100%);
            border-radius: 28px;
            padding: 24px 28px;
            color: white;
            box-shadow: 0 14px 36px rgba(0,0,0,0.25);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: 2rem;
            font-weight: 900;
            margin-bottom: 0.25rem;
            color: white;
        }

        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.95;
            margin-bottom: 0.25rem;
            color: white;
        }

        .section-card {
            background: rgba(255,255,255,0.035);
            border-radius: 22px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }

        .comparison-banner {
            background: linear-gradient(90deg, rgba(0,196,255,0.15), rgba(143,44,255,0.20));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 16px 18px;
            font-weight: 700;
            color: white;
            margin-bottom: 1rem;
        }

        .small-muted {
            color: #ead7ff;
            opacity: 0.9;
            font-size: 0.95rem;
        }

        span[data-baseweb="tag"] {
            background: rgba(255,255,255,0.10) !important;
            color: white !important;
            border-radius: 999px !important;
        }

        .pitch-wrapper {
            background: linear-gradient(180deg, #16a34a 0%, #15803d 100%);
            border-radius: 28px;
            padding: 22px 18px 18px 18px;
            border: 3px solid rgba(255,255,255,0.18);
            box-shadow: 0 14px 36px rgba(0,0,0,0.28);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .pitch-title {
            text-align: center;
            color: white;
            font-size: 1.2rem;
            font-weight: 800;
            margin-bottom: 12px;
        }

        .pitch-row {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
            margin: 14px 0;
        }

        .pitch-divider {
            width: 100%;
            height: 2px;
            background: rgba(255,255,255,0.28);
            margin: 10px 0;
        }

        .player-card {
            width: 128px;
            min-height: 96px;
            background: #f8f5ff;
            border-radius: 14px;
            padding: 10px 8px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0,0,0,0.18);
            border: 2px solid rgba(43,0,56,0.10);
        }

        .player-card-name {
            color: #2b0038 !important;
            font-weight: 800 !important;
            font-size: 0.92rem !important;
            line-height: 1.1 !important;
            margin-bottom: 5px !important;
            word-break: break-word;
        }

        .player-card-meta {
            color: #5b267d !important;
            font-size: 0.80rem !important;
            font-weight: 700 !important;
            margin-bottom: 4px !important;
        }

        .player-card-points {
            color: #176baf !important;
            font-size: 0.82rem !important;
            font-weight: 800 !important;
        }

        .bench-wrapper {
            background: rgba(255,255,255,0.05);
            border-radius: 22px;
            padding: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            margin-top: 1rem;
        }

        .bench-title {
            color: white;
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 10px;
            text-align: center;
        }

        /* Visibility fixes */
        div[data-baseweb="select"] * {
            color: white !important;
        }

        div[data-baseweb="popover"] * {
            color: white !important;
            background-color: #3a0046 !important;
        }

        div[data-baseweb="select"] input {
            color: white !important;
            -webkit-text-fill-color: white !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            color: white !important;
            -webkit-text-fill-color: white !important;
        }

        div[data-baseweb="select"] span {
            color: white !important;
        }

        div[data-baseweb="menu"] * {
            color: white !important;
            background-color: #3a0046 !important;
        }

        .stSlider label,
        .stNumberInput label,
        .stSelectbox label,
        .stMultiSelect label,
        .stRadio label {
            color: #f7ecff !important;
        }

        /* Dropdown option visibility and keyboard highlight */
        div[data-baseweb="menu"] {
            background: #3a0046 !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
        }

        div[role="option"] {
            background: #3a0046 !important;
            color: white !important;
        }

        div[role="option"]:hover {
            background: #5d2ca5 !important;
            color: white !important;
        }

        div[role="option"][aria-selected="true"] {
            background: #176baf !important;
            color: white !important;
            font-weight: 700 !important;
        }

        /* Keyboard-highlighted option */
        li[aria-selected="true"] {
            background: #176baf !important;
            color: white !important;
        }

        li:hover {
            background: #5d2ca5 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def hero_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_box_title(title: str, subtitle: str | None = None):
    if subtitle:
        st.markdown(
            f"""
            <div class="section-card">
                <div style="font-size:1.3rem;font-weight:800;color:white;">{title}</div>
                <div class="small-muted">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="section-card">
                <div style="font-size:1.3rem;font-weight:800;color:white;">{title}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


inject_custom_css()

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_predictions_cached() -> pd.DataFrame:
    return build_predictions_table(use_cache=True, verbose=False)


@st.cache_data(ttl=1800, show_spinner=False)
def load_gw1_hybrid_outputs_cached(
    include_unmatched: bool = True,
    unmatched_penalty: float = 0.85,
):
    return build_gw1_hybrid_outputs(
        use_cache=True,
        include_unmatched=include_unmatched,
        unmatched_penalty=unmatched_penalty,
        progress_callback=None,
        verbose=False,
    )

# -----------------------------
# Loading UI helpers
# -----------------------------
def load_predictions_with_ui() -> pd.DataFrame:
    if "predictions_df" in st.session_state and st.session_state["predictions_df"] is not None:
        return st.session_state["predictions_df"]

    progress_bar = st.progress(0, text="Starting live prediction pipeline...")
    status_box = st.empty()
    start_time = time.time()

    status_box.info("First live load may take around 3 to 5 minutes depending on API response time.")

    def streamlit_progress_callback(value: float, message: str) -> None:
        elapsed = time.time() - start_time

        if value > 0.02:
            estimated_total = elapsed / value
            eta_seconds = max(0, estimated_total - elapsed)
            eta_text = f" | ETA: ~{eta_seconds:.0f}s"
        else:
            eta_text = ""

        progress_bar.progress(int(value * 100), text=f"{message}{eta_text}")
        status_box.info(f"{message} | Elapsed: {elapsed:.1f}s{eta_text}")

    df = build_predictions_table(
        use_cache=True,
        verbose=False,
        progress_callback=streamlit_progress_callback,
    )

    elapsed = time.time() - start_time
    progress_bar.progress(100, text="Live predictions ready.")
    status_box.success(f"Live predictions loaded successfully in {elapsed:.1f} seconds.")

    st.session_state["predictions_df"] = df
    return df


def load_optimized_squad_with_ui() -> pd.DataFrame:
    if "optimized_squad_df" in st.session_state and st.session_state["optimized_squad_df"] is not None:
        return st.session_state["optimized_squad_df"]

    progress = st.progress(0, text="Preparing optimized squad...")
    status_box = st.empty()
    start_time = time.time()

    status_box.info("Building the optimized squad from the current live prediction table.")
    progress.progress(10, text="Loading live predictions...")

    predictions_df = load_predictions_with_ui()

    progress.progress(65, text="Running squad optimization...")
    squad_df = build_optimized_squad_from_predictions(
        predictions_df=predictions_df,
        verbose=False,
    )

    elapsed = time.time() - start_time
    progress.progress(100, text="Optimized squad ready.")
    status_box.success(f"Optimized squad loaded successfully in {elapsed:.1f} seconds.")

    st.session_state["optimized_squad_df"] = squad_df
    return squad_df


def load_gw1_hybrid_outputs_with_ui(
    include_unmatched: bool = True,
    unmatched_penalty: float = 0.85,
) -> dict:
    session_key = f"gw1_outputs_{include_unmatched}_{unmatched_penalty}"

    if session_key in st.session_state and st.session_state[session_key] is not None:
        return st.session_state[session_key]

    progress_bar = st.progress(0, text="Starting GW1 Squad Builder...")
    status_box = st.empty()
    start_time = time.time()

    status_box.info("Building the automated GW1 squad using the previous season summary and the live current player pool.")

    def streamlit_progress_callback(value: float, message: str) -> None:
        elapsed = time.time() - start_time

        if value > 0.02:
            estimated_total = elapsed / value
            eta_seconds = max(0, estimated_total - elapsed)
            eta_text = f" | ETA: ~{eta_seconds:.0f}s"
        else:
            eta_text = ""

        progress_bar.progress(int(value * 100), text=f"{message}{eta_text}")
        status_box.info(f"{message} | Elapsed: {elapsed:.1f}s{eta_text}")

    outputs = build_gw1_hybrid_outputs(
        use_cache=True,
        include_unmatched=include_unmatched,
        unmatched_penalty=unmatched_penalty,
        progress_callback=streamlit_progress_callback,
        verbose=False,
    )

    elapsed = time.time() - start_time
    progress_bar.progress(100, text="GW1 Squad Builder ready.")
    status_box.success(f"GW1 Squad Builder loaded successfully in {elapsed:.1f} seconds.")

    st.session_state[session_key] = outputs
    return outputs

# -----------------------------
# Formatting helpers
# -----------------------------
def format_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    rename_map = {
        "name": "Name",
        "team": "Team",
        "position": "Position",
        "price_m": "Price",
        "predicted_points": "Predicted Points",
        "model_used": "Model Used",
        "source_round": "Latest Completed Round",
    }
    temp = temp.rename(columns=rename_map)

    if "Price" in temp.columns:
        temp["Price"] = pd.to_numeric(temp["Price"], errors="coerce").round(1)
    if "Predicted Points" in temp.columns:
        temp["Predicted Points"] = pd.to_numeric(temp["Predicted Points"], errors="coerce").round(2)

    preferred_cols = [
        "Name",
        "Team",
        "Position",
        "Price",
        "Predicted Points",
        "Model Used",
        "Latest Completed Round",
    ]
    existing_cols = [col for col in preferred_cols if col in temp.columns]
    return temp[existing_cols]


def format_squad_table(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    rename_map = {
        "name": "Name",
        "team": "Team",
        "position": "Position",
        "price_m": "Price",
        "predicted_points": "Predicted Points",
        "model_used": "Model Used",
    }
    temp = temp.rename(columns=rename_map)

    if "Price" in temp.columns:
        temp["Price"] = pd.to_numeric(temp["Price"], errors="coerce").round(1)
    if "Predicted Points" in temp.columns:
        temp["Predicted Points"] = pd.to_numeric(temp["Predicted Points"], errors="coerce").round(2)

    preferred_cols = ["Name", "Team", "Position", "Price", "Predicted Points", "Model Used"]
    existing_cols = [col for col in preferred_cols if col in temp.columns]

    return temp[existing_cols].sort_values(
        ["Position", "Predicted Points"],
        ascending=[True, False]
    )


def format_one_transfer_table(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    rename_map = {
        "player_out": "Player Out",
        "player_out_team": "Out Team",
        "player_out_position": "Out Position",
        "player_out_price": "Out Price",
        "player_out_predicted_points": "Out Predicted Points",
        "player_in": "Player In",
        "player_in_team": "In Team",
        "player_in_position": "In Position",
        "player_in_price": "In Price",
        "player_in_predicted_points": "In Predicted Points",
        "predicted_points_gain": "Predicted Points Gain",
        "budget_change": "Budget Change",
        "remaining_money_in_bank": "Remaining Money In Bank",
        "outgoing_is_starter": "Outgoing Is Starter",
    }
    temp = temp.rename(columns=rename_map)

    for col in [
        "Out Price",
        "Out Predicted Points",
        "In Price",
        "In Predicted Points",
        "Predicted Points Gain",
        "Budget Change",
        "Remaining Money In Bank",
    ]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce").round(2)

    return temp


def format_two_transfer_table(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    rename_map = {
        "player_out_1": "Player Out 1",
        "player_out_2": "Player Out 2",
        "player_out_positions": "Out Positions",
        "player_in_1": "Player In 1",
        "player_in_2": "Player In 2",
        "player_in_positions": "In Positions",
        "predicted_points_gain": "Predicted Points Gain",
        "budget_change": "Budget Change",
        "remaining_money_in_bank": "Remaining Money In Bank",
        "outgoing_1_is_starter": "Outgoing 1 Is Starter",
        "outgoing_2_is_starter": "Outgoing 2 Is Starter",
    }
    temp = temp.rename(columns=rename_map)

    for col in [
        "Predicted Points Gain",
        "Budget Change",
        "Remaining Money In Bank",
    ]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce").round(2)

    return temp


def format_gw1_squad_table(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    rename_map = {
        "name": "Name",
        "team": "Team",
        "position": "Position",
        "price_m": "Price",
        "hybrid_score": "Hybrid Score",
        "matched_previous_season": "Matched Previous Season",
        "total_points_sum": "Previous Season Total Points",
        "total_points_avg": "Previous Season Avg Points",
        "value_efficiency": "Value Efficiency",
        "start_rate": "Start Rate",
        "minutes_per_appearance": "Minutes Per Appearance",
    }
    temp = temp.rename(columns=rename_map)

    if "Matched Previous Season" in temp.columns:
        temp["Matched Previous Season"] = temp["Matched Previous Season"].map({1: "Yes", 0: "No"})

    for col in [
        "Price",
        "Hybrid Score",
        "Previous Season Avg Points",
        "Value Efficiency",
        "Start Rate",
        "Minutes Per Appearance",
    ]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce").round(2)

    if "Previous Season Total Points" in temp.columns:
        temp["Previous Season Total Points"] = pd.to_numeric(
            temp["Previous Season Total Points"], errors="coerce"
        ).round(0)

    preferred_cols = [
        "Name",
        "Team",
        "Position",
        "Price",
        "Hybrid Score",
        "Matched Previous Season",
        "Previous Season Total Points",
        "Previous Season Avg Points",
        "Value Efficiency",
        "Start Rate",
        "Minutes Per Appearance",
    ]

    existing_cols = [col for col in preferred_cols if col in temp.columns]
    return temp[existing_cols].sort_values(
        ["Position", "Hybrid Score"],
        ascending=[True, False]
    )


def style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    styled = (
        df.style
        .set_properties(**{
            "background-color": "#2f0a3a",
            "color": "#f7ecff",
            "border-color": "rgba(255,255,255,0.08)",
            "font-size": "14px",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#4b1e5a"),
                    ("color", "white"),
                    ("font-weight", "800"),
                    ("border", "1px solid rgba(255,255,255,0.08)"),
                    ("text-align", "left"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid rgba(255,255,255,0.05)"),
                    ("padding", "8px"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("width", "100%"),
                    ("border-radius", "14px"),
                    ("overflow", "hidden"),
                ],
            },
        ])
    )
    return styled


def player_label(row: pd.Series) -> str:
    return f"{row['name']} | {row['position']} | {row['team']} | £{row['price_m']:.1f}m"


def short_name(name: str, max_len: int = 16) -> str:
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."


def build_player_card_html(row: pd.Series) -> str:
    name = short_name(str(row["name"]))
    team = str(row["team"])
    price = float(row["price_m"])

    if "predicted_points" in row.index and pd.notna(row["predicted_points"]):
        score_text = f"{float(row['predicted_points']):.2f} pts"
    elif "hybrid_score" in row.index and pd.notna(row["hybrid_score"]):
        score_text = f"Hybrid: {float(row['hybrid_score']):.2f}"
    else:
        score_text = ""

    return f"""
    <div class="player-card">
        <div class="player-card-name">{name}</div>
        <div class="player-card-meta">{team} • £{price:.1f}m</div>
        <div class="player-card-points">{score_text}</div>
    </div>
    """.strip()


def render_pitch(starting_df: pd.DataFrame, bench_df: pd.DataFrame) -> None:
    gk_df = starting_df[starting_df["position"] == "GK"]
    def_df = starting_df[starting_df["position"] == "DEF"]
    mid_df = starting_df[starting_df["position"] == "MID"]
    fwd_df = starting_df[starting_df["position"] == "FWD"]

    def row_html(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        cards = "".join(build_player_card_html(row) for _, row in df.iterrows())
        return f'<div class="pitch-row">{cards}</div>'

    gk_html = row_html(gk_df)
    def_html = row_html(def_df)
    mid_html = row_html(mid_df)
    fwd_html = row_html(fwd_df)
    bench_html = row_html(bench_df)

    pitch_html = f"""
    <div class="pitch-wrapper">
        <div class="pitch-title">Starting XI</div>
        {gk_html}
        <div class="pitch-divider"></div>
        {def_html}
        <div class="pitch-divider"></div>
        {mid_html}
        <div class="pitch-divider"></div>
        {fwd_html}
    </div>

    <div class="bench-wrapper">
        <div class="bench-title">Bench</div>
        {bench_html}
    </div>
    """

    st.markdown(pitch_html, unsafe_allow_html=True)


def build_best_current_starting_xi(squad_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Build the best valid starting XI from the optimized 15-player squad
    by checking all standard FPL-valid formations and choosing the one
    with the highest total predicted points.
    """
    if squad_df.empty:
        return pd.DataFrame(), pd.DataFrame(), ""

    gk_pool = squad_df[squad_df["position"] == "GK"].sort_values("predicted_points", ascending=False)
    def_pool = squad_df[squad_df["position"] == "DEF"].sort_values("predicted_points", ascending=False)
    mid_pool = squad_df[squad_df["position"] == "MID"].sort_values("predicted_points", ascending=False)
    fwd_pool = squad_df[squad_df["position"] == "FWD"].sort_values("predicted_points", ascending=False)

    valid_formations = [
        (3, 4, 3),
        (3, 5, 2),
        (4, 4, 2),
        (4, 3, 3),
        (4, 5, 1),
        (5, 4, 1),
        (5, 3, 2),
        (5, 2, 3),
    ]

    best_starting_df = pd.DataFrame()
    best_bench_df = pd.DataFrame()
    best_total = float("-inf")
    best_formation = ""

    for def_count, mid_count, fwd_count in valid_formations:
        gk_df = gk_pool.head(1)
        def_df = def_pool.head(def_count)
        mid_df = mid_pool.head(mid_count)
        fwd_df = fwd_pool.head(fwd_count)

        starting_df = pd.concat([gk_df, def_df, mid_df, fwd_df], ignore_index=True)

        if len(starting_df) != 11:
            continue

        total_points = starting_df["predicted_points"].sum()

        if total_points > best_total:
            best_total = total_points
            best_starting_df = starting_df.copy().reset_index(drop=True)
            best_bench_df = squad_df[~squad_df["name"].isin(best_starting_df["name"])].copy().reset_index(drop=True)
            best_formation = f"{def_count}-{mid_count}-{fwd_count}"

    return best_starting_df, best_bench_df, best_formation

def build_best_gw1_starting_xi(squad_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Build the best valid starting XI from the GW1 hybrid squad
    by checking standard FPL-valid formations and choosing the one
    with the highest total hybrid score.
    """
    if squad_df.empty:
        return pd.DataFrame(), pd.DataFrame(), ""

    gk_pool = squad_df[squad_df["position"] == "GK"].sort_values("hybrid_score", ascending=False)
    def_pool = squad_df[squad_df["position"] == "DEF"].sort_values("hybrid_score", ascending=False)
    mid_pool = squad_df[squad_df["position"] == "MID"].sort_values("hybrid_score", ascending=False)
    fwd_pool = squad_df[squad_df["position"] == "FWD"].sort_values("hybrid_score", ascending=False)

    valid_formations = [
        (3, 4, 3),
        (3, 5, 2),
        (4, 4, 2),
        (4, 3, 3),
        (4, 5, 1),
        (5, 4, 1),
        (5, 3, 2),
        (5, 2, 3),
    ]

    best_starting_df = pd.DataFrame()
    best_bench_df = pd.DataFrame()
    best_total = float("-inf")
    best_formation = ""

    for def_count, mid_count, fwd_count in valid_formations:
        gk_df = gk_pool.head(1)
        def_df = def_pool.head(def_count)
        mid_df = mid_pool.head(mid_count)
        fwd_df = fwd_pool.head(fwd_count)

        starting_df = pd.concat([gk_df, def_df, mid_df, fwd_df], ignore_index=True)

        if len(starting_df) != 11:
            continue

        total_score = starting_df["hybrid_score"].sum()

        if total_score > best_total:
            best_total = total_score
            best_starting_df = starting_df.copy().reset_index(drop=True)
            best_bench_df = squad_df[~squad_df["name"].isin(best_starting_df["name"])].copy().reset_index(drop=True)
            best_formation = f"{def_count}-{mid_count}-{fwd_count}"

    return best_starting_df, best_bench_df, best_formation

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown(
    """
    <div style="font-size:1.35rem;font-weight:900;color:white;margin-bottom:0.5rem;">
        Fantasy IQ
    </div>
    <div style="color:#e9d7ff;font-size:0.95rem;margin-bottom:1rem;">
        Live FPL Decision Support
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Player Prediction Engine",
        "Squad Builder",
        "Transfer Assistant",
    ]
)

st.sidebar.markdown("---")

if st.sidebar.button("Refresh Live Data"):
    st.cache_data.clear()
    st.session_state["predictions_df"] = None
    st.session_state["optimized_squad_df"] = None

    keys_to_clear = [key for key in st.session_state.keys() if key.startswith("gw1_outputs_")]
    for key in keys_to_clear:
        st.session_state[key] = None

    st.sidebar.success("Cache cleared. The next load will pull fresh live data.")

# -----------------------------
# Home
# -----------------------------
if page == "Home":
    hero_header(
        "Fantasy IQ",
        "Live FPL decision support powered by official FPL data and Machine Learning"
    )

    section_box_title(
        "What Fantasy IQ Does",
        "Fantasy IQ pulls live data, rebuilds the feature pipeline, loads the final models to support predictions, squad building, and transfer planning"
    )

    st.markdown('<div class="comparison-banner">Modules</div>', unsafe_allow_html=True)
    st.write("• Player Prediction Engine")
    st.write("• Squad Builder")
    st.write("• Transfer Assistant")

    st.info("The first live load can take a few minutes. After that, cache makes repeated loads much faster")

# -----------------------------
# Player Prediction Engine
# -----------------------------
elif page == "Player Prediction Engine":
    hero_header(
        "Player Prediction Engine",
        "View live predicted points for players"
    )

    predictions_df = load_predictions_with_ui()

    if predictions_df.empty:
        st.error("No prediction data was returned.")
        st.stop()

    section_box_title("Filters", "Filter players by position, team, price, and predicted points")

    col1, col2, col3 = st.columns(3)

    with col1:
        position_options = ["All"] + sorted(predictions_df["position"].dropna().unique().tolist())
        selected_position = st.selectbox("Position", position_options)

    with col2:
        team_options = ["All"] + sorted(predictions_df["team"].dropna().unique().tolist())
        selected_team = st.selectbox("Team", team_options)

    with col3:
        sort_order = st.selectbox("Sort by Predicted Points", ["Descending", "Ascending"])

    min_price = float(predictions_df["price_m"].min())
    max_price = float(predictions_df["price_m"].max())

    selected_price_range = st.slider(
        "Price Range",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=0.1,
    )

    filtered_df = predictions_df.copy()

    if selected_position != "All":
        filtered_df = filtered_df[filtered_df["position"] == selected_position]

    if selected_team != "All":
        filtered_df = filtered_df[filtered_df["team"] == selected_team]

    filtered_df = filtered_df[
        (filtered_df["price_m"] >= selected_price_range[0])
        & (filtered_df["price_m"] <= selected_price_range[1])
    ]

    ascending_flag = sort_order == "Ascending"
    filtered_df = filtered_df.sort_values("predicted_points", ascending=ascending_flag).reset_index(drop=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players Shown", len(filtered_df))
    c2.metric("Top Prediction", f"{filtered_df['predicted_points'].max():.2f}" if not filtered_df.empty else "0.00")
    c3.metric("Average Prediction", f"{filtered_df['predicted_points'].mean():.2f}" if not filtered_df.empty else "0.00")
    c4.metric("Teams Shown", int(filtered_df["team"].nunique()) if not filtered_df.empty else 0)

    st.markdown('<div class="comparison-banner">Prediction Results</div>', unsafe_allow_html=True)
    st.dataframe(
        style_table(format_prediction_table(filtered_df)),
        use_container_width=True
    )

    st.markdown("### Top 10 Predicted Players")
    st.dataframe(
        style_table(format_prediction_table(filtered_df.head(10))),
        use_container_width=True
    )

# -----------------------------
# Squad Builder
# -----------------------------
elif page == "Squad Builder":
    page_container = st.empty()

    with page_container.container():
        hero_header(
            "Squad Builder",
            "Show the best current squad from live predictions or View an automated GW1 squad to start the season with an advantage"
        )

        tab1, tab2 = st.tabs(["Best Current Squad", "GW1 Squad Builder"])

        with tab1:
            st.markdown(
                '<div class="comparison-banner">Best Current Squad</div>',
                unsafe_allow_html=True
            )

            st.write(
                "This section builds the best possible 15-player squad right now using the live prediction engine and follows FPL constraints"
            )

            squad_df = load_optimized_squad_with_ui()

            if squad_df.empty:
                st.error("Could not build the current optimized squad.")
            else:
                summary = summarize_squad(squad_df)

                c1, c2, c3 = st.columns(3)
                c1.metric("Players", summary["players"])
                c2.metric("Total Cost", f"{summary['total_cost']:.1f}")
                c3.metric("Total Predicted Points", f"{summary['total_predicted_points']:.2f}")

                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("### Position Counts")
                    pos_counts_df = pd.DataFrame.from_dict(
                        summary["position_counts"], orient="index", columns=["Count"]
                    )
                    st.dataframe(style_table(pos_counts_df), use_container_width=True)

                with col_right:
                    st.markdown("### Players Per Club")
                    club_counts_df = pd.DataFrame.from_dict(
                        summary["club_counts"], orient="index", columns=["Count"]
                    )
                    st.dataframe(style_table(club_counts_df), use_container_width=True)

                st.markdown(
                    '<div class="comparison-banner">Optimized 15-Player Current Squad</div>',
                    unsafe_allow_html=True
                )

                starting_df, bench_df, formation = build_best_current_starting_xi(squad_df)

                if not starting_df.empty:
                 st.markdown(f"### Best Current Starting XI ({formation})")
                render_pitch(starting_df, bench_df)

                st.markdown("### Full 15-Player Squad")
                st.dataframe(
                    style_table(format_squad_table(squad_df)),
                    use_container_width=True
                )

        with tab2:
            st.markdown(
                '<div class="comparison-banner">GW1 Squad Builder</div>',
                unsafe_allow_html=True
            )

            st.write(
                "This section builds an initial Gameweek 1 squad automatically using the previous season summary and the live current player pool"
            )

            col_a, col_b = st.columns(2)

            with col_a:
                include_unmatched = st.toggle(
                    "Include unmatched players",
                    value=True,
                    help="Keep players without a previous-season match in the pool using fallback scoring."
                )

            with col_b:
                unmatched_penalty = st.slider(
                    "Unmatched player penalty",
                    min_value=0.50,
                    max_value=1.00,
                    value=0.85,
                    step=0.05,
                    help="Lower values penalize unmatched players more strongly."
                )

            st.caption(
                "Unmatched player penalty reduces the hybrid score of players who do not have a usable previous-season match. "
                "Higher values trust new players more, while lower values push the optimizer toward players with Premier League Experience."
            )

            gw1_outputs = load_gw1_hybrid_outputs_with_ui(
                include_unmatched=include_unmatched,
                unmatched_penalty=unmatched_penalty,
            )

            candidate_pool = gw1_outputs.get("candidate_pool", pd.DataFrame())
            hybrid_squad = gw1_outputs.get("hybrid_squad", pd.DataFrame())
            summary_df = gw1_outputs.get("summary", pd.DataFrame())

            if candidate_pool.empty or hybrid_squad.empty:
                st.error("Could not build the GW1 hybrid squad.")
            else:
                matched_count = int(candidate_pool["matched_previous_season"].sum())
                unmatched_count = int((candidate_pool["matched_previous_season"] == 0).sum())
                hybrid_cost = hybrid_squad["value"].sum() / 10.0
                hybrid_score = hybrid_squad["hybrid_score"].sum()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Candidate Players", len(candidate_pool))
                c2.metric("Matched Players", matched_count)
                c3.metric("Unmatched Players", unmatched_count)
                c4.metric("Hybrid Squad Cost", f"{hybrid_cost:.1f}")

                c5, c6, c7 = st.columns(3)
                c5.metric("Hybrid Squad Players", len(hybrid_squad))
                c6.metric("Hybrid Score Total", f"{hybrid_score:.2f}")
                c7.metric(
                    "Solve Status",
                    summary_df.iloc[0]["Solve_Status"] if not summary_df.empty else "Unknown"
                )

                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("### Position Counts")
                    pos_counts = hybrid_squad["position"].value_counts().reindex(
                        ["GK", "DEF", "MID", "FWD"],
                        fill_value=0
                    )
                    st.dataframe(
                        style_table(pos_counts.rename("Count").to_frame()),
                        use_container_width=True
                    )

                with col_right:
                    st.markdown("### Players Per Club")
                    st.dataframe(
                        style_table(hybrid_squad["team"].value_counts().rename("Count").to_frame()),
                        use_container_width=True
                    )

                st.markdown(
                    '<div class="comparison-banner">Hybrid GW1 Squad</div>',
                     unsafe_allow_html=True
                )

                gw1_starting_df, gw1_bench_df, gw1_formation = build_best_gw1_starting_xi(hybrid_squad)

                if not gw1_starting_df.empty:
                    st.markdown(f"### Best GW1 Starting XI ({gw1_formation})")
                    render_pitch(gw1_starting_df, gw1_bench_df)

                st.markdown("### Full Hybrid GW1 Squad")
                st.dataframe(
                    style_table(format_gw1_squad_table(hybrid_squad)),
                    use_container_width=True
                )

                with st.expander("Show top GW1 candidate players by hybrid score"):
                    top_candidates = candidate_pool[
                        [
                            "name",
                            "team",
                            "position",
                            "price_m",
                            "hybrid_score",
                            "matched_previous_season",
                            "total_points_sum",
                            "total_points_avg",
                            "value_efficiency",
                            "start_rate",
                            "minutes_per_appearance",
                        ]
                    ].sort_values("hybrid_score", ascending=False).head(30)
                    st.dataframe(
                        style_table(format_gw1_squad_table(top_candidates)),
                        use_container_width=True
                    )

                with st.expander("Show GW1 summary table"):
                    st.dataframe(
                        style_table(summary_df),
                        use_container_width=True
                    )

# -----------------------------
# Transfer Assistant
# -----------------------------
elif page == "Transfer Assistant":
    hero_header(
        "Transfer Assistant",
        "Build your current 15-player squad by position, validate it, then choose your starting XI and get transfer recommendations. Squad Value may be over the 100m limit due to player price increase"        
    )

    predictions_df = load_predictions_with_ui()

    if predictions_df.empty:
        st.error("No prediction data was returned.")
        st.stop()

    section_box_title(
        "Step 1: Select Your Current 15-Player Squad",
        "Choose your squad by position first: 2 goalkeepers, 5 defenders, 5 midfielders, and 3 forwards."
    )

    sorted_predictions = predictions_df.sort_values(["position", "team", "name"]).reset_index(drop=True)

    def build_position_label_map(df: pd.DataFrame, position: str) -> dict[str, str]:
        position_df = df[df["position"] == position].copy()
        position_df = position_df.sort_values(["price_m", "name"], ascending=[False, True])

        label_map: dict[str, str] = {}
        for _, row in position_df.iterrows():
            label = f"{row['name']} | {row['team']} | £{row['price_m']:.1f}m"
            label_map[label] = row["name"]
        return label_map

    gk_label_map = build_position_label_map(sorted_predictions, "GK")
    def_label_map = build_position_label_map(sorted_predictions, "DEF")
    mid_label_map = build_position_label_map(sorted_predictions, "MID")
    fwd_label_map = build_position_label_map(sorted_predictions, "FWD")

    gk_options = list(gk_label_map.keys())
    def_options = list(def_label_map.keys())
    mid_options = list(mid_label_map.keys())
    fwd_options = list(fwd_label_map.keys())

    col1, col2 = st.columns(2)

    with col1:
        selected_gk_labels = st.multiselect(
            "Select 2 Goalkeepers",
            options=gk_options,
            default=[],
        )

        selected_def_labels = st.multiselect(
            "Select 5 Defenders",
            options=def_options,
            default=[],
        )

    with col2:
        selected_mid_labels = st.multiselect(
            "Select 5 Midfielders",
            options=mid_options,
            default=[],
        )

        selected_fwd_labels = st.multiselect(
            "Select 3 Forwards",
            options=fwd_options,
            default=[],
        )

    money_in_bank = st.number_input(
        "Money In Bank",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
    )

    transfer_count = st.radio(
        "Number of Transfers Available",
        [1, 2],
        horizontal=True,
    )

    selected_gk_names = [gk_label_map[label] for label in selected_gk_labels]
    selected_def_names = [def_label_map[label] for label in selected_def_labels]
    selected_mid_names = [mid_label_map[label] for label in selected_mid_labels]
    selected_fwd_names = [fwd_label_map[label] for label in selected_fwd_labels]

    selected_names = (
        selected_gk_names
        + selected_def_names
        + selected_mid_names
        + selected_fwd_names
    )

    current_squad_df = pd.DataFrame()

    if len(selected_names) > 0:
        current_squad_df = predictions_df[predictions_df["name"].isin(selected_names)].copy().reset_index(drop=True)

        section_box_title("Step 2: Validate Current Squad", "The squad must satisfy all FPL constraints before moving to the Starting XI step.")

        valid, reasons, squad_cost, total_budget_used = validate_full_squad(
            current_squad_df,
            money_in_bank=money_in_bank,
            enforce_budget=False,
        )

        counts = current_squad_df["position"].value_counts()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("GK", counts.get("GK", 0))
        c2.metric("DEF", counts.get("DEF", 0))
        c3.metric("MID", counts.get("MID", 0))
        c4.metric("FWD", counts.get("FWD", 0))
        c5.metric("Players", len(current_squad_df))
        c6.metric("Squad Cost", f"{squad_cost:.1f}")

        c7, c8 = st.columns(2)
        c7.metric("Money in Bank", f"{money_in_bank:.1f}")
        c8.metric("Squad Value + Bank", f"{total_budget_used:.1f}")

        st.markdown("### Current Squad Preview")
        st.dataframe(
            style_table(format_squad_table(current_squad_df)),
            use_container_width=True
        )

        if valid:
            st.success("Your 15-player squad is valid.")
        else:
            st.error("Your squad is not valid.")
            for reason in reasons:
                st.write(f"- {reason}")

        if valid:
            section_box_title(
                "Step 3: Select Your Starting XI",
                "Choose 1 goalkeeper, 3 to 5 defenders, 2 to 5 midfielders, and 1 to 3 forwards."
            )

            starting_gk_label = st.selectbox(
                "Starting Goalkeeper",
                options=selected_gk_labels,
            )

            col3, col4 = st.columns(2)

            with col3:
                starting_def_labels = st.multiselect(
                    "Starting Defenders (3 to 5)",
                    options=selected_def_labels,
                    default=[],
                )

                starting_mid_labels = st.multiselect(
                    "Starting Midfielders (2 to 5)",
                    options=selected_mid_labels,
                    default=[],
                )

            with col4:
                starting_fwd_labels = st.multiselect(
                    "Starting Forwards (1 to 3)",
                    options=selected_fwd_labels,
                    default=[],
                )

            starting_names = (
                [gk_label_map[starting_gk_label]]
                + [def_label_map[label] for label in starting_def_labels]
                + [mid_label_map[label] for label in starting_mid_labels]
                + [fwd_label_map[label] for label in starting_fwd_labels]
            )

            starting_df = current_squad_df[current_squad_df["name"].isin(starting_names)].copy()
            starting_valid = validate_starting_xi(starting_df)

            st.write(f"Starting XI selected: **{len(starting_df)}**")

            if starting_valid:
                st.success("Your starting XI is valid.")
            else:
                st.warning(
                    "Your starting XI is not valid. A valid XI needs exactly 11 players, exactly 1 GK, at least 3 DEF, at least 2 MID, and at least 1 FWD."
                )

            if starting_valid:
                c9, c10 = st.columns(2)
                c9.metric(
                    "Current Squad Predicted Total",
                    f"{current_squad_df['predicted_points'].sum():.2f}"
                )
                c10.metric(
                    "Current Starting XI Predicted Total",
                    f"{starting_df['predicted_points'].sum():.2f}"
                )

                bench_df = current_squad_df[~current_squad_df["name"].isin(starting_names)].copy()
                render_pitch(starting_df, bench_df)

                col_a, col_b = st.columns(2)

                with col_a:
                    run_one_transfer = st.button("Generate Best 1 Transfer")

                with col_b:
                    run_two_transfers = st.button("Generate Best 2 Transfers")

                if run_one_transfer:
                    with st.spinner("Generating best 1-transfer recommendations..."):
                        one_transfer_df = recommend_best_one_transfer(
                            current_squad_df=current_squad_df,
                            predictions_df=predictions_df,
                            money_in_bank=money_in_bank,
                            starting_names=starting_names,
                            verbose=False,
                        )

                    st.markdown('<div class="comparison-banner">Best 1-Transfer Recommendations</div>', unsafe_allow_html=True)

                    if one_transfer_df.empty:
                        st.info("No valid 1-transfer recommendations were found.")
                    else:
                        st.dataframe(
                            style_table(format_one_transfer_table(one_transfer_df.head(20))),
                            use_container_width=True,
                        )

                if run_two_transfers:
                    with st.spinner("Generating best 2-transfer recommendations..."):
                        two_transfer_df = recommend_best_two_transfers(
                            current_squad_df=current_squad_df,
                            predictions_df=predictions_df,
                            money_in_bank=money_in_bank,
                            starting_names=starting_names,
                            verbose=False,
                        )

                    st.markdown('<div class="comparison-banner">Best 2-Transfer Recommendations</div>', unsafe_allow_html=True)

                    if two_transfer_df.empty:
                        st.info("No valid 2-transfer recommendations were found.")
                    else:
                        st.dataframe(
                            style_table(format_two_transfer_table(two_transfer_df.head(20))),
                            use_container_width=True,
                        )