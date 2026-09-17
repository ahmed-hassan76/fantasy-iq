from __future__ import annotations

from datetime import datetime
import html
import time
import streamlit as st
import pandas as pd
from src.api import clear_api_cache, FPLApiError
from src.prediction_log import (infer_prediction_target_gameweek, build_prediction_log_export,
                                complete_prediction_log, ActualPointsUnavailable)
from src.predict import build_predictions_table
from src.fixture_planner import build_normalized_team_fixtures, build_team_fixture_summary
from src.optimizer import (
    build_optimized_squad_from_predictions,
    optimize_best_15_squad,
    summarize_squad,
)
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
            min-height: 108px;
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

        .player-card-risk {
            color: #9b1c31 !important;
            font-size: 0.70rem !important;
            font-weight: 800 !important;
            line-height: 1.1 !important;
            margin-top: 4px !important;
            word-break: break-word;
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


def infer_current_fpl_season_label(today: datetime | None = None) -> str:
    if today is None:
        today = datetime.now()

    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year}-{start_year + 1}"


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


@st.cache_data(ttl=1800, show_spinner=False)
def load_fixture_planner_data_cached() -> tuple[pd.DataFrame, pd.DataFrame]:
    fixtures_df = build_normalized_team_fixtures(use_cache=True)
    summary_df = build_team_fixture_summary(use_cache=True)
    return fixtures_df, summary_df


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

    try:
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
    except Exception as exc:
        elapsed = time.time() - start_time
        message = str(exc).strip() or "Best Current Squad could not be built right now."
        status_box.warning(
            "Best Current Squad is currently unavailable because live predictions do not have enough current-season history yet, especially early in a new season. The GW1 Squad Builder tab remains available."
        )
        st.caption(f"Details: {message}")
        st.session_state["optimized_squad_df"] = pd.DataFrame()
        progress.progress(100, text="Best Current Squad unavailable.")
        return pd.DataFrame()


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
def risk_badge_text(row: pd.Series) -> str:
    if "risk_level" not in row.index or pd.isna(row["risk_level"]):
        return ""

    risk_level = str(row["risk_level"]).strip()
    if risk_level.lower() in {"", "low", "none", "nan"}:
        return ""

    if risk_level.lower() == "high":
        return "🚨 High"

    if risk_level.lower() == "medium":
        return "⚠ Medium"

    return risk_level


def format_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    rename_map = {
        "name": "Name",
        "team": "Team",
        "position": "Position",
        "price_m": "Price",
        "predicted_points": "Predicted Points",
        "risk_level": "Risk Level",
        "risk_flags": "Risk Flags",
        "next_3_fixtures": "Next 3 Fixtures",
        "next_3_fdr_avg": "Next 3 FDR Avg",
        "next_5_fixtures": "Next 5 Fixtures",
        "next_5_fdr_avg": "Next 5 FDR Avg",
        "model_used": "Model Used",
        "source_round": "Latest Available Source Round",
    }
    temp = temp.rename(columns=rename_map)

    if "Price" in temp.columns:
        temp["Price"] = pd.to_numeric(temp["Price"], errors="coerce").round(1)
    if "Predicted Points" in temp.columns:
        temp["Predicted Points"] = pd.to_numeric(temp["Predicted Points"], errors="coerce").round(2)
    for col in ["Next 3 FDR Avg", "Next 5 FDR Avg"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce").round(2)

    preferred_cols = [
        "Name",
        "Team",
        "Position",
        "Price",
        "Predicted Points",
        "Risk Level",
        "Risk Flags",
        "Next 3 Fixtures",
        "Next 3 FDR Avg",
        "Next 5 Fixtures",
        "Next 5 FDR Avg",
        "Model Used",
        "Latest Available Source Round",
    ]
    existing_cols = [col for col in preferred_cols if col in temp.columns]
    return temp[existing_cols]


def render_previous_prediction_log() -> None:
    with st.expander("Complete Previous Prediction Log"):
        st.caption("Upload a saved pre-deadline Prediction Log from the current FPL season. Only official completed and checked gameweeks can be filled. Original predictions are preserved.")
        uploaded = st.file_uploader("Previous Prediction Log CSV", type=["csv"], key="previous_prediction_log")
        if uploaded is not None:
            try:
                # Read as strings so original prediction precision and other fields survive unchanged.
                snapshot = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
                st.dataframe(snapshot, use_container_width=True)
                if st.button("Fetch Official Actual Points", key="complete_prediction_log"):
                    with st.spinner("Fetching official FPL points..."):
                        completed, gameweek, unmatched = complete_prediction_log(snapshot)
                    st.session_state["completed_prediction_log"] = (
                        uploaded.getvalue(), completed, gameweek, unmatched
                    )
                saved = st.session_state.get("completed_prediction_log")
                if saved is not None and saved[0] == uploaded.getvalue():
                    _, completed, gameweek, unmatched = saved
                    if unmatched:
                        st.warning(f"{unmatched} player(s) could not be matched safely or have no official points. Their actuals and errors remain blank.")
                    else:
                        st.success(f"GW{gameweek} completed using official FPL points.")
                    st.dataframe(completed, use_container_width=True)
                    st.download_button("Download Completed Prediction Log CSV", completed.to_csv(index=False).encode("utf-8-sig"), file_name=f"fantasy_iq_prediction_log_GW{gameweek}_completed.csv", mime="text/csv", key="completed_prediction_log_csv")
            except (ActualPointsUnavailable, FPLApiError) as exc:
                st.info(str(exc))
            except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
                st.error(f"Unable to complete this prediction log: {exc}")


def render_prediction_results_header(
    df: pd.DataFrame,
    target_gameweek: int | None = None,
    unavailable_message: str = "Prediction export unavailable until prediction results are loaded.",
) -> None:
    title_col, export_col = st.columns([3, 1])

    with title_col:
        st.markdown('<div class="comparison-banner">Prediction Results</div>', unsafe_allow_html=True)

    with export_col:
        if df.empty:
            st.download_button(
                "Download Prediction Log CSV",
                data=b"",
                file_name=(
                    f"fantasy_iq_prediction_log_GW{target_gameweek}.csv"
                    if target_gameweek is not None
                    else "fantasy_iq_prediction_log.csv"
                ),
                mime="text/csv",
                key="download_prediction_log_csv",
                disabled=True,
                use_container_width=True,
            )
            st.caption(unavailable_message)
            return

        prediction_log_df, prediction_gameweek = build_prediction_log_export(df, target_gameweek)
        prediction_log_filename = f"fantasy_iq_prediction_log_GW{prediction_gameweek}.csv"
        st.download_button(
            "Download Prediction Log CSV",
            data=prediction_log_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=prediction_log_filename,
            mime="text/csv",
            key="download_prediction_log_csv",
            use_container_width=True,
        )


def format_fixture_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    rename_map = {
        "team_short_name": "Team Short Name",
        "team_name": "Team",
        "next_3_fixtures": "Next 3 Fixtures",
        "next_3_fdr_avg": "Next 3 FDR Avg",
        "next_3_fixture_count": "Next 3 Fixture Count",
        "next_5_fixtures": "Next 5 Fixtures",
        "next_5_fdr_avg": "Next 5 FDR Avg",
        "next_5_fixture_count": "Next 5 Fixture Count",
    }
    temp = temp.rename(columns=rename_map)

    for col in ["Next 3 FDR Avg", "Next 5 FDR Avg"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce").round(2)

    preferred_cols = [
        "Team Short Name",
        "Team",
        "Next 3 Fixtures",
        "Next 3 FDR Avg",
        "Next 3 Fixture Count",
        "Next 5 Fixtures",
        "Next 5 FDR Avg",
        "Next 5 Fixture Count",
    ]
    existing_cols = [col for col in preferred_cols if col in temp.columns]
    return temp[existing_cols]


def format_upcoming_fixtures_table(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()

    if "is_home" in temp.columns:
        temp["is_home"] = temp["is_home"].map({True: "H", False: "A"}).fillna("")

    if "kickoff_time" in temp.columns:
        kickoff = pd.to_datetime(temp["kickoff_time"], errors="coerce")
        temp["kickoff_time"] = kickoff.dt.strftime("%Y-%m-%d %H:%M").fillna("")

    rename_map = {
        "team_short_name": "Team Short Name",
        "team_name": "Team",
        "event": "Gameweek",
        "opponent_short_name": "Upcoming Opponent",
        "is_home": "Home/Away",
        "kickoff_time": "Kickoff Time",
        "difficulty": "FDR",
        "next_3_fdr_avg": "Next 3 FDR Avg",
        "next_5_fdr_avg": "Next 5 FDR Avg",
    }
    temp = temp.rename(columns=rename_map)

    for col in ["Gameweek", "FDR"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce").astype("Int64")

    for col in ["Next 3 FDR Avg", "Next 5 FDR Avg"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce").round(2)

    preferred_cols = [
        "Team Short Name",
        "Team",
        "Gameweek",
        "Upcoming Opponent",
        "Home/Away",
        "Kickoff Time",
        "FDR",
        "Next 3 FDR Avg",
        "Next 5 FDR Avg",
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
        "risk_level": "Risk Level",
        "risk_flags": "Risk Flags",
        "model_used": "Model Used",
    }
    temp = temp.rename(columns=rename_map)

    if "Price" in temp.columns:
        temp["Price"] = pd.to_numeric(temp["Price"], errors="coerce").round(1)
    if "Predicted Points" in temp.columns:
        temp["Predicted Points"] = pd.to_numeric(temp["Predicted Points"], errors="coerce").round(2)

    preferred_cols = ["Name", "Team", "Position", "Price", "Predicted Points", "Risk Level", "Risk Flags", "Model Used"]
    existing_cols = [col for col in preferred_cols if col in temp.columns]

    return temp[existing_cols].sort_values(
        ["Position", "Predicted Points"],
        ascending=[True, False]
    )


def format_captain_recommendation_table(
    df: pd.DataFrame,
    top_n: int = 5,
    roles: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty or "predicted_points" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()
    temp["predicted_points"] = pd.to_numeric(temp["predicted_points"], errors="coerce")
    temp = temp.dropna(subset=["predicted_points"])

    if temp.empty:
        return pd.DataFrame()

    temp = temp.sort_values("predicted_points", ascending=False).head(top_n).reset_index(drop=True)
    temp["predicted_captain_points"] = temp["predicted_points"] * 2

    if roles is not None:
        temp["captain_role"] = roles[:len(temp)]

    rename_map = {
        "captain_role": "Recommendation",
        "name": "Name",
        "team": "Team",
        "position": "Position",
        "price_m": "Price",
        "predicted_points": "Predicted Points",
        "predicted_captain_points": "Predicted Captain Points",
        "risk_level": "Risk Level",
        "risk_flags": "Risk Flags",
        "next_3_fixtures": "Next 3 Fixtures",
        "next_3_fdr_avg": "Next 3 FDR Avg",
        "next_5_fixtures": "Next 5 Fixtures",
        "next_5_fdr_avg": "Next 5 FDR Avg",
    }
    temp = temp.rename(columns=rename_map)

    if "Price" in temp.columns:
        temp["Price"] = pd.to_numeric(temp["Price"], errors="coerce").round(1)

    for col in ["Predicted Points", "Predicted Captain Points", "Next 3 FDR Avg", "Next 5 FDR Avg"]:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce").round(2)

    preferred_cols = [
        "Recommendation",
        "Name",
        "Team",
        "Position",
        "Price",
        "Predicted Points",
        "Predicted Captain Points",
        "Risk Level",
        "Risk Flags",
        "Next 3 Fixtures",
        "Next 3 FDR Avg",
        "Next 5 Fixtures",
        "Next 5 FDR Avg",
    ]
    existing_cols = [col for col in preferred_cols if col in temp.columns]

    return temp[existing_cols]


def build_team_rating_summary(
    current_squad_df: pd.DataFrame,
    starting_df: pd.DataFrame,
    bench_df: pd.DataFrame,
) -> dict[str, object]:
    starting_points = (
        pd.to_numeric(starting_df["predicted_points"], errors="coerce").fillna(0).sum()
        if "predicted_points" in starting_df.columns
        else 0.0
    )
    bench_points = (
        pd.to_numeric(bench_df["predicted_points"], errors="coerce").fillna(0).sum()
        if "predicted_points" in bench_df.columns
        else 0.0
    )

    starting_score = min(float(starting_points) * 1.2, 80)
    bench_score = min(float(bench_points) * 0.5, 15)

    fixture_adjustment = 0
    fixture_summary = "Fixture data unavailable"
    fixture_cols = [col for col in ["next_3_fdr_avg", "next_5_fdr_avg"] if col in starting_df.columns]
    if fixture_cols:
        fixture_values = pd.concat(
            [pd.to_numeric(starting_df[col], errors="coerce") for col in fixture_cols],
            ignore_index=True,
        ).dropna()
        if not fixture_values.empty:
            avg_fixture_difficulty = float(fixture_values.mean())
            if avg_fixture_difficulty <= 2.5:
                fixture_adjustment = 5
            elif avg_fixture_difficulty <= 3.0:
                fixture_adjustment = 3
            elif avg_fixture_difficulty <= 3.5:
                fixture_adjustment = 1
            elif avg_fixture_difficulty <= 4.0:
                fixture_adjustment = -2
            else:
                fixture_adjustment = -5
            fixture_summary = f"Average fixture difficulty: {avg_fixture_difficulty:.2f}"

    high_starters = medium_starters = high_bench = medium_bench = 0
    risk_penalty = 0
    risk_summary = "Risk data unavailable"

    if "risk_level" in current_squad_df.columns:
        starter_risk = starting_df["risk_level"].fillna("").astype(str).str.lower().str.strip()
        bench_risk = bench_df["risk_level"].fillna("").astype(str).str.lower().str.strip()

        high_starters = int(starter_risk.eq("high").sum())
        medium_starters = int(starter_risk.eq("medium").sum())
        high_bench = int(bench_risk.eq("high").sum())
        medium_bench = int(bench_risk.eq("medium").sum())

        raw_risk_penalty = (
            high_starters * 4
            + medium_starters * 2
            + high_bench * 2
            + medium_bench
        )
        risk_penalty = min(raw_risk_penalty, 15)

        risk_summary = (
            f"Starters: {high_starters} High, {medium_starters} Medium. "
            f"Bench: {high_bench} High, {medium_bench} Medium."
        )

        if "risk_flags" in current_squad_df.columns:
            flags = (
                current_squad_df["risk_flags"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            flags = [
                flag
                for flag in flags
                if flag and flag.lower() not in {"none", "nan"}
            ]
            if flags:
                risk_summary = f"{risk_summary} Flags: {', '.join(list(dict.fromkeys(flags))[:3])}."

    rating = max(0, min(100, starting_score + bench_score + fixture_adjustment - risk_penalty))

    return {
        "overall_rating": rating,
        "starting_xi_strength": float(starting_points),
        "bench_strength": float(bench_points),
        "starting_xi_score": starting_score,
        "bench_score": bench_score,
        "fixture_adjustment": fixture_adjustment,
        "fixture_summary": fixture_summary,
        "risk_penalty": risk_penalty,
        "risk_warning_summary": risk_summary,
    }


def format_team_rating_breakdown(summary: dict[str, object]) -> pd.DataFrame:
    risk_penalty = float(summary["risk_penalty"])
    risk_penalty_text = "0" if risk_penalty == 0 else f"-{risk_penalty:.0f}"

    return pd.DataFrame([
        {
            "Factor": "Overall Team Rating",
            "What it means": "Overall squad score out of 100",
            "Impact": (
                f"{float(summary['overall_rating']):.0f} / 100 "
                f"({describe_team_rating(float(summary['overall_rating']))})"
            ),
        },
        {
            "Factor": "Team Strength Level",
            "What it means": "Plain-English interpretation of the overall team rating",
            "Impact": describe_team_rating(float(summary["overall_rating"])),
        },
        {
            "Factor": "Starting XI Strength",
            "What it means": "Strength of the selected 11 starters",
            "Impact": (
                f"{float(summary['starting_xi_strength']):.2f} predicted points; "
                f"contributes {float(summary['starting_xi_score']):.2f} / 80"
            ),
        },
        {
            "Factor": "Bench Strength",
            "What it means": "Quality of the 4 bench players",
            "Impact": (
                f"{float(summary['bench_strength']):.2f} predicted points; "
                f"contributes {float(summary['bench_score']):.2f} / 15"
            ),
        },
        {
            "Factor": "Fixture Adjustment",
            "What it means": "Bonus or penalty based on upcoming fixture difficulty",
            "Impact": f"{float(summary['fixture_adjustment']):+.0f} points",
        },
        {
            "Factor": "Risk Penalty",
            "What it means": "Deduction for injured, doubtful, suspended, or limited-minute players",
            "Impact": f"{risk_penalty_text} points",
        },
    ])


def describe_team_rating(rating: float) -> str:
    if rating >= 80:
        return "Excellent"
    if rating >= 65:
        return "Strong"
    if rating >= 50:
        return "Average"
    if rating >= 35:
        return "Weak"
    return "Needs attention"


def format_one_transfer_table(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    rename_map = {
        "player_out": "Player Out",
        "player_out_position": "Out Position",
        "player_in": "Player In",
        "player_in_position": "In Position",
        "predicted_points_gain": "Predicted Points Gain",
        "remaining_money_in_bank": "Remaining Money In Bank",
        "outgoing_is_starter": "Outgoing Starter",
    }
    temp = temp.rename(columns=rename_map)

    transfer_columns = [
        "Player Out",
        "Out Position",
        "Player In",
        "In Position",
        "Predicted Points Gain",
        "Remaining Money In Bank",
        "Outgoing Starter",
    ]
    for column in transfer_columns:
        if column not in temp.columns:
            temp[column] = ""

    temp["Predicted Points Gain"] = pd.to_numeric(
        temp["Predicted Points Gain"], errors="coerce"
    ).round(2)
    temp["Remaining Money In Bank"] = pd.to_numeric(
        temp["Remaining Money In Bank"], errors="coerce"
    ).round(1)
    temp["Outgoing Starter"] = temp["Outgoing Starter"].map(
        lambda value: "Yes" if value is True or value == 1 else "No"
    )
    return temp[transfer_columns]


def build_two_transfer_display(
    recommendation: pd.Series,
    predictions_df: pd.DataFrame,
    starting_names: list[str],
) -> pd.DataFrame:
    """Expand one paired recommendation into two readable transfer rows."""
    player_lookup = predictions_df.drop_duplicates("name").set_index("name")
    display_rows: list[dict[str, object]] = []

    for transfer_number in (1, 2):
        player_out = recommendation.get(f"player_out_{transfer_number}", "")
        player_in = recommendation.get(f"player_in_{transfer_number}", "")
        outgoing = player_lookup.loc[player_out] if player_out in player_lookup.index else None
        incoming = player_lookup.loc[player_in] if player_in in player_lookup.index else None

        outgoing_points = pd.to_numeric(
            outgoing.get("predicted_points") if outgoing is not None else None,
            errors="coerce",
        )
        incoming_points = pd.to_numeric(
            incoming.get("predicted_points") if incoming is not None else None,
            errors="coerce",
        )
        points_gain = (
            float(incoming_points - outgoing_points)
            if pd.notna(incoming_points) and pd.notna(outgoing_points)
            else None
        )
        display_rows.append(
            {
                "Player Out": player_out,
                "Out Position": outgoing.get("position", "") if outgoing is not None else "",
                "Player In": player_in,
                "In Position": incoming.get("position", "") if incoming is not None else "",
                "Predicted Points Gain": points_gain,
                "Remaining Money In Bank": recommendation.get("remaining_money_in_bank"),
                "Outgoing Starter": player_out in starting_names,
            }
        )

    return format_one_transfer_table(pd.DataFrame(display_rows))


def build_wildcard_change_display(
    current_squad_df: pd.DataFrame,
    wildcard_squad_df: pd.DataFrame,
    starting_names: list[str],
    remaining_bank: float,
) -> pd.DataFrame:
    """Pair Wildcard changes by position and return the shared transfer display schema."""
    current_names = set(current_squad_df["name"])
    wildcard_names = set(wildcard_squad_df["name"])
    outgoing_df = current_squad_df[~current_squad_df["name"].isin(wildcard_names)].copy()
    incoming_df = wildcard_squad_df[~wildcard_squad_df["name"].isin(current_names)].copy()
    display_rows: list[dict[str, object]] = []

    for position in ["GK", "DEF", "MID", "FWD"]:
        outgoing_players = outgoing_df[outgoing_df["position"] == position].sort_values(
            "predicted_points", ascending=True
        ).to_dict("records")
        incoming_players = incoming_df[incoming_df["position"] == position].sort_values(
            "predicted_points", ascending=False
        ).to_dict("records")
        pair_count = max(len(outgoing_players), len(incoming_players))

        for index in range(pair_count):
            outgoing = outgoing_players[index] if index < len(outgoing_players) else {}
            incoming = incoming_players[index] if index < len(incoming_players) else {}
            outgoing_points = pd.to_numeric(outgoing.get("predicted_points"), errors="coerce")
            incoming_points = pd.to_numeric(incoming.get("predicted_points"), errors="coerce")
            points_gain = (
                float(incoming_points - outgoing_points)
                if pd.notna(incoming_points) and pd.notna(outgoing_points)
                else None
            )
            player_out = outgoing.get("name", "")
            display_rows.append(
                {
                    "Player Out": player_out,
                    "Out Position": outgoing.get("position", position),
                    "Player In": incoming.get("name", ""),
                    "In Position": incoming.get("position", position),
                    "Predicted Points Gain": points_gain,
                    "Remaining Money In Bank": remaining_bank,
                    "Outgoing Starter": player_out in starting_names if player_out else False,
                }
            )

    return format_one_transfer_table(pd.DataFrame(display_rows))


def apply_transfer_recommendation_to_squad(
    current_squad_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    outgoing_names: list[str],
    incoming_names: list[str],
) -> tuple[pd.DataFrame, str]:
    """Build and validate an atomic squad update without mutating the current squad."""
    if len(outgoing_names) != len(incoming_names) or not outgoing_names:
        return pd.DataFrame(), "The recommendation does not contain a complete set of transfers."
    if len(set(outgoing_names)) != len(outgoing_names) or len(set(incoming_names)) != len(incoming_names):
        return pd.DataFrame(), "The recommendation contains duplicate players and cannot be applied safely."

    current_names = current_squad_df["name"].astype(str).tolist()
    missing_outgoing = [name for name in outgoing_names if name not in current_names]
    if missing_outgoing:
        return pd.DataFrame(), f"Outgoing player(s) not found in the current squad: {', '.join(missing_outgoing)}."

    retained_names = [name for name in current_names if name not in outgoing_names]
    duplicate_incoming = [name for name in incoming_names if name in retained_names]
    if duplicate_incoming:
        return pd.DataFrame(), f"Incoming player(s) already exist in the squad: {', '.join(duplicate_incoming)}."

    updated_names = retained_names + incoming_names
    if len(updated_names) != 15 or len(set(updated_names)) != 15:
        return pd.DataFrame(), "Applying this recommendation would not leave exactly 15 unique players."

    prediction_rows = predictions_df[predictions_df["name"].isin(updated_names)].copy()
    name_counts = prediction_rows["name"].value_counts()
    ambiguous_names = [name for name in updated_names if int(name_counts.get(name, 0)) != 1]
    if ambiguous_names:
        return pd.DataFrame(), (
            "The following player records are missing or ambiguous: "
            f"{', '.join(ambiguous_names)}."
        )

    updated_squad_df = prediction_rows.set_index("name").loc[updated_names].reset_index()
    valid, reasons, _, _ = validate_full_squad(
        updated_squad_df,
        money_in_bank=0.0,
        enforce_budget=False,
    )
    if not valid:
        return pd.DataFrame(), "Recommendation was not applied: " + " ".join(reasons)

    predicted_values = pd.to_numeric(updated_squad_df["predicted_points"], errors="coerce")
    if predicted_values.isna().any():
        return pd.DataFrame(), "Recommendation was not applied because prediction values are missing."

    return updated_squad_df.reset_index(drop=True), ""


def format_transfer_assistant_squad(df: pd.DataFrame) -> pd.DataFrame:
    """Format the applied squad using the concise Transfer Assistant schema."""
    temp = df.rename(
        columns={
            "name": "Player",
            "team": "Team",
            "position": "Position",
            "price_m": "Price",
            "predicted_points": "Predicted Points",
            "risk_level": "Risk Level",
        }
    ).copy()
    columns = ["Player", "Team", "Position", "Price", "Predicted Points", "Risk Level"]
    for column in columns:
        if column not in temp.columns:
            temp[column] = ""
    temp["Price"] = pd.to_numeric(temp["Price"], errors="coerce").round(1)
    temp["Predicted Points"] = pd.to_numeric(
        temp["Predicted Points"], errors="coerce"
    ).round(2)
    return temp[columns]


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

    # Price should always display with 1 decimal in FPL format
    if "Price" in temp.columns:
        temp["Price"] = pd.to_numeric(temp["Price"], errors="coerce").round(1)

    for col in [
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


def _format_comparison_value(row: pd.Series, col: str) -> str:
    if col not in row.index or pd.isna(row[col]):
        return "N/A"

    value = row[col]

    if col == "price_m":
        numeric_value = pd.to_numeric(value, errors="coerce")
        return f"{numeric_value:.1f}m" if pd.notna(numeric_value) else "N/A"

    if col in {
        "predicted_points",
        "next_3_fdr_avg",
        "next_5_fdr_avg",
        "total_points_lag1",
        "total_points_rolling3",
        "minutes_lag1",
        "minutes_rolling3",
    }:
        numeric_value = pd.to_numeric(value, errors="coerce")
        return f"{numeric_value:.2f}" if pd.notna(numeric_value) else "N/A"

    if col in {"played_last_gw", "starts"}:
        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.notna(numeric_value):
            if numeric_value == 1:
                return "Yes"
            if numeric_value == 0:
                return "No"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value)

    if col == "source_round":
        numeric_value = pd.to_numeric(value, errors="coerce")
        return f"{numeric_value:.0f}" if pd.notna(numeric_value) else str(value)

    return str(value)


def _player_select_label(row: pd.Series) -> str:
    name = _format_comparison_value(row, "name")
    team = _format_comparison_value(row, "team")
    position = _format_comparison_value(row, "position")
    return f"{name} ({team}, {position})"


def render_player_comparison_card(row: pd.Series, title: str) -> None:
    fields = [
        ("Team", "team"),
        ("Position", "position"),
        ("Price", "price_m"),
        ("Predicted Points", "predicted_points"),
        ("Risk Level", "risk_level"),
        ("Risk Flags", "risk_flags"),
        ("Next 3 FDR Avg", "next_3_fdr_avg"),
        ("Next 5 FDR Avg", "next_5_fdr_avg"),
    ]
    field_html = ""

    for label, col in fields:
        if col in row.index:
            field_html += (
                '<div class="small-muted" style="margin-top:0.45rem;">'
                f"{html.escape(label)}"
                "</div>"
                f'<div style="color:white;font-weight:800;">{html.escape(_format_comparison_value(row, col))}</div>'
            )

    st.markdown(
        f"""
        <div class="section-card">
            <div class="small-muted">{html.escape(title)}</div>
            <div style="font-size:1.35rem;font-weight:900;color:white;">
                {html.escape(_format_comparison_value(row, "name"))}
            </div>
            {field_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_player_comparison_table(player_1: pd.Series, player_2: pd.Series) -> pd.DataFrame:
    comparison_fields = [
        ("Name", "name"),
        ("Team", "team"),
        ("Position", "position"),
        ("Price", "price_m"),
        ("Predicted Points", "predicted_points"),
        ("Risk Level", "risk_level"),
        ("Risk Flags", "risk_flags"),
        ("Latest Available Round", "source_round"),
        ("Next 3 Fixtures", "next_3_fixtures"),
        ("Next 3 FDR Avg", "next_3_fdr_avg"),
        ("Next 5 Fixtures", "next_5_fixtures"),
        ("Next 5 FDR Avg", "next_5_fdr_avg"),
        ("Total Points Lag 1", "total_points_lag1"),
        ("Total Points Rolling 3", "total_points_rolling3"),
        ("Minutes Lag 1", "minutes_lag1"),
        ("Minutes Rolling 3", "minutes_rolling3"),
        ("Played Last GW", "played_last_gw"),
        ("Starts", "starts"),
    ]

    rows = []
    for label, col in comparison_fields:
        if col in player_1.index or col in player_2.index:
            rows.append({
                "Field": label,
                "Player 1": _format_comparison_value(player_1, col),
                "Player 2": _format_comparison_value(player_2, col),
            })

    return pd.DataFrame(rows)


def _comparison_numeric(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None

    value = pd.to_numeric(row[col], errors="coerce")
    return float(value) if pd.notna(value) else None


def _risk_rank(row: pd.Series) -> int | None:
    if "risk_level" not in row.index or pd.isna(row["risk_level"]):
        return None

    risk_level = str(row["risk_level"]).strip().lower()
    ranks = {"low": 0, "medium": 1, "high": 2}
    return ranks.get(risk_level)


def build_better_option_summary(player_1: pd.Series, player_2: pd.Series) -> str:
    name_1 = _format_comparison_value(player_1, "name")
    name_2 = _format_comparison_value(player_2, "name")
    points_1 = _comparison_numeric(player_1, "predicted_points")
    points_2 = _comparison_numeric(player_2, "predicted_points")

    if points_1 is None or points_2 is None:
        return "Better Option: Not enough predicted point data is available to make a comparison."

    if points_1 >= points_2:
        winner, loser = player_1, player_2
        winner_name, loser_name = name_1, name_2
    else:
        winner, loser = player_2, player_1
        winner_name, loser_name = name_2, name_1

    winner_points = _comparison_numeric(winner, "predicted_points")
    loser_points = _comparison_numeric(loser, "predicted_points")
    point_gap = abs(points_1 - points_2)

    if point_gap > 0.5:
        return (
            f"Better Option: {winner_name}. "
            f"{winner_name} projects higher on predicted points "
            f"({winner_points:.2f} vs {loser_points:.2f})."
        )

    context = [
        f"Better Option: {winner_name}. Predicted points are close "
        f"({winner_points:.2f} vs {loser_points:.2f})."
    ]

    winner_risk = _risk_rank(winner)
    loser_risk = _risk_rank(loser)
    if winner_risk is not None and loser_risk is not None and winner_risk != loser_risk:
        lower_risk_name = winner_name if winner_risk < loser_risk else loser_name
        context.append(f"{lower_risk_name} has the lower risk level.")

    for fixture_col, label in [("next_3_fdr_avg", "Next 3 FDR"), ("next_5_fdr_avg", "Next 5 FDR")]:
        winner_fdr = _comparison_numeric(winner, fixture_col)
        loser_fdr = _comparison_numeric(loser, fixture_col)
        if winner_fdr is not None and loser_fdr is not None and winner_fdr != loser_fdr:
            easier_name = winner_name if winner_fdr < loser_fdr else loser_name
            context.append(f"{easier_name} has the easier {label}.")
            break

    return " ".join(context)


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

    risk_text = risk_badge_text(row)
    risk_html = f'<div class="player-card-risk">{risk_text}</div>' if risk_text else ""

    return f"""
    <div class="player-card">
        <div class="player-card-name">{name}</div>
        <div class="player-card-meta">{team} • £{price:.1f}m</div>
        <div class="player-card-points">{score_text}</div>
        {risk_html}
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


def build_transfer_assistant_starting_xi(
    squad_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Return the highest-predicted valid XI without inventing missing predictions."""
    required_columns = {"name", "position", "predicted_points"}
    if squad_df.empty or not required_columns.issubset(squad_df.columns):
        return pd.DataFrame(), ""

    available_df = squad_df.copy()
    available_df["predicted_points"] = pd.to_numeric(
        available_df["predicted_points"], errors="coerce"
    )
    available_df = available_df.dropna(subset=["predicted_points"])

    position_pools = {
        position: available_df[available_df["position"] == position].sort_values(
            "predicted_points", ascending=False
        )
        for position in ["GK", "DEF", "MID", "FWD"]
    }

    valid_formations = [
        (3, 4, 3),
        (3, 5, 2),
        (4, 3, 3),
        (4, 4, 2),
        (4, 5, 1),
        (5, 2, 3),
        (5, 3, 2),
        (5, 4, 1),
    ]
    best_xi = pd.DataFrame()
    best_formation = ""
    best_total = float("-inf")

    for defender_count, midfielder_count, forward_count in valid_formations:
        candidate = pd.concat(
            [
                position_pools["GK"].head(1),
                position_pools["DEF"].head(defender_count),
                position_pools["MID"].head(midfielder_count),
                position_pools["FWD"].head(forward_count),
            ],
            ignore_index=True,
        )
        if len(candidate) != 11 or not validate_starting_xi(candidate):
            continue

        predicted_total = float(candidate["predicted_points"].sum())
        if predicted_total > best_total:
            best_total = predicted_total
            best_xi = candidate.copy()
            best_formation = f"{defender_count}-{midfielder_count}-{forward_count}"

    return best_xi.reset_index(drop=True), best_formation


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
        "Player Comparison",
        "Fixture Planner",
        "Squad Builder",
        "Transfer Assistant",
    ]
)

st.sidebar.markdown("---")

if st.sidebar.button("Refresh Live Data"):
    st.cache_data.clear()
    clear_api_cache()
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
    st.write("• Player Comparison")
    st.write("• Fixture Planner")
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

    render_previous_prediction_log()

    predictions_df = load_predictions_with_ui()

    if predictions_df.empty:
        render_prediction_results_header(predictions_df)
        st.error("No prediction data is available for a target gameweek yet.")
        st.stop()

    inferred_target_gameweek = infer_prediction_target_gameweek(predictions_df)
    if inferred_target_gameweek is None:
        render_prediction_results_header(predictions_df.iloc[0:0])
        st.error("No valid next gameweek is available from the source data (supported targets: GW1 to GW38).")
        st.stop()

    predictions_df.attrs["latest_source_round"] = inferred_target_gameweek - 1
    selected_target_gameweek = inferred_target_gameweek
    st.info(f"Prediction Target Gameweek: GW{selected_target_gameweek} (latest source round + 1)")
    st.caption("Historical predictions are available only from previously saved snapshots.")

    section_box_title("Filters", "Filter players by position, team, price, and predicted points")

    if "risk_level" in predictions_df.columns:
        col1, col2, col3, col4 = st.columns(4)
    else:
        col1, col2, col3 = st.columns(3)

    with col1:
        position_options = ["All"] + sorted(predictions_df["position"].dropna().unique().tolist())
        selected_position = st.selectbox("Position", position_options)

    with col2:
        team_options = ["All"] + sorted(predictions_df["team"].dropna().unique().tolist())
        selected_team = st.selectbox("Team", team_options)

    with col3:
        sort_order = st.selectbox("Sort by Predicted Points", ["Descending", "Ascending"])

    selected_risk_level = "All"
    if "risk_level" in predictions_df.columns:
        with col4:
            risk_options = ["All"] + sorted(predictions_df["risk_level"].dropna().unique().tolist())
            selected_risk_level = st.selectbox("Risk Level", risk_options)

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

    if selected_risk_level != "All" and "risk_level" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["risk_level"] == selected_risk_level]

    filtered_df = filtered_df[
        (filtered_df["price_m"] >= selected_price_range[0])
        & (filtered_df["price_m"] <= selected_price_range[1])
    ]

    ascending_flag = sort_order == "Ascending"
    filtered_df = filtered_df.sort_values("predicted_points", ascending=ascending_flag).reset_index(drop=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Players Shown", len(filtered_df))
    c2.metric("Top Prediction", f"{filtered_df['predicted_points'].max():.2f}" if not filtered_df.empty else "0.00")
    c3.metric("Teams Shown", int(filtered_df["team"].nunique()) if not filtered_df.empty else 0)

    st.markdown('<div class="comparison-banner">Captain Recommendations</div>', unsafe_allow_html=True)
    captain_options_df = format_captain_recommendation_table(filtered_df, top_n=5)
    if len(captain_options_df) < 2:
        st.info("At least two available players are needed for captain recommendations.")
    else:
        st.dataframe(
            style_table(captain_options_df),
            use_container_width=True,
        )

    render_prediction_results_header(
        filtered_df,
        selected_target_gameweek,
        unavailable_message="Prediction export unavailable for the current filters.",
    )
    st.dataframe(
        style_table(format_prediction_table(filtered_df)),
        use_container_width=True
    )

# -----------------------------
# Player Comparison
# -----------------------------
elif page == "Player Comparison":
    hero_header(
        "Player Comparison",
        "Compare two players side by side using the live prediction table"
    )

    predictions_df = load_predictions_with_ui()

    if predictions_df.empty:
        st.info("No prediction data is available for player comparison right now.")
        st.stop()

    if "name" not in predictions_df.columns:
        st.info("Player names are not available in the prediction data right now.")
        st.stop()

    section_box_title("Filters", "Filter the player list before selecting two players")

    filtered_players = predictions_df.copy()
    filter_col1, filter_col2 = st.columns(2)

    selected_position = "All"
    if "position" in filtered_players.columns:
        with filter_col1:
            position_options = ["All"] + sorted(filtered_players["position"].dropna().astype(str).unique().tolist())
            selected_position = st.selectbox("Position", position_options, key="comparison_position")

    selected_team = "All"
    if "team" in filtered_players.columns:
        with filter_col2:
            team_options = ["All"] + sorted(filtered_players["team"].dropna().astype(str).unique().tolist())
            selected_team = st.selectbox("Team", team_options, key="comparison_team")

    if selected_position != "All" and "position" in filtered_players.columns:
        filtered_players = filtered_players[filtered_players["position"].astype(str) == selected_position]

    if selected_team != "All" and "team" in filtered_players.columns:
        filtered_players = filtered_players[filtered_players["team"].astype(str) == selected_team]

    sort_cols = [col for col in ["position", "team", "name"] if col in filtered_players.columns]
    if sort_cols:
        filtered_players = filtered_players.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    else:
        filtered_players = filtered_players.reset_index(drop=True)

    if filtered_players.empty:
        st.warning("No players match the selected comparison filters.")
        st.stop()

    player_options = filtered_players.index.tolist()
    player_2_default = 1 if len(player_options) > 1 else 0

    select_col1, select_col2 = st.columns(2)
    with select_col1:
        player_1_idx = st.selectbox(
            "Player 1",
            player_options,
            format_func=lambda idx: _player_select_label(filtered_players.loc[idx]),
            key="comparison_player_1",
        )

    with select_col2:
        player_2_idx = st.selectbox(
            "Player 2",
            player_options,
            index=player_2_default,
            format_func=lambda idx: _player_select_label(filtered_players.loc[idx]),
            key="comparison_player_2",
        )

    if player_1_idx == player_2_idx:
        st.warning("Select two different players to compare.")

    player_1 = filtered_players.loc[player_1_idx]
    player_2 = filtered_players.loc[player_2_idx]

    st.markdown('<div class="comparison-banner">Side-by-Side Comparison</div>', unsafe_allow_html=True)
    card_col1, card_col2 = st.columns(2)
    with card_col1:
        render_player_comparison_card(player_1, "Player 1")
    with card_col2:
        render_player_comparison_card(player_2, "Player 2")

    st.markdown('<div class="comparison-banner">Better Option</div>', unsafe_allow_html=True)
    st.info(build_better_option_summary(player_1, player_2))

    comparison_table = build_player_comparison_table(player_1, player_2)
    if comparison_table.empty:
        st.info("No comparable fields are available for these players right now.")
    else:
        st.markdown('<div class="comparison-banner">Comparison Table</div>', unsafe_allow_html=True)
        st.dataframe(
            style_table(comparison_table),
            use_container_width=True
        )

# -----------------------------
# Fixture Planner
# -----------------------------
elif page == "Fixture Planner":
    hero_header(
        "Fixture Planner",
        "View upcoming fixtures and fixture difficulty by team"
    )

    fixtures_df, summary_df = load_fixture_planner_data_cached()

    if fixtures_df.empty:
        st.info("No upcoming fixture data is available right now.")
        st.stop()

    section_box_title("Filters", "Filter teams and upcoming fixtures by fixture difficulty")

    team_filter_col = None
    if "team_name" in fixtures_df.columns and fixtures_df["team_name"].notna().any():
        team_filter_col = "team_name"
    elif "team_short_name" in fixtures_df.columns and fixtures_df["team_short_name"].notna().any():
        team_filter_col = "team_short_name"

    col1, col2, col3 = st.columns(3)

    with col1:
        if team_filter_col is not None:
            team_options = ["All"] + sorted(fixtures_df[team_filter_col].dropna().astype(str).unique().tolist())
        else:
            team_options = ["All"]
        selected_team = st.selectbox("Team", team_options)

    difficulty_values = (
        pd.to_numeric(fixtures_df["difficulty"], errors="coerce")
        if "difficulty" in fixtures_df.columns
        else pd.Series(dtype=float)
    )
    if difficulty_values.notna().any():
        min_fdr = int(difficulty_values.dropna().min())
        max_fdr = int(difficulty_values.dropna().max())
        if min_fdr == max_fdr:
            min_fdr = 1
            max_fdr = 5
    else:
        min_fdr = 1
        max_fdr = 5

    with col2:
        selected_fdr_range = st.slider(
            "FDR Range",
            min_value=min_fdr,
            max_value=max_fdr,
            value=(min_fdr, max_fdr),
            step=1,
        )

    with col3:
        fixture_limit = st.selectbox("Number of Upcoming Fixtures", [3, 5, "All available"])

    filtered_summary = summary_df.copy()
    filtered_fixtures = fixtures_df.copy()

    if selected_team != "All" and team_filter_col is not None:
        filtered_fixtures = filtered_fixtures[
            filtered_fixtures[team_filter_col].astype(str) == selected_team
        ].copy()
        if team_filter_col in filtered_summary.columns:
            filtered_summary = filtered_summary[
                filtered_summary[team_filter_col].astype(str) == selected_team
            ].copy()

    sort_cols = [col for col in ["event", "kickoff_time", "fixture_id"] if col in filtered_fixtures.columns]
    if sort_cols:
        filtered_fixtures = filtered_fixtures.sort_values(sort_cols, na_position="last")

    if fixture_limit != "All available":
        group_col = None
        if "team_id" in filtered_fixtures.columns:
            group_col = "team_id"
        elif team_filter_col is not None and team_filter_col in filtered_fixtures.columns:
            group_col = team_filter_col

        if group_col is not None:
            filtered_fixtures = (
                filtered_fixtures.groupby(group_col, group_keys=False, dropna=True)
                .head(int(fixture_limit))
                .reset_index(drop=True)
            )
        else:
            filtered_fixtures = filtered_fixtures.head(int(fixture_limit)).reset_index(drop=True)

    if "difficulty" in filtered_fixtures.columns and difficulty_values.notna().any():
        filtered_difficulty = pd.to_numeric(filtered_fixtures["difficulty"], errors="coerce")
        filtered_fixtures = filtered_fixtures[
            filtered_difficulty.between(selected_fdr_range[0], selected_fdr_range[1])
        ].copy()

    if "next_3_fdr_avg" in filtered_summary.columns and difficulty_values.notna().any():
        filtered_next_3 = pd.to_numeric(filtered_summary["next_3_fdr_avg"], errors="coerce")
        filtered_summary = filtered_summary[
            filtered_next_3.between(selected_fdr_range[0], selected_fdr_range[1])
        ].copy()

    metric_source = filtered_summary if not filtered_summary.empty else pd.DataFrame()
    if not metric_source.empty and "team_id" in metric_source.columns:
        teams_shown = int(metric_source["team_id"].nunique())
    elif not metric_source.empty and team_filter_col is not None and team_filter_col in metric_source.columns:
        teams_shown = int(metric_source[team_filter_col].nunique())
    else:
        teams_shown = int(filtered_fixtures["team_id"].nunique()) if "team_id" in filtered_fixtures.columns else 0

    next_3_avg = (
        pd.to_numeric(metric_source["next_3_fdr_avg"], errors="coerce")
        if not metric_source.empty and "next_3_fdr_avg" in metric_source.columns
        else pd.Series(dtype=float)
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teams Shown", teams_shown)
    c2.metric("Easiest Next 3 FDR Avg", f"{next_3_avg.min():.2f}" if next_3_avg.notna().any() else "0.00")
    c3.metric("Hardest Next 3 FDR Avg", f"{next_3_avg.max():.2f}" if next_3_avg.notna().any() else "0.00")
    c4.metric("Total Upcoming Fixtures Shown", len(filtered_fixtures))

    if not filtered_summary.empty and "next_3_fdr_avg" in filtered_summary.columns:
        filtered_summary = filtered_summary.sort_values("next_3_fdr_avg", na_position="last")

    st.markdown('<div class="comparison-banner">Team Fixture Summary</div>', unsafe_allow_html=True)
    if filtered_summary.empty:
        st.info("No team fixture summary data is available for the selected filters.")
    else:
        st.dataframe(
            style_table(format_fixture_summary_table(filtered_summary)),
            use_container_width=True
        )

    detailed_fixtures = filtered_fixtures.copy()
    if (
        not summary_df.empty
        and "team_id" in detailed_fixtures.columns
        and "team_id" in summary_df.columns
    ):
        summary_avg_cols = [
            col for col in ["team_id", "next_3_fdr_avg", "next_5_fdr_avg"] if col in summary_df.columns
        ]
        detailed_fixtures = detailed_fixtures.merge(
            summary_df[summary_avg_cols].drop_duplicates(subset=["team_id"]),
            on="team_id",
            how="left",
        )

    st.markdown('<div class="comparison-banner">Detailed Upcoming Fixtures</div>', unsafe_allow_html=True)
    if detailed_fixtures.empty:
        st.info("No upcoming fixtures match the selected filters.")
    else:
        st.dataframe(
            style_table(format_upcoming_fixtures_table(detailed_fixtures)),
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
                f'<div class="comparison-banner">GW1 Squad Builder - {infer_current_fpl_season_label()} Season</div>',
                unsafe_allow_html=True
            )

            st.write(
                "This section builds an initial Gameweek 1 squad automatically using the previous season summary and the live current player pool"
            )

            controls_col_a, controls_col_b = st.columns([1, 1], gap="large")

            with controls_col_a:
                include_unmatched = st.toggle(
                    "Include unmatched players",
                    value=True,
                    help="Keep players without a previous-season match in the pool using fallback scoring."
                )

            with controls_col_b:
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

                metric_row_1 = st.columns(4, gap="large")
                metric_row_1[0].metric("Candidate Players", len(candidate_pool))
                metric_row_1[1].metric("Matched Players", matched_count)
                metric_row_1[2].metric("Unmatched Players", unmatched_count)
                metric_row_1[3].metric("Hybrid Squad Cost", f"{hybrid_cost:.1f}")

                metric_row_2 = st.columns(4, gap="large")
                metric_row_2[0].metric("Hybrid Squad Players", len(hybrid_squad))
                metric_row_2[1].metric("Hybrid Score Total", f"{hybrid_score:.2f}")
                metric_row_2[2].metric(
                    "Solve Status",
                    summary_df.iloc[0]["Solve_Status"] if not summary_df.empty else "Unknown"
                )
                metric_row_2[3].empty()

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

    required_transfer_columns = {"name", "team", "position", "price_m", "predicted_points"}
    missing_transfer_columns = sorted(required_transfer_columns - set(predictions_df.columns))
    if missing_transfer_columns:
        st.error(
            "Transfer Assistant is unavailable because required prediction data is missing: "
            f"{', '.join(missing_transfer_columns)}."
        )
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

    squad_widget_keys = {
        "GK": "transfer_squad_gk",
        "DEF": "transfer_squad_def",
        "MID": "transfer_squad_mid",
        "FWD": "transfer_squad_fwd",
    }
    pending_squad_names = st.session_state.pop("transfer_pending_squad_names", None)
    if pending_squad_names is not None:
        pending_names = set(pending_squad_names)
        for position, label_map in [
            ("GK", gk_label_map),
            ("DEF", def_label_map),
            ("MID", mid_label_map),
            ("FWD", fwd_label_map),
        ]:
            st.session_state[squad_widget_keys[position]] = [
                label for label, name in label_map.items() if name in pending_names
            ]
        if "transfer_pending_money_in_bank" in st.session_state:
            st.session_state["transfer_money_in_bank"] = st.session_state.pop(
                "transfer_pending_money_in_bank"
            )

    col1, col2 = st.columns(2)

    with col1:
        selected_gk_labels = st.multiselect(
            "Select 2 Goalkeepers",
            options=gk_options,
            key=squad_widget_keys["GK"],
        )

        selected_def_labels = st.multiselect(
            "Select 5 Defenders",
            options=def_options,
            key=squad_widget_keys["DEF"],
        )

    with col2:
        selected_mid_labels = st.multiselect(
            "Select 5 Midfielders",
            options=mid_options,
            key=squad_widget_keys["MID"],
        )

        selected_fwd_labels = st.multiselect(
            "Select 3 Forwards",
            options=fwd_options,
            key=squad_widget_keys["FWD"],
        )

    money_in_bank = st.number_input(
        "Money In Bank",
        min_value=0.0,
        step=0.1,
        key="transfer_money_in_bank",
    )

    transfer_count = st.radio(
        "Number of Transfers Available",
        [0, 1, 2, "Unlimited"],
        horizontal=True,
    )
    if transfer_count == 0:
        st.info("Transfer Action: No transfer action selected.")
    elif transfer_count == "Unlimited":
        st.info("Wildcard mode will recommend the best full squad rebuild within your available squad value and bank.")

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
    st.session_state["transfer_assistant_squad_df"] = current_squad_df.copy()

    if len(selected_names) > 0:
        current_squad_df = predictions_df[predictions_df["name"].isin(selected_names)].copy().reset_index(drop=True)
        current_squad_signature = (
            tuple(sorted(current_squad_df["name"].astype(str).tolist())),
            round(float(money_in_bank), 1),
        )
        if st.session_state.get("transfer_recommendation_squad_signature") != current_squad_signature:
            st.session_state.pop("transfer_one_recommendations", None)
            st.session_state.pop("transfer_two_recommendations", None)
            st.session_state.pop("transfer_wildcard_squad", None)
            st.session_state["transfer_recommendation_squad_signature"] = current_squad_signature
        st.session_state["transfer_assistant_squad_df"] = current_squad_df.copy()
        applied_target_names = set(st.session_state.get("transfer_applied_target_names", []))
        if (
            st.session_state.get("transfer_has_applied_recommendation", False)
            and applied_target_names
            and applied_target_names != set(current_squad_df["name"].astype(str))
        ):
            st.session_state["transfer_has_applied_recommendation"] = False

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

        applied_message = st.session_state.pop("transfer_applied_message", "")
        if applied_message:
            st.success(applied_message)

        if st.session_state.get("transfer_has_applied_recommendation", False):
            st.markdown('<div class="comparison-banner">Updated Squad After Applied Recommendation</div>', unsafe_allow_html=True)
            st.dataframe(
                style_table(format_transfer_assistant_squad(current_squad_df)),
                use_container_width=True,
                hide_index=True,
            )
            if st.button("Undo Last Applied Recommendation", key="transfer_undo_recommendation"):
                undo_names = st.session_state.get("transfer_undo_squad_names")
                if undo_names:
                    st.session_state["transfer_pending_squad_names"] = list(undo_names)
                    st.session_state["transfer_pending_money_in_bank"] = float(
                        st.session_state.get("transfer_undo_money_in_bank", money_in_bank)
                    )
                    st.session_state["transfer_has_applied_recommendation"] = False
                    st.session_state["transfer_applied_message"] = "Last applied recommendation was undone."
                    st.rerun()
                else:
                    st.warning("The previous squad is unavailable, so the last recommendation cannot be undone safely.")

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

            st.caption(
                "Fantasy IQ recommends the highest predicted-points XI that satisfies FPL formation rules."
            )

            recommended_xi, recommended_formation = build_transfer_assistant_starting_xi(
                current_squad_df
            )
            recommended_names = set(recommended_xi["name"].tolist()) if not recommended_xi.empty else set()

            gk_name_to_label = {name: label for label, name in gk_label_map.items()}
            def_name_to_label = {name: label for label, name in def_label_map.items()}
            mid_name_to_label = {name: label for label, name in mid_label_map.items()}
            fwd_name_to_label = {name: label for label, name in fwd_label_map.items()}

            recommended_gk_labels = [
                gk_name_to_label[name]
                for name in recommended_names
                if name in gk_name_to_label
            ]
            recommended_def_labels = [
                def_name_to_label[name]
                for name in recommended_names
                if name in def_name_to_label
            ]
            recommended_mid_labels = [
                mid_name_to_label[name]
                for name in recommended_names
                if name in mid_name_to_label
            ]
            recommended_fwd_labels = [
                fwd_name_to_label[name]
                for name in recommended_names
                if name in fwd_name_to_label
            ]

            xi_widget_keys = {
                "gk": "transfer_starting_gk",
                "def": "transfer_starting_def",
                "mid": "transfer_starting_mid",
                "fwd": "transfer_starting_fwd",
            }
            squad_signature = tuple(sorted(selected_names))
            if st.session_state.get("transfer_xi_squad_signature") != squad_signature:
                st.session_state["transfer_xi_squad_signature"] = squad_signature
                st.session_state[xi_widget_keys["gk"]] = (
                    recommended_gk_labels[0] if recommended_gk_labels else selected_gk_labels[0]
                )
                st.session_state[xi_widget_keys["def"]] = recommended_def_labels
                st.session_state[xi_widget_keys["mid"]] = recommended_mid_labels
                st.session_state[xi_widget_keys["fwd"]] = recommended_fwd_labels

            if st.button(
                "Auto-select Best Starting XI",
                disabled=recommended_xi.empty,
                key="transfer_auto_select_xi",
            ):
                st.session_state[xi_widget_keys["gk"]] = recommended_gk_labels[0]
                st.session_state[xi_widget_keys["def"]] = recommended_def_labels
                st.session_state[xi_widget_keys["mid"]] = recommended_mid_labels
                st.session_state[xi_widget_keys["fwd"]] = recommended_fwd_labels
                st.rerun()

            if recommended_xi.empty:
                st.warning(
                    "A complete automatic XI cannot be recommended because required prediction values are unavailable. "
                    "You can still select the starting XI manually."
                )
            else:
                st.caption(f"Recommended formation: {recommended_formation}")

            starting_gk_label = st.selectbox(
                "Starting Goalkeeper",
                options=selected_gk_labels,
                key=xi_widget_keys["gk"],
            )

            col3, col4 = st.columns(2)

            with col3:
                starting_def_labels = st.multiselect(
                    "Starting Defenders (3 to 5)",
                    options=selected_def_labels,
                    key=xi_widget_keys["def"],
                )

                starting_mid_labels = st.multiselect(
                    "Starting Midfielders (2 to 5)",
                    options=selected_mid_labels,
                    key=xi_widget_keys["mid"],
                )

            with col4:
                starting_fwd_labels = st.multiselect(
                    "Starting Forwards (1 to 3)",
                    options=selected_fwd_labels,
                    key=xi_widget_keys["fwd"],
                )

            starting_names = (
                [gk_label_map[starting_gk_label]]
                + [def_label_map[label] for label in starting_def_labels]
                + [mid_label_map[label] for label in starting_mid_labels]
                + [fwd_label_map[label] for label in starting_fwd_labels]
            )

            starting_df = current_squad_df[current_squad_df["name"].isin(starting_names)].copy()
            starting_valid = validate_starting_xi(starting_df)
            starting_prediction_values = (
                pd.to_numeric(starting_df["predicted_points"], errors="coerce")
                if "predicted_points" in starting_df.columns
                else pd.Series(dtype="float64")
            )
            starting_predictions_complete = (
                len(starting_prediction_values) == 11
                and starting_prediction_values.notna().all()
            )
            squad_prediction_values = pd.to_numeric(
                current_squad_df["predicted_points"], errors="coerce"
            )
            squad_price_values = pd.to_numeric(
                current_squad_df["price_m"], errors="coerce"
            )
            rating_inputs_complete = (
                starting_predictions_complete
                and len(squad_prediction_values) == 15
                and squad_prediction_values.notna().all()
                and squad_price_values.notna().all()
            )

            st.write(f"Starting XI selected: **{len(starting_df)}**")

            if starting_valid:
                st.success("Your starting XI is valid.")
            else:
                st.warning(
                    "Your starting XI is not valid. A valid XI needs exactly 11 players, exactly 1 GK, at least 3 DEF, at least 2 MID, and at least 1 FWD."
                )
                st.info("Complete or auto-select a valid starting XI to view the team rating.")

            if starting_valid and not rating_inputs_complete:
                st.warning(
                    "Team rating and transfer recommendations are unavailable because one or more "
                    "squad players do not have valid predicted points or price data."
                )

            if starting_valid and rating_inputs_complete:
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

                st.markdown('<div class="comparison-banner">Captain & Vice-Captain Recommendation</div>', unsafe_allow_html=True)
                starting_captain_df = format_captain_recommendation_table(
                    starting_df,
                    top_n=2,
                    roles=["Captain", "Vice-Captain"],
                )
                if len(starting_captain_df) < 2:
                    st.info("At least two starting XI players are needed for captain and vice-captain recommendations.")
                else:
                    st.dataframe(
                        style_table(starting_captain_df),
                        use_container_width=True,
                    )

                st.markdown('<div class="comparison-banner">Team Rating</div>', unsafe_allow_html=True)
                team_rating = build_team_rating_summary(current_squad_df, starting_df, bench_df)
                rating_interpretation = describe_team_rating(float(team_rating["overall_rating"]))

                rating_col1, rating_col2, rating_col3 = st.columns(3)
                rating_col1.metric("Overall Team Rating", f"{team_rating['overall_rating']:.0f}/100")
                rating_col2.metric("Team Strength Level", rating_interpretation)
                rating_col3.metric("Starting XI Strength", f"{team_rating['starting_xi_strength']:.2f}")

                risk_penalty = float(team_rating["risk_penalty"])
                risk_penalty_text = "0" if risk_penalty == 0 else f"-{risk_penalty:.0f}"
                rating_col4, rating_col5, rating_col6 = st.columns(3)
                rating_col4.metric("Bench Strength", f"{team_rating['bench_strength']:.2f}")
                rating_col5.metric("Fixture Adjustment", f"{team_rating['fixture_adjustment']:+.0f}")
                rating_col6.metric("Risk Penalty", risk_penalty_text)

                st.caption(
                    f"Overall rating: {rating_interpretation}. "
                    "Bands: 80–100 Excellent; 65–79 Strong; 50–64 Average; "
                    "35–49 Weak; below 35 Needs attention."
                )

                st.info(
                    f"{team_rating['risk_warning_summary']} "
                    f"{team_rating['fixture_summary']}."
                )
                st.dataframe(
                    style_table(format_team_rating_breakdown(team_rating)),
                    use_container_width=True,
                )

                run_one_transfer = False
                run_two_transfers = False
                run_wildcard = False
                if transfer_count == 1:
                    run_one_transfer = st.button("Generate Best 1 Transfer")
                elif transfer_count == 2:
                    run_two_transfers = st.button("Generate Best 2 Transfers")
                elif transfer_count == "Unlimited":
                    run_wildcard = st.button("Generate Wildcard Recommendation")

                safe_recommendation_pool = predictions_df.copy()
                safe_recommendation_pool["predicted_points"] = pd.to_numeric(
                    safe_recommendation_pool["predicted_points"], errors="coerce"
                )
                safe_recommendation_pool["price_m"] = pd.to_numeric(
                    safe_recommendation_pool["price_m"], errors="coerce"
                )
                safe_recommendation_pool = safe_recommendation_pool.dropna(
                    subset=["name", "team", "position", "predicted_points", "price_m"]
                ).reset_index(drop=True)

                def queue_applied_squad(updated_squad_df: pd.DataFrame, updated_bank: float) -> None:
                    validated_bank = pd.to_numeric(updated_bank, errors="coerce")
                    if pd.isna(validated_bank) or float(validated_bank) < -1e-9:
                        st.warning(
                            "Recommendation was not applied because the selected changes would exceed "
                            "the available transfer budget."
                        )
                        return
                    st.session_state["transfer_undo_squad_names"] = current_squad_df["name"].tolist()
                    st.session_state["transfer_undo_money_in_bank"] = float(money_in_bank)
                    st.session_state["transfer_pending_squad_names"] = updated_squad_df["name"].tolist()
                    st.session_state["transfer_pending_money_in_bank"] = max(
                        0.0, round(float(validated_bank), 1)
                    )
                    st.session_state["transfer_has_applied_recommendation"] = True
                    st.session_state["transfer_applied_target_names"] = updated_squad_df["name"].tolist()
                    st.session_state["transfer_applied_message"] = (
                        "Recommendation applied to your Transfer Assistant squad."
                    )
                    st.session_state.pop("transfer_one_recommendations", None)
                    st.session_state.pop("transfer_two_recommendations", None)
                    st.session_state.pop("transfer_wildcard_squad", None)
                    st.rerun()

                if run_one_transfer:
                    try:
                        with st.spinner("Generating best 1-transfer recommendations..."):
                            st.session_state["transfer_one_recommendations"] = recommend_best_one_transfer(
                                current_squad_df=current_squad_df,
                                predictions_df=safe_recommendation_pool,
                                money_in_bank=money_in_bank,
                                starting_names=starting_names,
                                verbose=False,
                            )
                    except Exception as exc:
                        st.session_state["transfer_one_recommendations"] = pd.DataFrame()
                        st.warning(f"A 1-transfer recommendation could not be generated safely. Details: {exc}")

                if run_two_transfers:
                    try:
                        with st.spinner("Generating best 2-transfer recommendations..."):
                            st.session_state["transfer_two_recommendations"] = recommend_best_two_transfers(
                                current_squad_df=current_squad_df,
                                predictions_df=safe_recommendation_pool,
                                money_in_bank=money_in_bank,
                                starting_names=starting_names,
                                verbose=False,
                            )
                    except Exception as exc:
                        st.session_state["transfer_two_recommendations"] = pd.DataFrame()
                        st.warning(f"A 2-transfer recommendation could not be generated safely. Details: {exc}")

                if run_wildcard:
                    try:
                        wildcard_budget = float(squad_cost + money_in_bank)
                        with st.spinner("Generating the best Wildcard squad..."):
                            st.session_state["transfer_wildcard_squad"] = optimize_best_15_squad(
                                predictions_df=safe_recommendation_pool,
                                budget_limit=wildcard_budget,
                                club_limit=3,
                                verbose=False,
                            )
                    except Exception as exc:
                        st.session_state["transfer_wildcard_squad"] = pd.DataFrame()
                        st.warning(
                            "A safe Wildcard recommendation could not be generated from the available "
                            f"predictions. Details: {exc}"
                        )

                if transfer_count == 1 and "transfer_one_recommendations" in st.session_state:
                    one_transfer_df = st.session_state["transfer_one_recommendations"]
                    st.markdown('<div class="comparison-banner">Best 1-Transfer Recommendations</div>', unsafe_allow_html=True)
                    if one_transfer_df.empty:
                        st.info("Transfer Action: No clear transfer upgrade found.")
                        st.info("No valid 1-transfer recommendations were found.")
                    else:
                        useful_one_transfer = pd.to_numeric(
                            one_transfer_df["predicted_points_gain"], errors="coerce"
                        ).gt(0).any()
                        if useful_one_transfer:
                            st.success("Transfer Action: Transfer upgrade found.")
                        else:
                            st.info("Transfer Action: No clear transfer upgrade found.")
                        for option_number, (_, recommendation) in enumerate(
                            one_transfer_df.head(10).iterrows(), start=1
                        ):
                            with st.expander(f"Option {option_number}", expanded=option_number <= 3):
                                display_recommendation = recommendation.copy()
                                display_recommendation["outgoing_is_starter"] = (
                                    str(recommendation.get("player_out", "")) in starting_names
                                )
                                option_df = format_one_transfer_table(
                                    pd.DataFrame([display_recommendation])
                                )
                                st.dataframe(
                                    style_table(option_df),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                if st.button(
                                    "Apply This Recommendation",
                                    key=f"apply_one_transfer_{option_number}",
                                ):
                                    updated_squad_df, apply_error = apply_transfer_recommendation_to_squad(
                                        current_squad_df,
                                        safe_recommendation_pool,
                                        [str(recommendation.get("player_out", ""))],
                                        [str(recommendation.get("player_in", ""))],
                                    )
                                    if apply_error:
                                        st.warning(apply_error)
                                    else:
                                        queue_applied_squad(
                                            updated_squad_df,
                                            float(recommendation.get("remaining_money_in_bank", money_in_bank)),
                                        )

                if transfer_count == 2 and "transfer_two_recommendations" in st.session_state:
                    two_transfer_df = st.session_state["transfer_two_recommendations"]
                    st.markdown('<div class="comparison-banner">Best 2-Transfer Recommendations</div>', unsafe_allow_html=True)
                    if two_transfer_df.empty:
                        st.info("Transfer Action: No clear transfer upgrade found.")
                        st.info("No valid 2-transfer recommendations were found.")
                    else:
                        useful_two_transfer = pd.to_numeric(
                            two_transfer_df["predicted_points_gain"], errors="coerce"
                        ).gt(0).any()
                        if useful_two_transfer:
                            st.success("Transfer Action: Transfer upgrade found.")
                        else:
                            st.info("Transfer Action: No clear transfer upgrade found.")
                        for option_number, (_, recommendation) in enumerate(
                            two_transfer_df.head(10).iterrows(), start=1
                        ):
                            total_gain = pd.to_numeric(
                                recommendation.get("predicted_points_gain"), errors="coerce"
                            )
                            with st.expander(f"Option {option_number}", expanded=option_number <= 3):
                                if pd.notna(total_gain):
                                    st.metric("Total Predicted Points Gain", f"{float(total_gain):.2f}")
                                option_df = build_two_transfer_display(
                                    recommendation,
                                    safe_recommendation_pool,
                                    starting_names,
                                )
                                st.dataframe(
                                    style_table(option_df),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                selected_transfer_numbers = [
                                    transfer_number
                                    for transfer_number in (1, 2)
                                    if st.checkbox(
                                        (
                                            f"Select transfer {transfer_number}: "
                                            f"{recommendation.get(f'player_out_{transfer_number}', '')} → "
                                            f"{recommendation.get(f'player_in_{transfer_number}', '')}"
                                        ),
                                        key=f"select_two_transfer_{option_number}_{transfer_number}",
                                    )
                                ]
                                apply_full_two = st.button(
                                    "Apply Full 2-Transfer Option",
                                    key=f"apply_full_two_transfer_{option_number}",
                                )
                                apply_selected_two = st.button(
                                    "Apply Selected From This Option",
                                    key=f"apply_selected_two_transfer_{option_number}",
                                )
                                if apply_full_two or apply_selected_two:
                                    if apply_selected_two and not selected_transfer_numbers:
                                        st.warning("Select at least one transfer before applying selected changes.")
                                        continue
                                    transfers_to_apply = (
                                        [1, 2] if apply_full_two else selected_transfer_numbers
                                    )
                                    outgoing_names = [
                                        str(recommendation.get(f"player_out_{number}", ""))
                                        for number in transfers_to_apply
                                    ]
                                    incoming_names = [
                                        str(recommendation.get(f"player_in_{number}", ""))
                                        for number in transfers_to_apply
                                    ]
                                    updated_squad_df, apply_error = apply_transfer_recommendation_to_squad(
                                        current_squad_df,
                                        safe_recommendation_pool,
                                        outgoing_names,
                                        incoming_names,
                                    )
                                    if apply_error:
                                        st.warning(apply_error)
                                    else:
                                        updated_bank = float(
                                            money_in_bank
                                            + pd.to_numeric(current_squad_df["price_m"], errors="coerce").sum()
                                            - pd.to_numeric(updated_squad_df["price_m"], errors="coerce").sum()
                                        )
                                        queue_applied_squad(
                                            updated_squad_df,
                                            updated_bank,
                                        )

                if transfer_count == "Unlimited" and "transfer_wildcard_squad" in st.session_state:
                    wildcard_squad_df = st.session_state["transfer_wildcard_squad"]
                    st.markdown('<div class="comparison-banner">Wildcard Recommendation</div>', unsafe_allow_html=True)
                    if wildcard_squad_df.empty:
                        st.info("Transfer Action: No clear wildcard upgrade found.")
                        st.info("A safe Wildcard recommendation could not be generated from the available predictions.")
                    else:
                        wildcard_cost = float(
                            pd.to_numeric(wildcard_squad_df["price_m"], errors="coerce").sum()
                        )
                        wildcard_remaining_bank = float(squad_cost + money_in_bank - wildcard_cost)
                        current_names = set(current_squad_df["name"])
                        wildcard_names = set(wildcard_squad_df["name"])
                        players_changed = len(current_names - wildcard_names)
                        wildcard_gain = float(
                            pd.to_numeric(wildcard_squad_df["predicted_points"], errors="coerce").sum()
                            - pd.to_numeric(current_squad_df["predicted_points"], errors="coerce").sum()
                        )

                        if players_changed > 0:
                            st.success("Transfer Action: Wildcard changes found.")
                        else:
                            st.info(
                                "Transfer Action: No wildcard changes recommended. Your current squad "
                                "already matches the model's recommended squad."
                            )

                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        summary_col1.metric("Total Players Changed", players_changed)
                        summary_col2.metric("Estimated Predicted Points Gain", f"{wildcard_gain:.2f}")
                        summary_col3.metric("Remaining Money In Bank", f"{wildcard_remaining_bank:.1f}")

                        wildcard_changes_df = build_wildcard_change_display(
                            current_squad_df=current_squad_df,
                            wildcard_squad_df=wildcard_squad_df,
                            starting_names=starting_names,
                            remaining_bank=wildcard_remaining_bank,
                        )
                        if wildcard_changes_df.empty:
                            st.success("Your current squad already matches the recommended Wildcard squad.")
                        else:
                            selected_wildcard_rows: list[int] = []
                            for change_number, (_, change) in enumerate(
                                wildcard_changes_df.iterrows(), start=1
                            ):
                                if st.checkbox(
                                    (
                                        f"Select change {change_number}: {change['Player Out']} → "
                                        f"{change['Player In']}"
                                    ),
                                    key=f"select_wildcard_change_{change_number}",
                                ):
                                    selected_wildcard_rows.append(change_number - 1)
                                st.dataframe(
                                    style_table(pd.DataFrame([change])),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                            apply_selected_wildcard = st.button(
                                "Apply Selected Changes",
                                key="apply_selected_wildcard_changes",
                            )
                            apply_all_wildcard = st.button(
                                "Apply All Changes",
                                key="apply_all_wildcard_changes",
                            )
                            if apply_selected_wildcard or apply_all_wildcard:
                                if apply_selected_wildcard and not selected_wildcard_rows:
                                    st.warning("Select at least one Wildcard change before applying selected changes.")
                                else:
                                    changes_to_apply = (
                                        wildcard_changes_df
                                        if apply_all_wildcard
                                        else wildcard_changes_df.iloc[selected_wildcard_rows]
                                    )
                                    outgoing_names = changes_to_apply["Player Out"].dropna().astype(str).tolist()
                                    incoming_names = changes_to_apply["Player In"].dropna().astype(str).tolist()
                                    updated_squad_df, apply_error = apply_transfer_recommendation_to_squad(
                                        current_squad_df,
                                        safe_recommendation_pool,
                                        outgoing_names,
                                        incoming_names,
                                    )
                                    if apply_error:
                                        st.warning(apply_error)
                                    else:
                                        updated_bank = float(
                                            money_in_bank
                                            + pd.to_numeric(current_squad_df["price_m"], errors="coerce").sum()
                                            - pd.to_numeric(updated_squad_df["price_m"], errors="coerce").sum()
                                        )
                                        queue_applied_squad(updated_squad_df, updated_bank)
