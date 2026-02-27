# app.py
# Streamlit Macro Dashboard (auto-updating)
#
# Setup:
# 1) Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
# 2) Set env var: export FRED_API_KEY="..."
# 3) pip install streamlit pandas requests yfinance python-dateutil
# 4) streamlit run app.py

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
import time
import random

# -----------------------------
# Config
# -----------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
FRED_BASE = "https://api.stlouisfed.org/fred"

DEFAULT_LOOKBACK_YEARS = 8
CHART_YEARS = 5  # chart window

# Known-good FRED series IDs (except PMI where we search dynamically)
SERIES = {
    "real_gdp": "GDPC1",            # Real Gross Domestic Product (quarterly)
    "unrate": "UNRATE",             # Unemployment rate (monthly)
    "jobless_claims": "ICSA",       # Initial Claims (weekly)
    "cpi": "CPIAUCSL",              # CPI (monthly)
    "core_cpi": "CPILFESL",         # Core CPI (monthly)
    "fed_funds": "FEDFUNDS",        # Effective target proxy (monthly)
    "dgs10": "DGS10",               # 10Y treasury (daily)
    "dgs2": "DGS2",                 # 2Y treasury (daily)
    "ig_oas": "BAMLC0A0CM",         # IG option-adjusted spread (daily) :contentReference[oaicite:0]{index=0}
    "hy_oas": "BAMLH0A0HYM2",       # HY option-adjusted spread (daily) :contentReference[oaicite:1]{index=1}
    "nfci": "NFCI",                 # Chicago Fed NFCI (weekly) :contentReference[oaicite:2]{index=2}
}

# PMI: we’ll find series IDs via FRED search at runtime
PMI_QUERIES = {
    "mfg_activity": "ISM Manufacturing PMI",
    "broad_activity": "ISM Non-Manufacturing PMI",
}

PMI_FALLBACK_IDS = {
    "mfg_activity": ["NAPM"],                 # ISM Manufacturing PMI
    "broad_activity": ["NAPMNOI", "NAPMNMI"],   # try these in order (some accounts/regions differ)
}

MARKET = {
    "equity": "SPY",  # S&P 500 proxy
}

# -----------------------------
# Helpers
# -----------------------------
def _require_fred_key():
    if not FRED_API_KEY:
        st.error("Missing FRED_API_KEY env var. Set it, then rerun.")
        st.stop()

def fred_get(endpoint: str, params: dict) -> dict:
    _require_fred_key()
    p = {"api_key": FRED_API_KEY, "file_type": "json"}
    p.update(params)
    r = requests.get(f"{FRED_BASE}/{endpoint}", params=p, timeout=30)
    if not r.ok:
        raise RuntimeError(f"FRED error {r.status_code}: {r.text}")
    return r.json()

@st.cache_data(ttl=6 * 60 * 60)
def fred_search_series(search_text: str, limit: int = 10) -> pd.DataFrame:
    data = fred_get("series/search", {"search_text": search_text, "limit": limit})
    rows = data.get("seriess", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # keep a few useful columns
    cols = [c for c in ["id", "title", "frequency", "units", "seasonal_adjustment"] if c in df.columns]
    return df[cols]

@st.cache_data(ttl=6 * 60 * 60)
def fred_series(series_id: str, start_date: str) -> pd.Series:
    data = fred_get(
        "series/observations",
        {"series_id": series_id, "observation_start": start_date},
    )
    obs = data.get("observations", [])
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"].dropna().sort_index()
    s.name = series_id
    return s

def to_monthly(s: pd.Series) -> pd.Series:
    # robust monthly resample (last value in month)
    return s.resample("ME").last()  # month-end

def yoy(s: pd.Series, periods: int) -> pd.Series:
    # YoY % change (periods=12 for monthly, 4 for quarterly)
    return (s / s.shift(periods) - 1.0) * 100.0

def pct_change(s: pd.Series, months: int) -> float:
    if s.empty:
        return float("nan")
    end = s.dropna().iloc[-1]
    start_idx = s.dropna().index[-1] - relativedelta(months=months)
    start = s.dropna().loc[:start_idx].iloc[-1] if not s.dropna().loc[:start_idx].empty else float("nan")
    delta = end - start
    return float(delta.iloc[0]) if isinstance(delta, pd.Series) else float(delta)

def last_value(s: pd.Series) -> float:
    if s is None or s.dropna().empty:
        return float("nan")
    v = s.dropna().iloc[-1]
    return float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)

def fmt(x, unit=""):
    if pd.isna(x):
        return "—"
    if unit == "%":
        return f"{x:.2f}%"
    if unit == "idx":
        return f"{x:.1f}"
    if unit == "bps":
        return f"{x*100:.0f} bps"  # if x is percent units
    if unit == "k":
        return f"{x:,.0f}k"
    return f"{x:,.2f}"

def status_from_change(delta_3m: float, delta_6m: float, higher_is_better: bool) -> str:
    # Simple traffic light using direction
    # Higher is better => positive deltas good
    # Higher is worse => negative deltas good
    if pd.isna(delta_3m) or pd.isna(delta_6m):
        return "⚪"
    score = 0
    score += 1 if (delta_3m > 0) == higher_is_better else -1 if (delta_3m < 0) == higher_is_better else 0
    score += 1 if (delta_6m > 0) == higher_is_better else -1 if (delta_6m < 0) == higher_is_better else 0
    if score >= 1:
        return "🟢"
    if score == 0:
        return "🟡"
    return "🔴"

def score_indicator(value: float, good_when_high: bool, neutral_band: tuple[float, float] | None = None) -> int:
    # Returns +1 / 0 / -1 (simple and tunable)
    if pd.isna(value):
        return 0
    if neutral_band is not None and neutral_band[0] <= value <= neutral_band[1]:
        return 0
    return 1 if (value > 0) == good_when_high else -1

import time
import random

@st.cache_data(ttl=24 * 60 * 60)  # cache 24h to reduce rate limits
def fetch_spy(start: str, ticker: str = "SPY", max_retries: int = 6) -> pd.Series:
    last_err = None

    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                progress=False,
                auto_adjust=True,
                threads=False,   # helps a bit on rate limits
            )
            if df is None or df.empty or "Close" not in df.columns:
                raise RuntimeError("No data returned from yfinance")

            s = df["Close"].copy()
            s.index = pd.to_datetime(s.index)
            s.name = ticker
            return s

        except Exception as e:
            last_err = e
            # exponential backoff + jitter
            sleep_s = min(60, (2 ** attempt)) + random.random()
            time.sleep(sleep_s)

    raise RuntimeError(f"yfinance failed after {max_retries} attempts: {last_err}")

def first_working_series_id(candidates: list[str], start_date: str) -> str:
    last_err = None
    for sid in candidates:
        try:
            _ = fred_series(sid, start_date)
            return sid
        except Exception as e:
            last_err = e
    raise ValueError(f"No working FRED series from candidates={candidates}. Last error: {last_err}")

def help_text(text: str):
    st.caption(f"ℹ️ {text}")
    
def is_rising(s: pd.Series, months: int = 3) -> bool:
    """True if last value > value ~N months ago (monthly-resampled)."""
    if s is None or s.empty or not isinstance(s.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return False
    sm = to_monthly(s.dropna())
    if sm.empty:
        return False
    end = sm.iloc[-1]
    start_idx = sm.index[-1] - relativedelta(months=months)
    prior = sm.loc[:start_idx]
    if prior.empty:
        return False
    start = prior.iloc[-1]
    return float(end) > float(start)

def is_falling(s: pd.Series, months: int = 3) -> bool:
    return is_rising(-s, months=months)

def badge(ok: bool, good: str = "✅", bad: str = "⚠️") -> str:
    return good if ok else bad
   
# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Macro Dashboard", layout="wide")
st.title("Macro Dashboard (Auto-updating)")

start = (date.today() - relativedelta(years=DEFAULT_LOOKBACK_YEARS)).isoformat()
chart_start = (date.today() - relativedelta(years=CHART_YEARS)).isoformat()

playbook = [
    {
        "Indicator": "Yield curve (10Y–2Y)",
        "Risk-on tends to like": "Equities (cyclicals), small caps",
        "Risk-off tends to like": "Long-duration Treasuries, defensives",
        "How to read": "Steepening is usually pro-growth; inversion/flattening is caution."
    },
    {
        "Indicator": "Financial Conditions (NFCI)",
        "Risk-on tends to like": "Equities, credit, high beta",
        "Risk-off tends to like": "Cash, Treasuries",
        "How to read": "Rising NFCI = tightening liquidity; falling = easing."
    },
    {
        "Indicator": "Fed Funds vs Core CPI",
        "Risk-on tends to like": "Equities (especially when easing starts)",
        "Risk-off tends to like": "Cash/short duration during hikes",
        "How to read": "Rates > inflation = restrictive; gap shrinking = pressure easing."
    },
    {
        "Indicator": "Activity proxies (IPMAN/INDPRO) momentum",
        "Risk-on tends to like": "Cyclicals, industrials, commodities",
        "Risk-off tends to like": "Defensives, quality",
        "How to read": "Rising momentum supports earnings; falling momentum = slowdown risk."
    },
    {
        "Indicator": "Jobless Claims (4W MA)",
        "Risk-on tends to like": "Equities broadly",
        "Risk-off tends to like": "Treasuries, defensives",
        "How to read": "Claims rising is an early recession alarm."
    },
    {
        "Indicator": "Credit spreads (IG/HY OAS)",
        "Risk-on tends to like": "Credit, equities, small caps",
        "Risk-off tends to like": "Treasuries, cash",
        "How to read": "Widening spreads = stress building (often pre-equity weakness)."
    },
    {
        "Indicator": "SPY trend (50d–200d)",
        "Risk-on tends to like": "Equities",
        "Risk-off tends to like": "Trend-following defensive stance (cash/bonds)",
        "How to read": "Positive = risk-on regime; negative = risk-off."
    },
]

with st.sidebar:
    st.header("Settings")
    st.write("Data: FRED + Yahoo Finance (SPY).")
    chart_years = st.slider("Chart window (years)", 1, 15, CHART_YEARS)
    st.caption("Tip: keep this to 3–7 years for readability.")
    chart_start = (date.today() - relativedelta(years=chart_years)).isoformat()

    st.divider()
    st.subheader("Asset Class Playbook")

    df_playbook = pd.DataFrame(playbook)
    df_playbook_inverted = df_playbook.set_index("Indicator").T
    st.dataframe(
        df_playbook_inverted,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Rule-of-thumb mapping for regime awareness. "
        "Not investment advice."
    )
# --- Load series
try:
    # PMI (dynamic search)
    st.subheader("Series overrides (advanced)")

    mfg_activity_id = st.text_input("Manufacturing activity (default IPMAN)", value="IPMAN")
    broad_activity_id = st.text_input("Broad activity (default INDPRO)", value="INDPRO")

    mfg_activity = fred_series(mfg_activity_id.strip(), start)
    broad_activity = fred_series(broad_activity_id.strip(), start)

    gdp = fred_series(SERIES["real_gdp"], start)
    unrate = fred_series(SERIES["unrate"], start)
    claims = fred_series(SERIES["jobless_claims"], start)

    cpi = fred_series(SERIES["cpi"], start)
    core_cpi = fred_series(SERIES["core_cpi"], start)

    fed_funds = fred_series(SERIES["fed_funds"], start)
    dgs10 = fred_series(SERIES["dgs10"], start)
    dgs2 = fred_series(SERIES["dgs2"], start)

    ig_oas = fred_series(SERIES["ig_oas"], start)
    hy_oas = fred_series(SERIES["hy_oas"], start)

    nfci = fred_series(SERIES["nfci"], start)

    try:
        spy = fetch_spy(start)
    except Exception as e:
        st.warning(f"Market data unavailable (rate-limited): {e}")
        spy = pd.Series(dtype=float)
except Exception as e:
    st.exception(e)
    st.stop()

# --- Transformations
gdp_yoy = yoy(gdp, 4)
cpi_yoy = yoy(cpi, 12)
core_cpi_yoy = yoy(core_cpi, 12)

# Yield curve (10y - 2y)
yc = (dgs10 - dgs2).dropna()
yc.name = "10Y-2Y"

# Jobless claims: 4-week moving avg
claims_4w = claims.rolling(4).mean().dropna()
claims_4w.name = "ICSA_4WMA"

# Equity: trend = 200-day vs 50-day MA (simple)
spy_50 = spy.rolling(50).mean()
spy_200 = spy.rolling(200).mean()
spy_trend = (spy_50 - spy_200).dropna()
spy_trend.name = "SPY_50minus200"

# --- Compute deltas (3m / 6m)
def deltas(s: pd.Series) -> tuple[float, float]:
    if s is None or s.empty:
        return float("nan"), float("nan")

    # Ensure datetime index
    if not isinstance(s.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return float("nan"), float("nan")

    sm = to_monthly(s)
    return pct_change(sm, 3), pct_change(sm, 6)

metrics = []

# Growth/Demand
mfg_activity_yoy = yoy(to_monthly(mfg_activity), 12)
broad_activity_yoy = yoy(to_monthly(broad_activity), 12)

mfg_d3, mfg_d6 = deltas(mfg_activity_yoy)
srv_d3, srv_d6 = deltas(broad_activity_yoy)

gdp_d3, gdp_d6 = deltas(gdp_yoy)

metrics += [
    ("Manufacturing Activity", last_value(mfg_activity), "", mfg_d3, mfg_d6, True),
    ("Broad Activity", last_value(broad_activity), "", srv_d3, srv_d6, True),
    ("Real GDP YoY", last_value(gdp_yoy), "%", gdp_d3, gdp_d6, True),
]

# Labor
un_d3, un_d6 = deltas(unrate)
cl_d3, cl_d6 = deltas(claims_4w)

metrics += [
    ("Unemployment", last_value(unrate), "%", un_d3, un_d6, False),  # higher worse
    ("Jobless Claims (4W MA)", last_value(claims_4w), "k", cl_d3, cl_d6, False),  # higher worse
]

# Inflation
cpi_d3, cpi_d6 = deltas(cpi_yoy)
core_d3, core_d6 = deltas(core_cpi_yoy)

metrics += [
    ("CPI YoY", last_value(cpi_yoy), "%", cpi_d3, cpi_d6, False),       # higher worse
    ("Core CPI YoY", last_value(core_cpi_yoy), "%", core_d3, core_d6, False),
]

# Monetary/Credit
ff_d3, ff_d6 = deltas(fed_funds)
yc_d3, yc_d6 = deltas(yc)
ig_d3, ig_d6 = deltas(ig_oas)
hy_d3, hy_d6 = deltas(hy_oas)

metrics += [
    ("Policy Rate (Fed Funds)", last_value(fed_funds), "%", ff_d3, ff_d6, None),  # context-dependent
    ("Yield Curve (10Y-2Y)", last_value(yc), "%", yc_d3, yc_d6, True),            # steeper usually better
    ("Credit Spreads (IG OAS)", last_value(ig_oas), "%", ig_d3, ig_d6, False),    # higher worse
    ("Credit Spreads (HY OAS)", last_value(hy_oas), "%", hy_d3, hy_d6, False),
]

# Markets/Conditions
nf_d3, nf_d6 = deltas(nfci)
tr_d3, tr_d6 = deltas(spy_trend)

metrics += [
    ("Financial Conditions (NFCI)", last_value(nfci), "", nf_d3, nf_d6, False),   # higher = tighter = worse
    ("Equity Trend (SPY 50-200)", last_value(spy_trend), "", tr_d3, tr_d6, True),
]

# --- Scoring (simple, tunable)
# Use 3m change as the main “direction” signal for score.
score = 0
score += score_indicator(mfg_d3, True, neutral_band=(-0.5, 0.5))
score += score_indicator(srv_d3, True, neutral_band=(-0.5, 0.5))
score += score_indicator(gdp_d3, True, neutral_band=(-0.2, 0.2))

score += score_indicator(-un_d3, True, neutral_band=(-0.05, 0.05))  # falling unemployment is good
score += score_indicator(-cl_d3, True, neutral_band=(-5_000, 5_000))

score += score_indicator(-cpi_d3, True, neutral_band=(-0.1, 0.1))
score += score_indicator(-core_d3, True, neutral_band=(-0.1, 0.1))

score += score_indicator(yc_d3, True, neutral_band=(-0.05, 0.05))
score += score_indicator(-ig_d3, True, neutral_band=(-0.05, 0.05))
score += score_indicator(-hy_d3, True, neutral_band=(-0.05, 0.05))

score += score_indicator(-nf_d3, True, neutral_band=(-0.05, 0.05))
score += score_indicator(tr_d3, True, neutral_band=(-0.5, 0.5))

def regime_from_score(s: int) -> str:
    if s >= 5:
        return "Early/Mid Expansion"
    if 1 <= s <= 4:
        return "Late Cycle"
    if -3 <= s <= 0:
        return "Slowdown Risk"
    return "Recessionary"

# -----------------------------
# Layout (one screen)
# -----------------------------
st.subheader(f"Cycle Score: {score}  →  {regime_from_score(score)}")

st.markdown("## What would worry me (checklist)")

# Build signals from your existing series
yc_flattening = is_falling(yc, months=3)              # curve flattening
nfci_tightening = is_rising(nfci, months=3)           # NFCI rising = tightening
claims_rising = is_rising(claims_4w, months=3)         # layoffs
spreads_widening = is_rising(hy_oas, months=3) or is_rising(ig_oas, months=3)
inflation_reaccel = is_rising(core_cpi_yoy, months=3)  # core inflation picking up
equity_trend_down = last_value(spy_trend) < 0 if not spy_trend.empty else False
gdp_slipping = is_falling(gdp_yoy, months=6)

checks = [
    (badge(not yc_flattening), "Yield curve is not flattening", "Flattening/inversion risk rising (growth risk)"),
    (badge(not nfci_tightening), "Financial conditions not tightening fast", "Liquidity tightening (risk-off pressure)"),
    (badge(not claims_rising), "Jobless claims not rising", "Labor market cracking (recession risk)"),
    (badge(not spreads_widening), "Credit spreads not widening", "Credit stress building (often leads equities)"),
    (badge(not inflation_reaccel), "Core inflation not re-accelerating", "Sticky/re-accelerating inflation (Fed constraint)"),
    (badge(not equity_trend_down), "Equity trend still supportive", "Trend turning down (regime shift risk)"),
    (badge(not gdp_slipping), "GDP trend not deteriorating fast", "Growth decelerating (confirmation of slowdown)"),
]

for icon, ok_txt, warn_txt in checks:
    st.write(f"{icon} {ok_txt if icon=='✅' else warn_txt}")

st.caption("If you start seeing multiple ⚠️ at once, your regime is usually shifting toward slowdown/risk-off.")

# Top row – Where are we?
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### Yield Curve (10Y–2Y)")
    st.metric("Latest", fmt(last_value(yc), "%"), delta=fmt(yc_d3, "%"))
    help_text("10Y minus 2Y Treasury yield. Steepening (more positive) is usually pro-growth; inversion is a classic recession warning.")
    st.line_chart(yc.loc[chart_start:], height=300)

with c2:
    st.markdown("### Financial Conditions (NFCI)")
    st.metric("Latest", fmt(last_value(nfci)), delta=fmt(nf_d3))
    help_text("Chicago Fed NFCI. Higher = tighter credit/less liquidity (worse for risk assets). Lower/negative = easier conditions.")
    st.line_chart(nfci.loc[chart_start:], height=300)

with c3:
    st.markdown("### Policy Rate vs Inflation")
    latest_ff = last_value(fed_funds)
    latest_core = last_value(core_cpi_yoy)
    st.metric("Fed Funds", fmt(latest_ff, "%"), delta=fmt(ff_d3, "%"))
    st.metric("Core CPI YoY", fmt(latest_core, "%"), delta=fmt(core_d3, "%"))
    help_text("Fed Funds = policy rate. Core CPI YoY = underlying inflation. If rates > inflation, policy is restrictive; gap shrinking usually means less pressure ahead.")
    comb = pd.DataFrame({"Interest Rates": fed_funds, "Inflation": core_cpi_yoy}).loc[chart_start:]
    st.line_chart(comb, height=300)

# Middle – Is growth accelerating?
st.markdown("### Is growth accelerating?")
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("**Activity (Manufacturing + Broad)**")
    st.metric("Manufacturing", fmt(last_value(mfg_activity), "idx"), delta=fmt(mfg_d3, "idx"))
    st.metric("Broad Activity", fmt(last_value(broad_activity), "idx"), delta=fmt(srv_d3, "idx"))
    help_text("These are activity proxies (levels). Focus on the direction and your 3M/6M deltas; rising momentum suggests improving growth.")
    st.line_chart(pd.DataFrame({"Mfg": mfg_activity, "Broad": broad_activity}).loc[chart_start:], height=300)

with m2:
    st.markdown("**Jobless Claims (4-week avg)**")
    st.metric("ICSA 4W MA", fmt(last_value(claims_4w), "k"), delta=fmt(cl_d3, "k"))
    help_text("Weekly layoffs signal. Rising claims = labor market weakening (often early recession risk); falling = labor still solid.")
    st.line_chart(claims_4w.loc[chart_start:], height=300)

with m3:
    st.markdown("**Real GDP YoY (confirmation)**")
    st.metric("GDP YoY", fmt(last_value(gdp_yoy), "%"), delta=fmt(gdp_d3, "%"))
    help_text("Backward-looking confirmation. GDP tends to move after surveys/claims. Use it to confirm, not to lead.")
    st.line_chart(gdp_yoy.loc[chart_start:], height=300)

# Bottom – Risk signals
st.markdown("### Risk signals")
b1, b2, b3 = st.columns(3)

with b1:
    st.markdown("**Inflation (Headline vs Core)**")
    st.metric("CPI YoY", fmt(last_value(cpi_yoy), "%"), delta=fmt(cpi_d3, "%"))
    st.metric("Core CPI YoY", fmt(last_value(core_cpi_yoy), "%"), delta=fmt(core_d3, "%"))
    help_text("Disinflation supports future easing. Core is stickier and matters more for the Fed; re-acceleration is a risk signal.")
    st.line_chart(pd.DataFrame({"CPI_YoY": cpi_yoy, "CoreCPI_YoY": core_cpi_yoy}).loc[chart_start:], height=300)

with b2:
    st.markdown("**Credit Spreads (IG / HY OAS)**")
    st.metric("IG OAS", fmt(last_value(ig_oas), "%"), delta=fmt(ig_d3, "%"))
    st.metric("HY OAS", fmt(last_value(hy_oas), "%"), delta=fmt(hy_d3, "%"))
    help_text("Credit stress gauge. Widening spreads = risk rising/default fear; tightening = confidence. Credit often cracks before equities.")
    st.line_chart(pd.DataFrame({"IG_OAS": ig_oas, "HY_OAS": hy_oas}).loc[chart_start:], height=300)

with b3:
    st.markdown("**Equity Trend (SPY 50d vs 200d)**")
    st.metric("Trend (50-200)", fmt(last_value(spy_trend)), delta=fmt(tr_d3))
    help_text("Simple trend filter. Positive = risk-on regime; negative = risk-off. Useful for avoiding the worst drawdowns, not for perfect timing.")
    st.line_chart(spy_trend.loc[chart_start:], height=300)

# -----------------------------
# Summary table
# -----------------------------
st.markdown("### All indicators (direction-first)")
rows = []
for name, val, unit, d3, d6, hib in metrics:
    if hib is None:
        light = "⚪"  # context-only
    else:
        light = status_from_change(d3, d6, higher_is_better=hib)
    rows.append(
        {
            "Signal": name,
            "Status": light,
            "Latest": fmt(val, unit if unit else ""),
            "Δ 3M": fmt(d3, unit if unit in ["%", "idx"] else ""),
            "Δ 6M": fmt(d6, unit if unit in ["%", "idx"] else ""),
        }
    )

st.dataframe(pd.DataFrame(rows), width="stretch")
st.caption("Activity proxies are loaded from FRED (defaults: IPMAN, INDPRO). Use the advanced overrides if you want different series.")

