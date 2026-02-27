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
    "pmi_mfg": "ISM Manufacturing PMI",
    "pmi_srv": "ISM Non-Manufacturing PMI",
}

PMI_FALLBACK_IDS = {
    "pmi_mfg": ["NAPM"],                 # ISM Manufacturing PMI
    "pmi_srv": ["NAPMNOI", "NAPMNMI"],   # try these in order (some accounts/regions differ)
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
    return float(end - start)

def last_value(s: pd.Series) -> float:
    return float(s.dropna().iloc[-1]) if not s.dropna().empty else float("nan")

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

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Macro Dashboard", layout="wide")
st.title("Macro Dashboard (Auto-updating)")

start = (date.today() - relativedelta(years=DEFAULT_LOOKBACK_YEARS)).isoformat()
chart_start = (date.today() - relativedelta(years=CHART_YEARS)).isoformat()

with st.sidebar:
    st.header("Settings")
    st.write("Data: FRED + Yahoo Finance (SPY).")
    chart_years = st.slider("Chart window (years)", 1, 15, CHART_YEARS)
    st.caption("Tip: keep this to 3–7 years for readability.")
    chart_start = (date.today() - relativedelta(years=chart_years)).isoformat()
    st.subheader("Find PMI series IDs")
    q = st.text_input("Search FRED series", value="PMI")
    if q:
        res = fred_search_series(q, limit=10)
        if res.empty:
            st.write("No results.")
        else:
            st.dataframe(res, width="stretch")
            st.caption("Copy the 'id' you want into the overrides below.")

# --- Load series
try:
    # PMI (dynamic search)
    st.subheader("Series overrides")

    pmi_mfg_id_input = st.text_input("PMI Manufacturing series id", value="NAPM")
    pmi_srv_id_input = st.text_input("PMI Services series id", value="NAPMNOI")
    # PMI (explicit IDs; no search fallback)
    pmi_mfg_id = pmi_mfg_id_input.strip()
    pmi_srv_id = pmi_srv_id_input.strip()

    try:
        pmi_mfg = fred_series(pmi_mfg_id, start)
        pmi_srv = fred_series(pmi_srv_id, start)
    except Exception as e:
        st.warning(f"PMI unavailable: {e}")
        pmi_mfg = pd.Series(dtype=float)
        pmi_srv = pd.Series(dtype=float)

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
mfg_d3, mfg_d6 = deltas(pmi_mfg)
srv_d3, srv_d6 = deltas(pmi_srv)
gdp_d3, gdp_d6 = deltas(gdp_yoy)

metrics += [
    ("PMI (Mfg)", last_value(pmi_mfg), "idx", mfg_d3, mfg_d6, True),
    ("PMI (Services)", last_value(pmi_srv), "idx", srv_d3, srv_d6, True),
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

# Top row – Where are we?
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### Yield Curve (10Y–2Y)")
    st.metric("Latest", fmt(last_value(yc), "%"), delta=fmt(yc_d3, "%"))
    st.line_chart(yc.loc[chart_start:])

with c2:
    st.markdown("### Financial Conditions (NFCI)")
    st.metric("Latest", fmt(last_value(nfci)), delta=fmt(nf_d3))
    st.line_chart(nfci.loc[chart_start:])

with c3:
    st.markdown("### Policy Rate vs Inflation")
    latest_ff = last_value(fed_funds)
    latest_core = last_value(core_cpi_yoy)
    st.metric("Fed Funds", fmt(latest_ff, "%"), delta=fmt(ff_d3, "%"))
    st.metric("Core CPI YoY", fmt(latest_core, "%"), delta=fmt(core_d3, "%"))
    comb = pd.DataFrame(
        {"FedFunds": fed_funds, "CoreCPI_YoY": core_cpi_yoy}
    ).loc[chart_start:]
    st.line_chart(comb)

# Middle – Is growth accelerating?
st.markdown("### Is growth accelerating?")
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("**PMI Trend (Mfg + Services)**")
    st.metric("PMI (Mfg)", fmt(last_value(pmi_mfg), "idx"), delta=fmt(mfg_d3, "idx"))
    st.metric("PMI (Services)", fmt(last_value(pmi_srv), "idx"), delta=fmt(srv_d3, "idx"))
    st.line_chart(pd.DataFrame({"PMI_Mfg": pmi_mfg, "PMI_Services": pmi_srv}).loc[chart_start:])

with m2:
    st.markdown("**Jobless Claims (4-week avg)**")
    st.metric("ICSA 4W MA", fmt(last_value(claims_4w), "k"), delta=fmt(cl_d3, "k"))
    st.line_chart(claims_4w.loc[chart_start:])

with m3:
    st.markdown("**Real GDP YoY (confirmation)**")
    st.metric("GDP YoY", fmt(last_value(gdp_yoy), "%"), delta=fmt(gdp_d3, "%"))
    st.line_chart(gdp_yoy.loc[chart_start:])

# Bottom – Risk signals
st.markdown("### Risk signals")
b1, b2, b3 = st.columns(3)

with b1:
    st.markdown("**Inflation (Headline vs Core)**")
    st.metric("CPI YoY", fmt(last_value(cpi_yoy), "%"), delta=fmt(cpi_d3, "%"))
    st.metric("Core CPI YoY", fmt(last_value(core_cpi_yoy), "%"), delta=fmt(core_d3, "%"))
    st.line_chart(pd.DataFrame({"CPI_YoY": cpi_yoy, "CoreCPI_YoY": core_cpi_yoy}).loc[chart_start:])

with b2:
    st.markdown("**Credit Spreads (IG / HY OAS)**")
    st.metric("IG OAS", fmt(last_value(ig_oas), "%"), delta=fmt(ig_d3, "%"))
    st.metric("HY OAS", fmt(last_value(hy_oas), "%"), delta=fmt(hy_d3, "%"))
    st.line_chart(pd.DataFrame({"IG_OAS": ig_oas, "HY_OAS": hy_oas}).loc[chart_start:])

with b3:
    st.markdown("**Equity Trend (SPY 50d vs 200d)**")
    st.metric("Trend (50-200)", fmt(last_value(spy_trend)), delta=fmt(tr_d3))
    st.line_chart(spy_trend.loc[chart_start:])

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

st.caption(
    "PMIs are fetched via FRED search at runtime (top match). "
    "If you want specific PMI providers/series IDs, hardcode them once you choose the source."
)