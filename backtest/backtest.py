#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기관 N일 연속 순매수 종목 → 익일 시가 매수 → M거래일 보유 후 시가 매도 백테스트

기본 설정: 최근 3년 / 3일 연속 순매수 / 5거래일 보유 / KOSPI+KOSDAQ

실행:
    pip install -r requirements.txt
    python backtest.py                      # 기본 (최근 3년, 전 종목)
    python backtest.py --years 3 --streak 3 --hold 5
    python backtest.py --max-tickers 200    # 빠른 테스트용 샘플

데이터 출처: KRX (pykrx)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
ENV_PATH = os.path.join(BASE_DIR, ".env")


def load_env(path: str = ENV_PATH) -> None:
    """
    .env 파일에서 KRX_ID / KRX_PW 등을 읽어 환경변수로 올린다.

    pykrx 1.2.x는 모듈을 import 하는 시점에 os.getenv("KRX_ID")를 읽으므로,
    반드시 pykrx import 전에 호출해야 한다. (이 파일은 pykrx를 지연 import 한다)
    이미 설정된 환경변수는 덮어쓰지 않는다.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and not os.environ.get(k):
                    os.environ[k] = v
    except Exception as e:
        print(f"[경고] .env 읽기 실패: {e}", file=sys.stderr)


load_env()

# 투자자별 순매수 컬럼 후보 (pykrx 버전에 따라 명칭이 다름)
INST_COLS = ["기관합계", "기관", "기관계"]
FRGN_COLS = ["외국인합계", "외국인", "외국인계"]

# 캐시 포맷 버전. 컬럼 구성이 바뀌면 올려서 기존 캐시를 무효화한다.
CACHE_VER = "v2"


# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
@dataclass
class Config:
    years: int = 3
    streak: int = 3            # 기관 연속 순매수 일수
    hold: int = 5              # 보유 거래일
    markets: tuple = ("KOSPI", "KOSDAQ")
    cost_bps: float = 33.0     # 왕복 비용(bp). 수수료 0.015%x2 + 증권거래세 0.15% + 슬리피지
    require_foreign: bool = False  # True면 같은 날들에 외국인도 순매수여야 함
    min_inst_amount: float = 0.0   # 연속 기간 기관 순매수 합계 최소 금액(원). 0=제한없음
    min_frgn_amount: float = 0.0   # 연속 기간 외국인 순매수 합계 최소 금액(원). 0=제한없음
    min_turnover: float = 0.0      # 매수 직전일 거래대금 최소(원). 0=제한없음
    max_tickers: int = 0           # 0=전체
    exclude_pref: bool = True      # 우선주 제외
    exclude_spac: bool = True      # 스팩 제외


# ----------------------------------------------------------------------------
# 데이터 수집
# ----------------------------------------------------------------------------
def _pykrx():
    try:
        from pykrx import stock
    except ImportError:
        sys.exit("pykrx 미설치. `pip install -r requirements.txt` 실행 후 다시 시도하세요.")
    return stock


def check_krx() -> int:
    """KRX 계정 설정 상태와 데이터 수신 여부를 점검한다."""
    import importlib.metadata as md

    try:
        ver = md.version("pykrx")
    except Exception:
        ver = "?"
    print(f"pykrx 버전      : {ver}")
    print(f".env 파일       : {ENV_PATH} {'(있음)' if os.path.exists(ENV_PATH) else '(없음)'}")

    kid, kpw = os.environ.get("KRX_ID"), os.environ.get("KRX_PW")
    if kid:
        print(f"KRX_ID          : {kid[:2]}{'*' * max(len(kid) - 2, 0)}  (설정됨)")
    else:
        print("KRX_ID          : 없음")
    print(f"KRX_PW          : {'설정됨 (' + '*' * len(kpw) + ')' if kpw else '없음'}")

    if ver.startswith("1.0") and not kid:
        print("\n→ pykrx 1.0.x는 로그인이 필요 없습니다. 이대로 실행하시면 됩니다.")
    elif ver.startswith("1.2") and not kid:
        print("\n→ pykrx 1.2.x는 KRX 로그인이 필요합니다. .env에 KRX_ID/KRX_PW를 넣으세요.")

    print("\n데이터 수신 테스트 (삼성전자 005930, 최근 10영업일)…")
    end = datetime.today()
    start = end - timedelta(days=20)
    df = fetch_ticker("005930", start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), use_cache=False)
    if df is None or df.empty:
        print("  ✗ 실패. 데이터를 받지 못했습니다.")
        print("    · 계정/비밀번호 확인")
        print("    · KRX 사이트에서 직접 로그인되는지 확인 (비밀번호 만료 여부)")
        print("    · 회사 방화벽/프록시가 data.krx.co.kr을 막고 있는지 확인")
        return 1
    print(f"  ✓ 성공. {len(df)}일치 수신")
    print(df.tail(3)[["open", "close", "inst", "foreign"]].to_string())
    return 0


def _call(stock, names, *args, **kwargs):
    """pykrx 버전별 함수명 차이 흡수."""
    for n in names:
        fn = getattr(stock, n, None)
        if fn is not None:
            return fn(*args, **kwargs)
    raise AttributeError(f"pykrx에 {names} 함수가 없습니다. 버전을 확인하세요.")


def get_universe(cfg: Config, asof: str) -> pd.DataFrame:
    """종목코드 / 종목명 / 시장 목록."""
    stock = _pykrx()
    rows = []
    for mkt in cfg.markets:
        for t in stock.get_market_ticker_list(asof, market=mkt):
            rows.append({"ticker": t, "market": mkt})
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    names = []
    for t in df["ticker"]:
        try:
            names.append(stock.get_market_ticker_name(t))
        except Exception:
            names.append("")
    df["name"] = names

    if cfg.exclude_pref:
        # 보통주는 종목코드 끝자리가 0
        df = df[df["ticker"].str.endswith("0")]
    if cfg.exclude_spac:
        df = df[~df["name"].str.contains("스팩", na=False)]
    df = df[~df["name"].str.contains("리츠|우선주", na=False)]
    df = df.reset_index(drop=True)
    if cfg.max_tickers:
        df = df.head(cfg.max_tickers)
    return df


def fetch_ticker(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame | None:
    """한 종목의 OHLCV + 투자자별 순매수(기관) 일별 데이터."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}_{CACHE_VER}.pkl")
    if use_cache and os.path.exists(path):
        try:
            cached = pd.read_pickle(path)
            if "foreign" in cached.columns:
                return cached
        except Exception:
            pass

    stock = _pykrx()
    try:
        ohlcv = _call(stock, ["get_market_ohlcv_by_date", "get_market_ohlcv"], start, end, ticker)
        time.sleep(0.15)
        inv = stock.get_market_trading_value_by_date(start, end, ticker, on="순매수")
        time.sleep(0.15)
    except Exception as e:
        print(f"  [skip] {ticker}: {e}", file=sys.stderr)
        return None

    if ohlcv is None or inv is None or ohlcv.empty or inv.empty:
        return None

    inst_col = next((c for c in INST_COLS if c in inv.columns), None)
    frgn_col = next((c for c in FRGN_COLS if c in inv.columns), None)
    if inst_col is None or frgn_col is None:
        print(f"  [skip] {ticker}: 투자자별 컬럼 없음 ({list(inv.columns)})", file=sys.stderr)
        return None

    df = pd.DataFrame(
        {
            "open": ohlcv["시가"].astype(float),
            "high": ohlcv["고가"].astype(float),
            "low": ohlcv["저가"].astype(float),
            "close": ohlcv["종가"].astype(float),
            "volume": ohlcv["거래량"].astype(float),
            "turnover": ohlcv["거래대금"].astype(float) if "거래대금" in ohlcv.columns else np.nan,
        }
    )
    df["inst"] = inv[inst_col].astype(float)
    df["foreign"] = inv[frgn_col].astype(float)
    df = df.dropna(subset=["open", "close", "inst", "foreign"])
    df = df[df["volume"] > 0]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if use_cache:
        try:
            df.to_pickle(path)
        except Exception:
            pass
    return df


# ----------------------------------------------------------------------------
# 백테스트 엔진 (네트워크 불필요 — 단위 테스트 가능)
# ----------------------------------------------------------------------------
def run_ticker(df: pd.DataFrame, cfg: Config, ticker: str = "", name: str = "",
               market: str = "") -> pd.DataFrame:
    """
    시그널: D-(streak-1) ~ D일까지 기관 순매수 > 0 연속
    진입:   D+1 시가
    청산:   D+1+hold 시가
    """
    if df is None or len(df) < cfg.streak + cfg.hold + 2:
        return pd.DataFrame()

    d = df.reset_index().rename(columns={df.index.name or "index": "date"})
    d.columns = ["date"] + list(d.columns[1:])

    if "foreign" not in d.columns:
        d["foreign"] = np.nan

    inst_pos = (d["inst"] > 0).astype(int)
    if cfg.require_foreign:
        # 연속 기간의 '매일' 기관·외국인이 동시에 순매수여야 함
        both = ((d["inst"] > 0) & (d["foreign"] > 0)).astype(int)
        streak_ok = both.rolling(cfg.streak).sum() == cfg.streak
    else:
        streak_ok = inst_pos.rolling(cfg.streak).sum() == cfg.streak

    inst_sum = d["inst"].rolling(cfg.streak).sum()
    frgn_sum = d["foreign"].rolling(cfg.streak).sum()

    n = len(d)
    sig_idx = np.where(streak_ok.fillna(False).values)[0]
    # 진입 i+1, 청산 i+1+hold 이 존재해야 함
    sig_idx = sig_idx[(sig_idx + 1 + cfg.hold) < n]
    if len(sig_idx) == 0:
        return pd.DataFrame()

    entry_i = sig_idx + 1
    exit_i = entry_i + cfg.hold

    entry_px = d["open"].values[entry_i]
    exit_px = d["open"].values[exit_i]
    ok = (entry_px > 0) & (exit_px > 0)
    sig_idx, entry_i, exit_i = sig_idx[ok], entry_i[ok], exit_i[ok]
    entry_px, exit_px = entry_px[ok], exit_px[ok]

    gross = exit_px / entry_px - 1.0
    net = gross - cfg.cost_bps / 10000.0

    out = pd.DataFrame(
        {
            "ticker": ticker,
            "name": name,
            "market": market,
            "signal_date": d["date"].values[sig_idx],
            "entry_date": d["date"].values[entry_i],
            "exit_date": d["date"].values[exit_i],
            "entry_px": entry_px,
            "exit_px": exit_px,
            "inst_sum": inst_sum.values[sig_idx],
            "frgn_sum": frgn_sum.values[sig_idx],
            "prev_turnover": d["turnover"].values[sig_idx],
            "gross_ret": gross,
            "net_ret": net,
        }
    )

    if cfg.min_inst_amount:
        out = out[out["inst_sum"] >= cfg.min_inst_amount]
    if cfg.min_frgn_amount:
        out = out[out["frgn_sum"] >= cfg.min_frgn_amount]
    if cfg.min_turnover:
        out = out[out["prev_turnover"].fillna(0) >= cfg.min_turnover]
    return out


def baseline_ticker(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """비교군: 같은 종목/기간의 '아무 날' 진입했을 때의 hold일 수익률 분포."""
    if df is None or len(df) < cfg.hold + 2:
        return np.array([])
    o = df["open"].values
    n = len(o)
    i = np.arange(0, n - cfg.hold - 1)
    e = o[i + 1]
    x = o[i + 1 + cfg.hold]
    m = (e > 0) & (x > 0)
    return (x[m] / e[m] - 1.0) - cfg.cost_bps / 10000.0


# ----------------------------------------------------------------------------
# 집계 / 리포트
# ----------------------------------------------------------------------------
def summarize(r: pd.Series) -> dict:
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return {}
    wins = r[r > 0]
    losses = r[r <= 0]
    return {
        "거래수": len(r),
        "승률(%)": round(100 * len(wins) / len(r), 2),
        "평균수익률(%)": round(100 * r.mean(), 3),
        "중앙값(%)": round(100 * r.median(), 3),
        "표준편차(%)": round(100 * r.std(), 3),
        "평균이익(%)": round(100 * wins.mean(), 3) if len(wins) else 0.0,
        "평균손실(%)": round(100 * losses.mean(), 3) if len(losses) else 0.0,
        "손익비": round(abs(wins.mean() / losses.mean()), 2) if len(losses) and losses.mean() != 0 else np.nan,
        "Profit Factor": round(wins.sum() / abs(losses.sum()), 2) if len(losses) and losses.sum() != 0 else np.nan,
        "최대이익(%)": round(100 * r.max(), 2),
        "최대손실(%)": round(100 * r.min(), 2),
    }


def report(trades: pd.DataFrame, baseline: np.ndarray, cfg: Config) -> str:
    L = []
    A = L.append
    subj = "기관+외국인 동시" if cfg.require_foreign else "기관"
    A("=" * 78)
    A(f"{subj} {cfg.streak}일 연속 순매수 → 익일 시가 매수 → {cfg.hold}거래일 보유 백테스트")
    A("=" * 78)
    A(f"기간            : {trades['entry_date'].min():%Y-%m-%d} ~ {trades['exit_date'].max():%Y-%m-%d}")
    A(f"대상 시장       : {', '.join(cfg.markets)}")
    A(f"종목 수         : {trades['ticker'].nunique():,}개")
    A(f"거래 비용(왕복) : {cfg.cost_bps:.1f}bp ({cfg.cost_bps/100:.2f}%)")
    A("")

    A("[전체 결과 — 비용 차감 후]")
    for k, v in summarize(trades["net_ret"]).items():
        A(f"  {k:<16}: {v}")
    A("")
    A("[전체 결과 — 비용 차감 전]")
    for k, v in summarize(trades["gross_ret"]).items():
        A(f"  {k:<16}: {v}")
    A("")

    if len(baseline):
        b = summarize(pd.Series(baseline))
        A("[비교군 — 같은 종목/기간에 아무 날이나 매수했을 때 (비용 차감 후)]")
        A(f"  거래수          : {b['거래수']:,}")
        A(f"  승률(%)         : {b['승률(%)']}")
        A(f"  평균수익률(%)   : {b['평균수익률(%)']}")
        edge_w = summarize(trades['net_ret'])['승률(%)'] - b['승률(%)']
        edge_r = summarize(trades['net_ret'])['평균수익률(%)'] - b['평균수익률(%)']
        A("")
        A(f"  ▶ 초과 승률     : {edge_w:+.2f}%p")
        A(f"  ▶ 초과 수익률   : {edge_r:+.3f}%p   ← 이 값이 전략의 실제 엣지")
        A("")

    A("[연도별]")
    t = trades.copy()
    t["연도"] = pd.to_datetime(t["entry_date"]).dt.year
    yr = t.groupby("연도")["net_ret"].agg(
        거래수="count",
        승률=lambda s: round(100 * (s > 0).mean(), 2),
        평균수익률=lambda s: round(100 * s.mean(), 3),
    )
    A(yr.to_string())
    A("")

    A("[시장별]")
    mk = t.groupby("market")["net_ret"].agg(
        거래수="count",
        승률=lambda s: round(100 * (s > 0).mean(), 2),
        평균수익률=lambda s: round(100 * s.mean(), 3),
    )
    A(mk.to_string())
    A("")

    A("[기관 순매수 금액 5분위별] (1=소액 … 5=대액)")
    try:
        t["분위"] = pd.qcut(t["inst_sum"], 5, labels=[1, 2, 3, 4, 5])
        q = t.groupby("분위", observed=True)["net_ret"].agg(
            거래수="count",
            승률=lambda s: round(100 * (s > 0).mean(), 2),
            평균수익률=lambda s: round(100 * s.mean(), 3),
        )
        A(q.to_string())
    except Exception:
        A("  (샘플 부족)")
    A("")

    if t["frgn_sum"].notna().any():
        A("[외국인 순매수 금액 5분위별] (1=소액/순매도 … 5=대액)")
        try:
            t["외인분위"] = pd.qcut(t["frgn_sum"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
            q2 = t.groupby("외인분위", observed=True)["net_ret"].agg(
                거래수="count",
                승률=lambda s: round(100 * (s > 0).mean(), 2),
                평균수익률=lambda s: round(100 * s.mean(), 3),
            )
            A(q2.to_string())
        except Exception:
            A("  (샘플 부족)")
        A("")

    A("[주의]")
    A("  · 현재 상장 종목만 대상 → 상장폐지 종목 제외로 인한 생존 편향(수익률 과대) 존재")
    A("  · 시가 체결 가정. 실제로는 시초가 슬리피지가 추가로 발생")
    A("  · 동시 진입 종목 수 제한 없음(무한 자금 가정). 실제 운용 시 종목 선별 필요")
    A("=" * 78)
    return "\n".join(L)



def _f(x):
    """numpy 스칼라 → JSON 직렬화 가능한 float."""
    try:
        v = float(x)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)
    except Exception:
        return None


def _group_stats(t: pd.DataFrame, key) -> list:
    out = []
    for k, g in t.groupby(key, observed=True):
        r = g["net_ret"]
        out.append({
            "key": str(k),
            "n": int(len(r)),
            "win_rate": _f(100 * (r > 0).mean()),
            "mean_ret": _f(100 * r.mean()),
        })
    return out


def build_dashboard_data(res: dict, baseline: np.ndarray, cfg: Config, uni_n: int) -> dict:
    """대시보드(dashboard.html)가 읽는 단일 JSON 구조를 만든다."""
    variants = []
    for label, t in res.items():
        if t is None or t.empty:
            continue
        t = t.copy()
        t["entry_date"] = pd.to_datetime(t["entry_date"])
        r = t["net_ret"]
        wins, losses = r[r > 0], r[r <= 0]

        # 수익률 분포 (−15% ~ +15%, 1%p 단위)
        edges = np.arange(-0.15, 0.1501, 0.01)
        counts, _ = np.histogram(r.clip(-0.1499, 0.1499), bins=edges)
        hist = [{"lo": _f(100 * edges[i]), "hi": _f(100 * edges[i + 1]), "n": int(c)}
                for i, c in enumerate(counts)]

        # 누적 수익 곡선: 진입일별 평균 수익률의 누적합 (매일 균등분산 가정)
        daily = t.groupby("entry_date")["net_ret"].mean().sort_index()
        equity = [{"d": d.strftime("%Y-%m-%d"), "v": _f(100 * v)}
                  for d, v in daily.cumsum().items()]

        def quint(col):
            if col not in t.columns or t[col].isna().all():
                return []
            try:
                q = pd.qcut(t[col], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
                return _group_stats(t.assign(_q=q), "_q")
            except Exception:
                return []

        cols = ["ticker", "name", "market", "entry_date", "exit_date", "net_ret", "inst_sum", "frgn_sum"]
        cols = [c for c in cols if c in t.columns]

        def recs(d):
            out = []
            for _, x in d[cols].iterrows():
                out.append({
                    "ticker": str(x.get("ticker", "")),
                    "name": str(x.get("name", "")),
                    "market": str(x.get("market", "")),
                    "entry": pd.to_datetime(x["entry_date"]).strftime("%Y-%m-%d"),
                    "exit": pd.to_datetime(x["exit_date"]).strftime("%Y-%m-%d"),
                    "ret": _f(100 * x["net_ret"]),
                    "inst": _f(x.get("inst_sum")),
                    "frgn": _f(x.get("frgn_sum")),
                })
            return out

        variants.append({
            "label": label,
            "n": int(len(r)),
            "win_rate": _f(100 * (r > 0).mean()),
            "mean_ret": _f(100 * r.mean()),
            "median_ret": _f(100 * r.median()),
            "std": _f(100 * r.std()),
            "avg_win": _f(100 * wins.mean()) if len(wins) else None,
            "avg_loss": _f(100 * losses.mean()) if len(losses) else None,
            "payoff": _f(abs(wins.mean() / losses.mean())) if len(losses) and losses.mean() else None,
            "pf": _f(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else None,
            "by_year": _group_stats(t.assign(_y=t["entry_date"].dt.year), "_y"),
            "by_market": _group_stats(t, "market"),
            "by_inst_q": quint("inst_sum"),
            "by_frgn_q": quint("frgn_sum"),
            "hist": hist,
            "equity": equity,
            "top": recs(t.nlargest(15, "net_ret")),
            "bottom": recs(t.nsmallest(15, "net_ret")),
        })

    b = None
    if len(baseline):
        bs = pd.Series(baseline)
        b = {"n": int(len(bs)), "win_rate": _f(100 * (bs > 0).mean()), "mean_ret": _f(100 * bs.mean())}

    all_dates = [v["equity"][0]["d"] for v in variants if v["equity"]]
    all_end = [v["equity"][-1]["d"] for v in variants if v["equity"]]
    return {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "start": min(all_dates) if all_dates else "",
            "end": max(all_end) if all_end else "",
            "streak": cfg.streak,
            "hold": cfg.hold,
            "cost_bps": cfg.cost_bps,
            "markets": list(cfg.markets),
            "universe": int(uni_n),
        },
        "baseline": b,
        "variants": variants,
    }


def compare(res: dict, baseline: np.ndarray, cfg: Config) -> str:
    """조건별 성과 비교표. res = {라벨: trades DataFrame}"""
    L = []
    A = L.append
    A("=" * 78)
    A(f"조건별 비교  ({cfg.streak}일 연속 순매수 → 익일 시가 매수 → {cfg.hold}거래일 보유)")
    A("=" * 78)

    base_win = base_ret = None
    if len(baseline):
        b = summarize(pd.Series(baseline))
        base_win, base_ret = b["승률(%)"], b["평균수익률(%)"]

    rows = []
    for label, tr in res.items():
        if tr is None or tr.empty:
            rows.append({"조건": label, "거래수": 0})
            continue
        s = summarize(tr["net_ret"])
        rows.append({
            "조건": label,
            "거래수": s["거래수"],
            "승률(%)": s["승률(%)"],
            "평균수익률(%)": s["평균수익률(%)"],
            "중앙값(%)": s["중앙값(%)"],
            "손익비": s["손익비"],
            "PF": s["Profit Factor"],
            "초과승률(%p)": round(s["승률(%)"] - base_win, 2) if base_win is not None else np.nan,
            "초과수익(%p)": round(s["평균수익률(%)"] - base_ret, 3) if base_ret is not None else np.nan,
        })
    if base_win is not None:
        rows.append({
            "조건": "(비교군) 임의 진입",
            "거래수": len(baseline),
            "승률(%)": base_win,
            "평균수익률(%)": base_ret,
            "초과승률(%p)": 0.0,
            "초과수익(%p)": 0.0,
        })

    A(pd.DataFrame(rows).to_string(index=False))
    A("")
    A("  ※ 모두 거래비용 차감 후. '초과수익'이 전략의 실제 엣지입니다.")
    A("  ※ 외국인 조건을 걸면 거래수가 크게 줄어듭니다. 거래수가 100건 미만이면")
    A("     승률 숫자는 우연일 가능성이 높으니 신뢰하지 마십시오.")
    A("=" * 78)
    return "\n".join(L)


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="기관 연속 순매수 백테스트")
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--streak", type=int, default=3)
    p.add_argument("--hold", type=int, default=5)
    p.add_argument("--cost-bps", type=float, default=33.0)
    p.add_argument("--min-inst-amount", type=float, default=0.0, help="기관 순매수 합계 최소 금액(원)")
    p.add_argument("--min-frgn-amount", type=float, default=0.0, help="외국인 순매수 합계 최소 금액(원)")
    p.add_argument(
        "--mode",
        default="both",
        choices=["inst", "inst+frgn", "both"],
        help="inst=기관만, inst+frgn=기관+외국인 동시, both=둘 다 돌려서 비교(기본)",
    )
    p.add_argument("--min-turnover", type=float, default=0.0, help="시그널일 거래대금 최소(원)")
    p.add_argument("--max-tickers", type=int, default=0, help="0=전 종목")
    p.add_argument("--market", default="KOSPI,KOSDAQ")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out", default="result")
    p.add_argument("--check-krx", action="store_true",
                   help="KRX 계정 설정과 데이터 수신만 점검하고 종료")
    args = p.parse_args()

    if args.check_krx:
        sys.exit(check_krx())

    cfg = Config(
        years=args.years,
        streak=args.streak,
        hold=args.hold,
        markets=tuple(m.strip() for m in args.market.split(",") if m.strip()),
        cost_bps=args.cost_bps,
        min_inst_amount=args.min_inst_amount,
        min_frgn_amount=args.min_frgn_amount,
        min_turnover=args.min_turnover,
        max_tickers=args.max_tickers,
    )

    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=int(365.25 * cfg.years) + 20)
    start, end = start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")

    print(f"▶ 유니버스 조회 중… ({start} ~ {end})")
    uni = get_universe(cfg, end)
    print(f"  대상 종목: {len(uni):,}개")

    # 돌릴 조건 구성 (데이터는 한 번만 받고 조건만 달리 적용)
    variants = []
    if args.mode in ("inst", "both"):
        variants.append(("기관 단독", replace(cfg, require_foreign=False), "inst"))
    if args.mode in ("inst+frgn", "both"):
        variants.append(("기관+외국인 동시", replace(cfg, require_foreign=True), "inst_frgn"))

    buckets = {label: [] for label, _, _ in variants}
    all_base = []
    t0 = time.time()
    for i, row in uni.iterrows():
        df = fetch_ticker(row["ticker"], start, end, use_cache=not args.no_cache)
        if df is None or df.empty:
            continue
        for label, vcfg, _ in variants:
            tr = run_ticker(df, vcfg, row["ticker"], row["name"], row["market"])
            if not tr.empty:
                buckets[label].append(tr)
        all_base.append(baseline_ticker(df, cfg))
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(uni) - i - 1)
            print(f"  {i+1:>5}/{len(uni)}  경과 {el/60:.1f}분  남은시간 약 {eta/60:.1f}분")

    baseline = np.concatenate([b for b in all_base if len(b)]) if all_base else np.array([])

    res, parts = {}, []
    for label, vcfg, slug in variants:
        frames = buckets[label]
        trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        res[label] = trades
        if trades.empty:
            print(f"\n[{label}] 거래 시그널이 없습니다. 조건을 완화해 보세요.")
            continue
        txt = report(trades, baseline, vcfg)
        print("\n" + txt)
        parts.append(txt)
        trades.to_csv(f"{args.out}_{slug}_trades.csv", index=False, encoding="utf-8-sig")

    if not any(not t.empty for t in res.values()):
        sys.exit("전 조건에서 거래 시그널이 없습니다.")

    cmp_txt = compare(res, baseline, cfg)
    print("\n" + cmp_txt)
    parts.append(cmp_txt)

    with open(f"{args.out}_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))

    data = build_dashboard_data(res, baseline, cfg, len(uni))
    with open(f"{args.out}_dashboard.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"\n저장: {args.out}_*_trades.csv / {args.out}_summary.txt")
    print(f"      {args.out}_dashboard.json  ← dashboard.html 에 올리면 화면으로 보입니다")


if __name__ == "__main__":
    main()
