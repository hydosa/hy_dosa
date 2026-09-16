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
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# 기관 순매수 컬럼 후보 (pykrx 버전에 따라 명칭이 다름)
INST_COLS = ["기관합계", "기관", "기관계"]


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
    min_inst_amount: float = 0.0   # 연속 기간 기관 순매수 합계 최소 금액(원). 0=제한없음
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
    path = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end}.pkl")
    if use_cache and os.path.exists(path):
        try:
            return pd.read_pickle(path)
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
    if inst_col is None:
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
    df = df.dropna(subset=["open", "close", "inst"])
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

    pos = (d["inst"] > 0).astype(int)
    streak_ok = pos.rolling(cfg.streak).sum() == cfg.streak       # D일 기준 충족
    inst_sum = d["inst"].rolling(cfg.streak).sum()

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
            "prev_turnover": d["turnover"].values[sig_idx],
            "gross_ret": gross,
            "net_ret": net,
        }
    )

    if cfg.min_inst_amount:
        out = out[out["inst_sum"] >= cfg.min_inst_amount]
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
    A("=" * 78)
    A(f"기관 {cfg.streak}일 연속 순매수 → 익일 시가 매수 → {cfg.hold}거래일 보유 백테스트")
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

    A("[주의]")
    A("  · 현재 상장 종목만 대상 → 상장폐지 종목 제외로 인한 생존 편향(수익률 과대) 존재")
    A("  · 시가 체결 가정. 실제로는 시초가 슬리피지가 추가로 발생")
    A("  · 동시 진입 종목 수 제한 없음(무한 자금 가정). 실제 운용 시 종목 선별 필요")
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
    p.add_argument("--min-turnover", type=float, default=0.0, help="시그널일 거래대금 최소(원)")
    p.add_argument("--max-tickers", type=int, default=0, help="0=전 종목")
    p.add_argument("--market", default="KOSPI,KOSDAQ")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out", default="result")
    args = p.parse_args()

    cfg = Config(
        years=args.years,
        streak=args.streak,
        hold=args.hold,
        markets=tuple(m.strip() for m in args.market.split(",") if m.strip()),
        cost_bps=args.cost_bps,
        min_inst_amount=args.min_inst_amount,
        min_turnover=args.min_turnover,
        max_tickers=args.max_tickers,
    )

    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=int(365.25 * cfg.years) + 20)
    start, end = start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")

    print(f"▶ 유니버스 조회 중… ({start} ~ {end})")
    uni = get_universe(cfg, end)
    print(f"  대상 종목: {len(uni):,}개")

    all_trades, all_base = [], []
    t0 = time.time()
    for i, row in uni.iterrows():
        df = fetch_ticker(row["ticker"], start, end, use_cache=not args.no_cache)
        if df is None or df.empty:
            continue
        tr = run_ticker(df, cfg, row["ticker"], row["name"], row["market"])
        if not tr.empty:
            all_trades.append(tr)
        all_base.append(baseline_ticker(df, cfg))
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(uni) - i - 1)
            print(f"  {i+1:>5}/{len(uni)}  경과 {el/60:.1f}분  남은시간 약 {eta/60:.1f}분")

    if not all_trades:
        sys.exit("거래 시그널이 없습니다. 조건을 완화해 보세요.")

    trades = pd.concat(all_trades, ignore_index=True)
    baseline = np.concatenate([b for b in all_base if len(b)]) if all_base else np.array([])

    txt = report(trades, baseline, cfg)
    print("\n" + txt)

    trades.to_csv(f"{args.out}_trades.csv", index=False, encoding="utf-8-sig")
    with open(f"{args.out}_summary.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"\n저장: {args.out}_trades.csv / {args.out}_summary.txt")


if __name__ == "__main__":
    main()
