# -*- coding: utf-8 -*-
"""엔진 로직 검증 (네트워크 불필요). 실행: python test_engine.py"""
import numpy as np
import pandas as pd

from backtest import Config, run_ticker, baseline_ticker, summarize


def make_df(inst, opens):
    idx = pd.bdate_range("2024-01-01", periods=len(opens))
    return pd.DataFrame(
        {"open": opens, "high": opens, "low": opens, "close": opens,
         "volume": [1e6] * len(opens), "turnover": [1e9] * len(opens),
         "inst": inst}, index=idx)


def test_entry_exit_timing():
    """기관 순매수 3일 연속(idx 0,1,2) → idx3 시가 매수 → idx8 시가 매도"""
    cfg = Config(streak=3, hold=5, cost_bps=0.0)
    inst = [1, 1, 1] + [-1] * 9
    opens = [100.0] * 3 + [100.0] + [0.0] * 8
    opens = [100, 100, 100, 100, 101, 102, 103, 104, 110, 120, 130, 140]
    df = make_df(inst, [float(x) for x in opens])
    tr = run_ticker(df, cfg, "TEST")
    assert len(tr) == 1, tr
    r = tr.iloc[0]
    assert r["entry_px"] == 100 and r["exit_px"] == 110, (r["entry_px"], r["exit_px"])
    assert abs(r["gross_ret"] - 0.10) < 1e-9
    print("✓ 진입/청산 시점 정확 (D+1 시가 매수, D+6 시가 매도)")


def test_streak_requires_consecutive():
    """중간에 순매도가 끼면 시그널 없음"""
    cfg = Config(streak=3, hold=5, cost_bps=0.0)
    df = make_df([1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1], [100.0] * 12)
    assert run_ticker(df, cfg, "T").empty
    print("✓ 연속 조건 검증 (비연속 시그널 배제)")


def test_cost_applied():
    cfg = Config(streak=3, hold=5, cost_bps=33.0)
    df = make_df([1] * 12, [100.0] * 12)
    tr = run_ticker(df, cfg, "T")
    assert abs(tr["gross_ret"].iloc[0]) < 1e-12
    assert abs(tr["net_ret"].iloc[0] + 0.0033) < 1e-12
    print("✓ 거래비용 반영")


def test_no_lookahead():
    """시그널 당일 종가로 매수하지 않는지 — 시그널일 이후 가격만 사용"""
    cfg = Config(streak=3, hold=5, cost_bps=0.0)
    opens = [10, 10, 10, 50, 50, 50, 50, 50, 99, 10, 10, 10]
    df = make_df([1, 1, 1] + [0] * 9, [float(x) for x in opens])
    tr = run_ticker(df, cfg, "T")
    assert tr.iloc[0]["entry_px"] == 50  # 시그널 다음날(idx3) 시가
    assert tr.iloc[0]["exit_px"] == 99   # idx8 시가
    print("✓ 미래참조(look-ahead) 없음")


def test_filters():
    cfg = Config(streak=3, hold=5, cost_bps=0.0, min_inst_amount=1e9)
    df = make_df([1e6] * 12, [100.0] * 12)   # 합계 3e6 < 1e9 → 필터됨
    assert run_ticker(df, cfg, "T").empty
    print("✓ 순매수 금액 필터 동작")


def test_summary_stats():
    s = pd.Series([0.10, -0.05, 0.02, -0.01, 0.03])
    out = summarize(s)
    assert out["거래수"] == 5 and out["승률(%)"] == 60.0
    assert abs(out["평균수익률(%)"] - 1.8) < 1e-6
    print("✓ 통계 집계 정확")


def test_baseline():
    cfg = Config(hold=5, cost_bps=0.0)
    df = make_df([0] * 12, [float(x) for x in range(100, 112)])
    b = baseline_ticker(df, cfg)
    assert len(b) == 6 and all(x > 0 for x in b)
    print("✓ 비교군(baseline) 산출")


def test_random_walk_has_no_edge():
    """랜덤 데이터에서는 엣지가 나오지 않아야 함 (과최적화 방지 sanity check)"""
    rng = np.random.default_rng(42)
    rets, base = [], []
    cfg = Config(streak=3, hold=5, cost_bps=0.0)
    for _ in range(300):
        px = 1000 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
        inst = rng.normal(0, 1, 400)
        df = make_df(inst, px)
        tr = run_ticker(df, cfg, "R")
        if not tr.empty:
            rets.extend(tr["net_ret"].tolist())
        base.extend(baseline_ticker(df, cfg).tolist())
    edge = np.mean(rets) - np.mean(base)
    assert abs(edge) < 0.004, f"랜덤 데이터에 엣지 발생: {edge:.4%}"
    print(f"✓ 랜덤워크 엣지 ≈ 0 ({edge:+.4%}, 거래 {len(rets):,}건)")


if __name__ == "__main__":
    for fn in [test_entry_exit_timing, test_streak_requires_consecutive, test_cost_applied,
               test_no_lookahead, test_filters, test_summary_stats, test_baseline,
               test_random_walk_has_no_edge]:
        fn()
    print("\n전체 통과")
