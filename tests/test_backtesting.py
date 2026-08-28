import numpy as np
import pandas as pd
import pytest

from src.backtesting import BacktestConfig, Backtester


def market_frame(closes: list[float]) -> pd.DataFrame:
    values = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"high": values, "low": values, "close": values},
        index=pd.date_range("2024-01-01", periods=len(values), freq="1h", tz="UTC"),
    )


def test_config_rejects_invalid_financial_values() -> None:
    with pytest.raises(ValueError, match="initial_capital"):
        BacktestConfig(initial_capital=0)
    with pytest.raises(ValueError, match="position_size_pct"):
        BacktestConfig(position_size_pct=1.1)


def test_inputs_must_be_aligned_and_binary() -> None:
    data = market_frame([100, 101])
    with pytest.raises(ValueError, match="mesmo tamanho"):
        Backtester().run(data, np.array([1]))
    with pytest.raises(ValueError, match="binários"):
        Backtester().run(data, np.array([1, 2]))


def test_signal_exit_and_round_trip_fees_are_accounted() -> None:
    config = BacktestConfig(
        initial_capital=1_000,
        position_size_pct=1,
        fee_pct=0.001,
        slippage_pct=0,
        stop_loss_pct=1,
        take_profit_pct=1,
        min_confidence=0,
    )
    backtester = Backtester(config)
    result = backtester.run(market_frame([100, 100]), np.array([1, 0]))

    trade = backtester.trades[0]
    assert trade.exit_reason == "signal"
    assert trade.pnl == pytest.approx(-2.0)
    assert result["capital"]["final"] == pytest.approx(998.0)


def test_short_uses_probability_of_predicted_class() -> None:
    config = BacktestConfig(
        allow_short=True,
        min_confidence=0.8,
        fee_pct=0,
        slippage_pct=0,
        stop_loss_pct=1,
        take_profit_pct=1,
    )
    backtester = Backtester(config)
    backtester.run(
        market_frame([100, 99]),
        np.array([0, 0]),
        probabilities=np.array([0.1, 0.1]),
    )
    assert backtester.trades[0].direction == "short"
    assert backtester.trades[0].pnl > 0


def test_stop_loss_has_priority_when_triggered() -> None:
    data = market_frame([100, 100])
    data.iloc[1, data.columns.get_loc("low")] = 97
    config = BacktestConfig(
        position_size_pct=1,
        fee_pct=0,
        slippage_pct=0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        min_confidence=0,
    )
    backtester = Backtester(config)
    backtester.run(data, np.array([1, 1]))
    assert backtester.trades[0].exit_reason == "stop_loss"
    assert backtester.trades[0].exit_price == pytest.approx(98)
