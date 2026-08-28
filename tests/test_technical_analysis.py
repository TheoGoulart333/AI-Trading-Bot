import numpy as np
import pandas as pd
import pytest

from src.technical_analysis import TechnicalAnalysis


def test_rejects_missing_ohlcv_columns() -> None:
    with pytest.raises(ValueError, match="Colunas ausentes"):
        TechnicalAnalysis(pd.DataFrame({"close": [1, 2, 3]}))


def test_indicators_are_added_without_mutating_source(sample_df: pd.DataFrame) -> None:
    original = sample_df.copy(deep=True)
    result = TechnicalAnalysis(sample_df).add_all_indicators()

    expected = {
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_9",
        "ema_21",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "atr_14",
        "vwap",
    }
    assert expected <= set(result.columns)
    assert not result.isna().any().any()
    pd.testing.assert_frame_equal(sample_df, original)


def test_indicator_invariants(sample_df: pd.DataFrame) -> None:
    result = TechnicalAnalysis(sample_df).add_all_indicators()

    assert result["rsi_14"].between(0, 100).all()
    assert (result["atr_14"] >= 0).all()
    assert (result["bb_upper"] >= result["bb_middle"]).all()
    assert (result["bb_middle"] >= result["bb_lower"]).all()
    np.testing.assert_allclose(
        result["macd_histogram"],
        result["macd_line"] - result["macd_signal"],
    )


def test_methods_support_chaining(sample_df: pd.DataFrame) -> None:
    analysis = TechnicalAnalysis(sample_df)
    result = analysis.add_sma([20]).add_ema([9]).add_rsi(14)
    assert result is analysis
