import numpy as np
import pandas as pd
import pytest

from src.technical_analysis import TechnicalAnalysis


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Série OHLCV reproduzível, longa o bastante para a SMA de 200 períodos."""
    rng = np.random.default_rng(42)
    periods = 320
    dates = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    close = 42_000 + np.cumsum(rng.normal(0, 100, periods))
    open_price = close * (1 + rng.normal(0, 0.001, periods))

    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.005,
            "low": np.minimum(open_price, close) * 0.995,
            "close": close,
            "volume": rng.uniform(100, 1_000, periods),
        },
        index=dates,
    )


@pytest.fixture
def feature_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    return TechnicalAnalysis(sample_df).add_all_indicators()
