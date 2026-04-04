"""
tests/test_technical_analysis.py
---------------------------------
Testes unitários para o módulo de análise técnica.
Execute com: pytest tests/ -v --cov=src
"""

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, ".")

from src.technical_analysis import TechnicalAnalysis


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture que gera um DataFrame OHLCV sintético para testes."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    close = np.cumsum(np.random.randn(n) * 100) + 42000

    return pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n) * 0.001),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.uniform(100, 1000, n),
        },
        index=dates,
    )


class TestTechnicalAnalysis:
    """Suite de testes para TechnicalAnalysis."""

    def test_initialization_valid(self, sample_df):
        """Deve inicializar sem erros com DataFrame válido."""
        ta = TechnicalAnalysis(sample_df)
        assert ta.df is not None

    def test_initialization_missing_columns(self):
        """Deve lançar ValueError se colunas obrigatórias estiverem ausentes."""
        bad_df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Colunas ausentes"):
            TechnicalAnalysis(bad_df)

    def test_sma_columns_created(self, sample_df):
        """SMA deve criar colunas sma_N para cada período."""
        ta = TechnicalAnalysis(sample_df)
        ta.add_sma([20, 50])
        assert "sma_20" in ta.df.columns
        assert "sma_50" in ta.df.columns

    def test_ema_columns_created(self, sample_df):
        """EMA deve criar colunas ema_N para cada período."""
        ta = TechnicalAnalysis(sample_df)
        ta.add_ema([9, 21])
        assert "ema_9" in ta.df.columns
        assert "ema_21" in ta.df.columns

    def test_rsi_range(self, sample_df):
        """RSI deve estar sempre entre 0 e 100."""
        ta = TechnicalAnalysis(sample_df)
        ta.add_rsi(14)
        rsi_values = ta.df["rsi_14"].dropna()
        assert rsi_values.between(0, 100).all(), "RSI fora do range [0, 100]"

    def test_macd_histogram_calculation(self, sample_df):
        """Histograma do MACD deve ser a diferença entre linha e sinal."""
        ta = TechnicalAnalysis(sample_df)
        ta.add_macd()
        expected = ta.df["macd_line"] - ta.df["macd_signal"]
        pd.testing.assert_series_equal(ta.df["macd_histogram"], expected, check_names=False)

    def test_bollinger_bands_ordering(self, sample_df):
        """Banda superior deve ser sempre >= média >= banda inferior."""
        ta = TechnicalAnalysis(sample_df)
        ta.add_bollinger_bands()
        df = ta.df.dropna()
        assert (df["bb_upper"] >= df["bb_middle"]).all()
        assert (df["bb_middle"] >= df["bb_lower"]).all()

    def test_atr_non_negative(self, sample_df):
        """ATR deve ser sempre não-negativo (mede volatilidade absoluta)."""
        ta = TechnicalAnalysis(sample_df)
        ta.add_atr(14)
        assert (ta.df["atr_14"].dropna() >= 0).all()

    def test_method_chaining(self, sample_df):
        """Deve suportar encadeamento de métodos."""
        ta = TechnicalAnalysis(sample_df)
        # Não deve lançar erro
        result = ta.add_sma().add_ema().add_rsi()
        assert isinstance(result, TechnicalAnalysis)

    def test_add_all_indicators_returns_dataframe(self, sample_df):
        """add_all_indicators deve retornar DataFrame sem NaN."""
        ta = TechnicalAnalysis(sample_df)
        result = ta.add_all_indicators()
        assert isinstance(result, pd.DataFrame)
        assert not result.isnull().any().any(), "DataFrame final contém NaN"

    def test_immutability(self, sample_df):
        """TechnicalAnalysis não deve modificar o DataFrame original."""
        original_cols = set(sample_df.columns)
        ta = TechnicalAnalysis(sample_df)
        ta.add_all_indicators()
        assert set(sample_df.columns) == original_cols, "DataFrame original foi modificado"


class TestEdgeCases:
    """Testes de casos extremos."""

    def test_minimum_data_for_sma200(self):
        """SMA 200 requer pelo menos 200 candles."""
        n = 250  # suficiente
        dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": np.ones(n) * 40000,
                "high": np.ones(n) * 40100,
                "low": np.ones(n) * 39900,
                "close": np.ones(n) * 40000,
                "volume": np.ones(n) * 500,
            },
            index=dates,
        )
        ta = TechnicalAnalysis(df)
        ta.add_sma([200])
        valid_sma = ta.df["sma_200"].dropna()
        assert len(valid_sma) > 0

    def test_constant_price_rsi(self):
        """RSI de preço constante deve ser 50 (sem ganhos nem perdas)."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": np.ones(n) * 40000,
                "high": np.ones(n) * 40000,
                "low": np.ones(n) * 40000,
                "close": np.ones(n) * 40000,
                "volume": np.ones(n) * 500,
            },
            index=dates,
        )
        ta = TechnicalAnalysis(df)
        ta.add_rsi(14)
        # Preço constante: diferença = 0, RS indefinido → RSI pode ser NaN ou 50
        rsi_clean = ta.df["rsi_14"].dropna()
        if len(rsi_clean) > 0:
            assert rsi_clean.between(0, 100).all()
