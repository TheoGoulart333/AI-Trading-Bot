"""
technical_analysis.py
---------------------
Módulo de análise técnica para cálculo de indicadores de mercado.
Todos os indicadores são calculados manualmente com Pandas/NumPy
para fins didáticos e independência de bibliotecas externas.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalAnalysis:
    """
    Calcula indicadores técnicos clássicos sobre um DataFrame OHLCV.
    Os indicadores são adicionados como novas colunas ao DataFrame.

    Indicadores disponíveis:
        - SMA  : Simple Moving Average
        - EMA  : Exponential Moving Average
        - RSI  : Relative Strength Index
        - MACD : Moving Average Convergence Divergence
        - BB   : Bollinger Bands
        - ATR  : Average True Range
        - VWAP : Volume Weighted Average Price
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame OHLCV com colunas open, high, low, close, volume.
        """
        self._validate_dataframe(df)
        self.df = df.copy()

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Valida se o DataFrame possui as colunas obrigatórias."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colunas ausentes no DataFrame: {missing}")

    # ------------------------------------------------------------------ #
    #  Médias Móveis                                                       #
    # ------------------------------------------------------------------ #

    def add_sma(self, periods: list[int] = [20, 50, 200]) -> "TechnicalAnalysis":
        """
        Adiciona Médias Móveis Simples (SMA).

        Args:
            periods: Lista de períodos a calcular.
        """
        for period in periods:
            col_name = f"sma_{period}"
            self.df[col_name] = self.df["close"].rolling(window=period).mean()
            logger.debug(f"Indicador adicionado: {col_name}")
        return self  # Suporta encadeamento: ta.add_sma().add_rsi()

    def add_ema(self, periods: list[int] = [9, 21]) -> "TechnicalAnalysis":
        """
        Adiciona Médias Móveis Exponenciais (EMA).

        Args:
            periods: Lista de períodos a calcular.
        """
        for period in periods:
            col_name = f"ema_{period}"
            self.df[col_name] = (
                self.df["close"].ewm(span=period, adjust=False).mean()
            )
            logger.debug(f"Indicador adicionado: {col_name}")
        return self

    # ------------------------------------------------------------------ #
    #  Momentum / Força                                                    #
    # ------------------------------------------------------------------ #

    def add_rsi(self, period: int = 14) -> "TechnicalAnalysis":
        """
        Adiciona o RSI (Relative Strength Index).
        Valores > 70 indicam sobrecompra; < 30 indicam sobrevenda.

        Args:
            period: Período de lookback (padrão: 14).
        """
        delta = self.df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)  # Evita divisão por zero
        self.df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        logger.debug(f"Indicador adicionado: rsi_{period}")
        return self

    # ------------------------------------------------------------------ #
    #  Trend / Divergência                                                 #
    # ------------------------------------------------------------------ #

    def add_macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> "TechnicalAnalysis":
        """
        Adiciona MACD (Moving Average Convergence Divergence).
        Gera 3 colunas: macd_line, macd_signal, macd_histogram.

        Args:
            fast: Período da EMA rápida.
            slow: Período da EMA lenta.
            signal: Período da linha de sinal.
        """
        ema_fast = self.df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["close"].ewm(span=slow, adjust=False).mean()

        self.df["macd_line"] = ema_fast - ema_slow
        self.df["macd_signal"] = (
            self.df["macd_line"].ewm(span=signal, adjust=False).mean()
        )
        self.df["macd_histogram"] = self.df["macd_line"] - self.df["macd_signal"]
        logger.debug("Indicador adicionado: MACD (line, signal, histogram)")
        return self

    # ------------------------------------------------------------------ #
    #  Volatilidade                                                        #
    # ------------------------------------------------------------------ #

    def add_bollinger_bands(
        self, period: int = 20, std_dev: float = 2.0
    ) -> "TechnicalAnalysis":
        """
        Adiciona Bandas de Bollinger.
        Gera 3 colunas: bb_upper, bb_middle (SMA), bb_lower.

        Args:
            period: Período da SMA central.
            std_dev: Número de desvios padrão para as bandas.
        """
        sma = self.df["close"].rolling(window=period).mean()
        std = self.df["close"].rolling(window=period).std()

        self.df["bb_upper"] = sma + (std * std_dev)
        self.df["bb_middle"] = sma
        self.df["bb_lower"] = sma - (std * std_dev)
        self.df["bb_width"] = (self.df["bb_upper"] - self.df["bb_lower"]) / sma
        logger.debug("Indicador adicionado: Bollinger Bands")
        return self

    def add_atr(self, period: int = 14) -> "TechnicalAnalysis":
        """
        Adiciona ATR (Average True Range) — mede a volatilidade do mercado.

        Args:
            period: Período de lookback.
        """
        high_low = self.df["high"] - self.df["low"]
        high_close = (self.df["high"] - self.df["close"].shift()).abs()
        low_close = (self.df["low"] - self.df["close"].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df[f"atr_{period}"] = true_range.ewm(com=period - 1, adjust=False).mean()
        logger.debug(f"Indicador adicionado: atr_{period}")
        return self

    # ------------------------------------------------------------------ #
    #  Volume                                                              #
    # ------------------------------------------------------------------ #

    def add_vwap(self) -> "TechnicalAnalysis":
        """
        Adiciona VWAP (Volume Weighted Average Price).
        Indicador de referência institucional.
        """
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        vwap = (typical_price * self.df["volume"]).cumsum() / self.df["volume"].cumsum()
        self.df["vwap"] = vwap
        logger.debug("Indicador adicionado: VWAP")
        return self

    # ------------------------------------------------------------------ #
    #  Pipeline completo                                                   #
    # ------------------------------------------------------------------ #

    def add_all_indicators(self) -> pd.DataFrame:
        """
        Método conveniente que adiciona todos os indicadores com configurações padrão.

        Returns:
            DataFrame enriquecido com todos os indicadores.
        """
        logger.info("Calculando todos os indicadores técnicos...")
        (
            self.add_sma([20, 50, 200])
            .add_ema([9, 21])
            .add_rsi(14)
            .add_macd()
            .add_bollinger_bands()
            .add_atr()
            .add_vwap()
        )
        # Remove linhas com NaN geradas pelo período de aquecimento dos indicadores
        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        dropped = initial_rows - len(self.df)
        logger.info(
            f"✅ Indicadores calculados. {dropped} linhas de aquecimento removidas. "
            f"{len(self.df)} candles disponíveis para análise."
        )
        return self.df


# --- Execução standalone para testes rápidos ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Simula dados para teste sem precisar de API
    dates = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    mock_df = pd.DataFrame(
        {
            "open": np.random.uniform(40000, 45000, 300),
            "high": np.random.uniform(45000, 50000, 300),
            "low": np.random.uniform(35000, 40000, 300),
            "close": np.random.uniform(40000, 45000, 300),
            "volume": np.random.uniform(100, 1000, 300),
        },
        index=dates,
    )
    ta = TechnicalAnalysis(mock_df)
    result = ta.add_all_indicators()
    print(result.tail())
    print(f"\nColunas geradas: {list(result.columns)}")
