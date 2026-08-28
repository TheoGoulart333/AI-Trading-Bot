"""
data_ingestion.py
-----------------
Módulo de ingestão de dados históricos de preços via CCXT.
Responsável por buscar OHLCV (Open, High, Low, Close, Volume) de exchanges.
"""

import logging
import time

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Classe responsável por buscar e normalizar dados de mercado
    utilizando a biblioteca CCXT (suporta 100+ exchanges).
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        sandbox: bool = True,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ):
        """
        Inicializa a conexão com a exchange.

        Args:
            exchange_id: ID da exchange suportada pelo CCXT (ex: 'binance', 'kraken').
            sandbox: Se True, usa ambiente de testes (sem dinheiro real).
            max_retries: Número máximo de novas tentativas em falhas de rede.
            retry_backoff: Espera inicial, em segundos, entre tentativas.
        """
        if max_retries < 0:
            raise ValueError("max_retries não pode ser negativo.")
        if retry_backoff < 0:
            raise ValueError("retry_backoff não pode ser negativo.")
        self.exchange_id = exchange_id
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.exchange = self._initialize_exchange(sandbox)

    def _initialize_exchange(self, sandbox: bool) -> ccxt.Exchange:
        """Inicializa e configura o objeto da exchange."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class(
                {
                    "enableRateLimit": True,  # Respeita os limites de requisição da API
                    "timeout": 30000,
                }
            )
            if sandbox and exchange.has.get("sandbox"):
                exchange.set_sandbox_mode(True)
                logger.info(f"Exchange '{self.exchange_id}' iniciada em modo SANDBOX.")
            else:
                logger.info(f"Exchange '{self.exchange_id}' iniciada em modo LIVE.")
            return exchange
        except AttributeError as error:
            raise ValueError(
                f"Exchange '{self.exchange_id}' não encontrada no CCXT."
            ) from error

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        since: int | None = None,
    ) -> pd.DataFrame:
        """
        Busca dados históricos OHLCV e retorna um DataFrame estruturado.

        Args:
            symbol: Par de trading (ex: 'BTC/USDT').
            timeframe: Intervalo de tempo ('1m', '5m', '1h', '1d', etc.).
            limit: Número de candles a buscar.
            since: Timestamp Unix em milissegundos para início da busca.

        Returns:
            DataFrame com colunas: timestamp, open, high, low, close, volume.
        """
        logger.info(
            "Buscando %s candles de %s (%s) na %s...",
            limit,
            symbol,
            timeframe,
            self.exchange_id,
        )

        raw_data = None
        for attempt in range(self.max_retries + 1):
            try:
                raw_data = self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=limit, since=since
                )
                break
            except ccxt.NetworkError as error:
                if attempt == self.max_retries:
                    logger.error(
                        "Falha de rede após %s tentativas: %s", attempt + 1, error
                    )
                    raise
                wait_seconds = self.retry_backoff * (2**attempt)
                logger.warning(
                    "Falha de rede. Nova tentativa em %.1fs (%s/%s).",
                    wait_seconds,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(wait_seconds)
            except ccxt.ExchangeError as error:
                logger.error("Erro não recuperável da exchange: %s", error)
                raise

        if not raw_data:
            raise ValueError(f"Nenhum dado retornado para {symbol}/{timeframe}.")

        df = pd.DataFrame(
            raw_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        # Garante tipos numéricos corretos
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col])

        logger.info(
            f"✅ {len(df)} candles carregados. Período: {df.index[0]} → {df.index[-1]}"
        )
        return df

    def fetch_multiple_symbols(
        self, symbols: list[str], timeframe: str = "1h", limit: int = 500
    ) -> dict[str, pd.DataFrame]:
        """
        Busca dados para múltiplos símbolos com rate limiting.

        Args:
            symbols: Lista de pares (ex: ['BTC/USDT', 'ETH/USDT']).
            timeframe: Intervalo de tempo.
            limit: Número de candles por símbolo.

        Returns:
            Dicionário {symbol: DataFrame}.
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch_ohlcv(symbol, timeframe, limit)
                time.sleep(self.exchange.rateLimit / 1000)  # Respeita o rate limit
            except Exception as e:
                logger.warning(f"Falha ao buscar {symbol}: {e}. Pulando...")
        return data


# --- Execução standalone para testes rápidos ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestion = DataIngestion(exchange_id="binance")
    df = ingestion.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=100)
    print(df.tail())
