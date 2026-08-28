from unittest.mock import Mock

import ccxt
import pandas as pd
import pytest

from src.data_ingestion import DataIngestion


def make_ingestion(monkeypatch, exchange: Mock, **kwargs) -> DataIngestion:
    monkeypatch.setattr(DataIngestion, "_initialize_exchange", lambda *_: exchange)
    return DataIngestion(**kwargs)


def test_retries_transient_network_errors(monkeypatch) -> None:
    exchange = Mock(rateLimit=0)
    exchange.fetch_ohlcv.side_effect = [
        ccxt.NetworkError("temporário"),
        [[1_700_000_000_000, 1, 2, 0.5, 1.5, 100]],
    ]
    sleep = Mock()
    monkeypatch.setattr("src.data_ingestion.time.sleep", sleep)
    ingestion = make_ingestion(monkeypatch, exchange, max_retries=2, retry_backoff=0.25)

    result = ingestion.fetch_ohlcv("BTC/USDT")

    assert isinstance(result.index, pd.DatetimeIndex)
    assert exchange.fetch_ohlcv.call_count == 2
    sleep.assert_called_once_with(0.25)


def test_does_not_retry_exchange_errors(monkeypatch) -> None:
    exchange = Mock(rateLimit=0)
    exchange.fetch_ohlcv.side_effect = ccxt.ExchangeError("símbolo inválido")
    ingestion = make_ingestion(monkeypatch, exchange, max_retries=2)

    with pytest.raises(ccxt.ExchangeError):
        ingestion.fetch_ohlcv("INVALID")
    assert exchange.fetch_ohlcv.call_count == 1
