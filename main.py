"""
main.py
-------
Pipeline principal do AI Trading Bot.

Orquestra todos os módulos em sequência:
    1. Ingestão de Dados  →  data_ingestion.py
    2. Análise Técnica    →  technical_analysis.py
    3. Modelo de IA       →  ai_model.py
    4. Backtesting        →  backtesting.py
    5. Relatório Final    →  logs + JSON

Uso:
    python main.py                        # Configuração padrão
    python main.py --symbol ETH/USDT      # Outro ativo
    python main.py --timeframe 4h         # Outro timeframe
    python main.py --model lstm           # Usar LSTM (requer TensorFlow)
    python main.py --dry-run              # Usar dados sintéticos (sem API)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ======================================================================= #
#  Configuração de Logging                                                  #
# ======================================================================= #

def setup_logging(log_level: str = "INFO") -> None:
    """
    Configura logging para console (colorido) e arquivo rotativo.

    Args:
        log_level: Nível de log ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"trading_bot_{timestamp}.log"

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.info(f"📄 Log salvo em: {log_file}")


logger = logging.getLogger(__name__)


# ======================================================================= #
#  Gerador de Dados Sintéticos (Modo Dry-Run)                               #
# ======================================================================= #

def generate_synthetic_data(n_candles: int = 500) -> pd.DataFrame:
    """
    Gera dados OHLCV sintéticos com padrão de random walk.
    Útil para testar o pipeline sem dependência de API externa.

    Args:
        n_candles: Número de candles a gerar.

    Returns:
        DataFrame OHLCV com index temporal.
    """
    logger.info(f"🎲 Gerando {n_candles} candles sintéticos (modo dry-run)...")
    np.random.seed(42)

    dates = pd.date_range("2024-01-01", periods=n_candles, freq="1h", tz="UTC")
    close = np.cumsum(np.random.randn(n_candles) * 150) + 42000
    close = np.abs(close)  # Garante preços positivos

    volatility = np.random.uniform(0.003, 0.008, n_candles)
    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n_candles) * 0.001),
            "high": close * (1 + volatility),
            "low": close * (1 - volatility),
            "close": close,
            "volume": np.random.uniform(500, 5000, n_candles),
        },
        index=dates,
    )
    # Garante consistência OHLC
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


# ======================================================================= #
#  Pipeline Principal                                                        #
# ======================================================================= #

def run_pipeline(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 500,
    model_type: str = "random_forest",
    dry_run: bool = False,
    output_dir: str = "results",
) -> dict:
    """
    Executa o pipeline completo de análise e backtesting.

    Args:
        symbol: Par de trading (ex: 'BTC/USDT').
        timeframe: Intervalo dos candles.
        limit: Número de candles históricos.
        model_type: 'random_forest' ou 'lstm'.
        dry_run: Se True, usa dados sintéticos sem acesso à API.
        output_dir: Diretório para salvar resultados.

    Returns:
        Dicionário com resultados e métricas.
    """
    logger.info("=" * 65)
    logger.info("  🤖 AI TRADING BOT — Pipeline de Análise e Backtesting")
    logger.info("=" * 65)
    logger.info(f"  Ativo     : {symbol}")
    logger.info(f"  Timeframe : {timeframe}")
    logger.info(f"  Candles   : {limit}")
    logger.info(f"  Modelo    : {model_type}")
    logger.info(f"  Modo      : {'DRY-RUN (sintético)' if dry_run else 'LIVE DATA'}")
    logger.info("=" * 65)

    results_dir = Path(output_dir)
    results_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ #
    #  ETAPA 1: Ingestão de Dados                                         #
    # ------------------------------------------------------------------ #
    logger.info("\n📥 [ETAPA 1/4] Ingestão de Dados...")

    if dry_run:
        df_raw = generate_synthetic_data(n_candles=limit)
    else:
        from src.data_ingestion import DataIngestion
        ingestion = DataIngestion(exchange_id="binance")
        df_raw = ingestion.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    logger.info(f"✅ Dados carregados: {len(df_raw)} candles")

    # ------------------------------------------------------------------ #
    #  ETAPA 2: Análise Técnica                                           #
    # ------------------------------------------------------------------ #
    logger.info("\n📊 [ETAPA 2/4] Calculando Indicadores Técnicos...")

    from src.technical_analysis import TechnicalAnalysis
    ta = TechnicalAnalysis(df_raw)
    df_features = ta.add_all_indicators()
    logger.info(f"✅ {len(df_features.columns)} indicadores calculados")

    # ------------------------------------------------------------------ #
    #  ETAPA 3: Treinamento e Avaliação do Modelo de IA                  #
    # ------------------------------------------------------------------ #
    logger.info(f"\n🧠 [ETAPA 3/4] Treinando modelo: {model_type}...")

    from src.ai_model import LSTMModel, RandomForestModel

    if model_type == "lstm":
        model = LSTMModel(lookback=60, epochs=30)
    else:
        model = RandomForestModel(n_estimators=200)

    X, y = model.prepare_features(df_features)

    # Split temporal (80% treino / 20% teste) — NUNCA use split aleatório em séries temporais!
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"   Amostras treino : {len(X_train)}")
    logger.info(f"   Amostras teste  : {len(X_test)}")

    model.fit(X_train, y_train)
    eval_metrics = model.evaluate(X_test, y_test)
    accuracy = eval_metrics["classification_report"].get("accuracy", 0)
    logger.info(f"✅ Acurácia no conjunto de teste: {accuracy:.2%}")

    # Salva modelo treinado
    if model_type == "random_forest":
        model_path = str(results_dir / "model_rf.pkl")
        model.save(model_path)

    # Gera previsões para o período de teste (para o backtesting)
    predictions = model.predict(X_test)
    probabilities = None
    if model_type == "random_forest" and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------ #
    #  ETAPA 4: Backtesting                                               #
    # ------------------------------------------------------------------ #
    logger.info("\n🔄 [ETAPA 4/4] Executando Backtesting...")

    from src.backtesting import Backtester, BacktestConfig

    # Alinha o DataFrame com o período de teste
    # (desconta o warmup dos indicadores e o split de treino)
    n_warmup = len(df_features) - len(X)  # linhas removidas pelo dropna
    test_df = df_features.iloc[n_warmup + split_idx : n_warmup + split_idx + len(X_test)]

    config = BacktestConfig(
        initial_capital=10_000,
        position_size_pct=0.10,
        fee_pct=0.001,
        slippage_pct=0.0005,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        min_confidence=0.55,
    )
    backtester = Backtester(config)
    backtest_results = backtester.run(test_df, predictions, probabilities)

    # ------------------------------------------------------------------ #
    #  Salva Resultados                                                   #
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "run_config": {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "model": model_type,
            "dry_run": dry_run,
            "timestamp": timestamp,
        },
        "model_accuracy": round(accuracy, 4),
        "backtest": backtest_results,
    }

    output_file = results_dir / f"results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n💾 Resultados salvos em: {output_file}")

    logger.info("\n✅ Pipeline concluído com sucesso!")
    return results


# ======================================================================= #
#  Entry Point CLI                                                           #
# ======================================================================= #

def parse_args() -> argparse.Namespace:
    """Parse dos argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="🤖 AI Trading Bot — Backtesting com Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py --dry-run
  python main.py --symbol ETH/USDT --timeframe 4h --limit 1000
  python main.py --model lstm --limit 2000

⚠️  DISCLAIMER: Este software é apenas para fins educacionais.
    Não constitui aconselhamento financeiro. Use por sua conta e risco.
        """,
    )
    parser.add_argument("--symbol",    default="BTC/USDT",     help="Par de trading (padrão: BTC/USDT)")
    parser.add_argument("--timeframe", default="1h",            help="Timeframe (padrão: 1h)")
    parser.add_argument("--limit",     default=500, type=int,   help="Número de candles (padrão: 500)")
    parser.add_argument("--model",     default="random_forest", choices=["random_forest", "lstm"],
                        help="Modelo de IA (padrão: random_forest)")
    parser.add_argument("--log-level", default="INFO",          help="Nível de log (padrão: INFO)")
    parser.add_argument("--dry-run",   action="store_true",     help="Usa dados sintéticos (sem API)")
    parser.add_argument("--output",    default="results",       help="Diretório de saída")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)

    try:
        results = run_pipeline(
            symbol=args.symbol,
            timeframe=args.timeframe,
            limit=args.limit,
            model_type=args.model,
            dry_run=args.dry_run,
            output_dir=args.output,
        )
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Execução interrompida pelo usuário.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"❌ Erro crítico no pipeline: {e}")
        sys.exit(1)
