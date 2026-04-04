"""
backtesting.py
--------------
Motor de backtesting para validar estratégias de trading.

Simula a execução de ordens históricas com controle de:
    - Capital inicial e alavancagem
    - Custos de transação (taxa + slippage)
    - Gestão de risco (stop-loss, take-profit)
    - Métricas de performance (Sharpe, Drawdown, Win Rate)

⚠️  AVISO: Backtest não garante performance futura.
    Resultados passados não são indicativos de resultados futuros.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Estruturas de Dados                                                      #
# ======================================================================= #

@dataclass
class Trade:
    """Representa uma operação completa (entrada + saída)."""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    direction: str = "long"          # 'long' ou 'short'
    size: float = 0.0                # Quantidade em unidades do ativo
    pnl: float = 0.0                 # Lucro/Prejuízo em USDT
    pnl_pct: float = 0.0             # Retorno percentual
    exit_reason: str = ""            # 'signal', 'stop_loss', 'take_profit', 'end_of_data'


@dataclass
class BacktestConfig:
    """Configurações do backtesting."""
    initial_capital: float = 10_000.0   # Capital inicial em USDT
    position_size_pct: float = 0.10     # % do capital por operação (10%)
    fee_pct: float = 0.001              # Taxa de transação (0.1% — Binance maker)
    slippage_pct: float = 0.0005        # Slippage estimado (0.05%)
    stop_loss_pct: float = 0.02         # Stop-loss em 2%
    take_profit_pct: float = 0.04       # Take-profit em 4% (ratio R:R = 1:2)
    allow_short: bool = False           # Permite operações vendidas
    min_confidence: float = 0.55        # Confiança mínima do modelo para operar


# ======================================================================= #
#  Motor de Backtesting                                                     #
# ======================================================================= #

class Backtester:
    """
    Simula operações históricas usando previsões do modelo de IA.

    Fluxo:
        1. Modelo prevê direção do próximo candle (sinal: 1=Alta / 0=Baixa)
        2. Backtester abre posição long (ou short se permitido)
        3. Gerencia stop-loss e take-profit candle a candle
        4. Fecha posição no próximo sinal contrário ou nos limites de risco
        5. Calcula métricas de performance ao final
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []
        self.capital = self.config.initial_capital

    def run(self, df: pd.DataFrame, predictions: np.ndarray, probabilities: Optional[np.ndarray] = None) -> dict:
        """
        Executa o backtest sobre o DataFrame histórico.

        Args:
            df: DataFrame OHLCV com indicadores técnicos.
            predictions: Array de previsões binárias (0/1) do modelo de IA.
            probabilities: Array de probabilidades da classe positiva (opcional,
                           usado para filtrar sinais com baixa confiança).

        Returns:
            Dicionário com métricas de performance detalhadas.
        """
        self.trades = []
        self.equity_curve = []
        self.capital = self.config.initial_capital

        logger.info("=" * 60)
        logger.info("🚀 Iniciando Backtesting...")
        logger.info(f"   Capital inicial: ${self.config.initial_capital:,.2f}")
        logger.info(f"   Período: {df.index[0]} → {df.index[-1]}")
        logger.info(f"   Candles: {len(df)}")
        logger.info("=" * 60)

        current_trade: Optional[Trade] = None

        for i, (timestamp, row) in enumerate(df.iterrows()):
            self.equity_curve.append(self.capital)

            # Proteção: não opera nos últimos candles (sem dados futuros)
            if i >= len(predictions):
                break

            signal = int(predictions[i])
            confidence = float(probabilities[i]) if probabilities is not None else 1.0

            # ----------------------------------------------------------- #
            #  Gerenciamento de posição aberta                             #
            # ----------------------------------------------------------- #
            if current_trade is not None:
                current_trade, closed = self._manage_open_position(
                    current_trade, row, timestamp
                )
                if closed:
                    self.trades.append(current_trade)
                    self.capital += current_trade.pnl
                    current_trade = None
                    continue

            # ----------------------------------------------------------- #
            #  Abertura de nova posição                                    #
            # ----------------------------------------------------------- #
            if current_trade is None and confidence >= self.config.min_confidence:
                if signal == 1:  # Sinal de ALTA
                    current_trade = self._open_trade(timestamp, row, "long")
                elif signal == 0 and self.config.allow_short:  # Sinal de BAIXA
                    current_trade = self._open_trade(timestamp, row, "short")

        # Fecha posição aberta ao final dos dados
        if current_trade is not None:
            current_trade.exit_time = df.index[-1]
            current_trade.exit_price = df["close"].iloc[-1]
            current_trade = self._close_trade(current_trade, "end_of_data")
            self.trades.append(current_trade)
            self.capital += current_trade.pnl

        return self._calculate_metrics(df)

    def _open_trade(self, timestamp: pd.Timestamp, row: pd.Series, direction: str) -> Trade:
        """Cria uma nova operação aplicando fees e slippage."""
        # Slippage: assume execução a preço levemente pior
        slippage = row["close"] * self.config.slippage_pct
        entry_price = row["close"] + slippage if direction == "long" else row["close"] - slippage

        position_value = self.capital * self.config.position_size_pct
        fee = position_value * self.config.fee_pct
        size = (position_value - fee) / entry_price

        logger.debug(
            f"📈 Abrindo {direction.upper()} | "
            f"Tempo: {timestamp} | Preço: ${entry_price:,.2f} | Size: {size:.6f}"
        )
        return Trade(
            entry_time=timestamp,
            entry_price=entry_price,
            direction=direction,
            size=size,
        )

    def _manage_open_position(
        self, trade: Trade, row: pd.Series, timestamp: pd.Timestamp
    ) -> tuple[Trade, bool]:
        """Verifica stop-loss e take-profit para a posição aberta."""
        sl_price = (
            trade.entry_price * (1 - self.config.stop_loss_pct)
            if trade.direction == "long"
            else trade.entry_price * (1 + self.config.stop_loss_pct)
        )
        tp_price = (
            trade.entry_price * (1 + self.config.take_profit_pct)
            if trade.direction == "long"
            else trade.entry_price * (1 - self.config.take_profit_pct)
        )

        # Stop-loss ativado?
        if (trade.direction == "long" and row["low"] <= sl_price) or \
           (trade.direction == "short" and row["high"] >= sl_price):
            trade.exit_price = sl_price
            trade.exit_time = timestamp
            return self._close_trade(trade, "stop_loss"), True

        # Take-profit ativado?
        if (trade.direction == "long" and row["high"] >= tp_price) or \
           (trade.direction == "short" and row["low"] <= tp_price):
            trade.exit_price = tp_price
            trade.exit_time = timestamp
            return self._close_trade(trade, "take_profit"), True

        return trade, False

    def _close_trade(self, trade: Trade, reason: str) -> Trade:
        """Calcula P&L da operação fechada."""
        fee = trade.size * trade.exit_price * self.config.fee_pct

        if trade.direction == "long":
            gross_pnl = (trade.exit_price - trade.entry_price) * trade.size
        else:
            gross_pnl = (trade.entry_price - trade.exit_price) * trade.size

        trade.pnl = gross_pnl - fee
        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.size)
        trade.exit_reason = reason

        emoji = "🟢" if trade.pnl > 0 else "🔴"
        logger.debug(
            f"{emoji} Fechando por [{reason}] | "
            f"PnL: ${trade.pnl:+.2f} ({trade.pnl_pct:+.2%})"
        )
        return trade

    # ------------------------------------------------------------------ #
    #  Métricas de Performance                                            #
    # ------------------------------------------------------------------ #

    def _calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calcula e reporta métricas completas de performance."""
        if not self.trades:
            logger.warning("Nenhuma operação realizada no período.")
            return {}

        trades_df = pd.DataFrame(
            [
                {
                    "entry_time": t.entry_time, "exit_time": t.exit_time,
                    "direction": t.direction, "entry_price": t.entry_price,
                    "exit_price": t.exit_price, "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct, "exit_reason": t.exit_reason,
                }
                for t in self.trades
            ]
        )

        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] <= 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_pnl = trades_df["pnl"].sum()
        final_capital = self.config.initial_capital + total_pnl
        total_return = (final_capital / self.config.initial_capital - 1) * 100

        # Sharpe Ratio (anualizado, assumindo retornos diários)
        daily_returns = trades_df["pnl_pct"].values
        sharpe = (
            (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
            if np.std(daily_returns) > 0 else 0
        )

        # Maximum Drawdown
        equity_series = pd.Series(self.equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # Profit Factor
        gross_profit = winning_trades["pnl"].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades["pnl"].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        metrics = {
            "period": {"start": str(df.index[0]), "end": str(df.index[-1])},
            "capital": {
                "initial": self.config.initial_capital,
                "final": round(final_capital, 2),
                "total_pnl": round(total_pnl, 2),
                "total_return_pct": round(total_return, 2),
            },
            "trades": {
                "total": total_trades,
                "wins": len(winning_trades),
                "losses": len(losing_trades),
                "win_rate_pct": round(win_rate * 100, 2),
                "avg_win": round(winning_trades["pnl"].mean(), 2) if len(winning_trades) > 0 else 0,
                "avg_loss": round(losing_trades["pnl"].mean(), 2) if len(losing_trades) > 0 else 0,
            },
            "risk": {
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown_pct": round(max_drawdown, 2),
                "profit_factor": round(profit_factor, 3),
            },
            "exit_reasons": trades_df["exit_reason"].value_counts().to_dict(),
        }

        self._print_report(metrics)
        return metrics

    def _print_report(self, metrics: dict) -> None:
        """Imprime relatório formatado no terminal."""
        c = metrics["capital"]
        t = metrics["trades"]
        r = metrics["risk"]

        logger.info("\n" + "=" * 60)
        logger.info("📊 RELATÓRIO DE BACKTESTING")
        logger.info("=" * 60)
        logger.info(f"  Retorno Total   : {c['total_return_pct']:+.2f}%")
        logger.info(f"  Capital Final   : ${c['final']:>10,.2f}")
        logger.info(f"  P&L Total       : ${c['total_pnl']:>+10,.2f}")
        logger.info("-" * 60)
        logger.info(f"  Total de Trades : {t['total']}")
        logger.info(f"  Win Rate        : {t['win_rate_pct']:.2f}%")
        logger.info(f"  Média Win       : ${t['avg_win']:>+8,.2f}")
        logger.info(f"  Média Loss      : ${t['avg_loss']:>+8,.2f}")
        logger.info("-" * 60)
        logger.info(f"  Sharpe Ratio    : {r['sharpe_ratio']:.3f}")
        logger.info(f"  Max Drawdown    : {r['max_drawdown_pct']:.2f}%")
        logger.info(f"  Profit Factor   : {r['profit_factor']:.3f}")
        logger.info("=" * 60)
        logger.info(f"  Motivos de Saída: {metrics['exit_reasons']}")
        logger.info("=" * 60)


# --- Execução standalone para testes rápidos ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simula 300 candles de BTC
    dates = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    prices = np.cumsum(np.random.randn(300) * 100) + 42000
    mock_df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices + np.random.randn(300) * 50,
            "volume": np.random.uniform(100, 1000, 300),
        },
        index=dates,
    )

    # Previsões aleatórias (benchmark ingênuo)
    mock_predictions = np.random.randint(0, 2, 300)
    mock_proba = np.random.uniform(0.45, 0.65, 300)

    config = BacktestConfig(initial_capital=10000, stop_loss_pct=0.02, take_profit_pct=0.04)
    backtester = Backtester(config)
    results = backtester.run(mock_df, mock_predictions, mock_proba)
