# 🤖 AI Trading Bot — Machine Learning meets Financial Markets

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![CCXT](https://img.shields.io/badge/CCXT-4.2+-2B2B2B?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Educational-yellow?style=for-the-badge)

**Um projeto de portfólio demonstrando a interseção entre Engenharia de Software, Machine Learning e Finanças Quantitativas.**

[Arquitetura](#-arquitetura) • [Instalação](#-instalação) • [Uso](#-uso) • [Módulos](#-módulos) • [Disclaimer](#️-disclaimer)

</div>

---

## 🎯 Sobre o Projeto

Este projeto implementa um **bot de análise de mercado baseado em IA** com pipeline completo:
ingestão de dados → análise técnica → modelo preditivo → backtesting. O objetivo é demonstrar
boas práticas de engenharia de software em um domínio complexo e real.

> ⚠️ Este projeto é **exclusivamente educacional**. Não é um sistema de trading em produção,
> não gerencia dinheiro real e não garante retornos financeiros de qualquer espécie.

### ✨ Destaques Técnicos

- **Arquitetura modular** com Programação Orientada a Objetos e interfaces abstratas (Strategy Pattern)
- **Pipeline de ML completo**: feature engineering, TimeSeriesSplit, validação cruzada temporal
- **Dois modelos intercambiáveis**: Random Forest (baseline interpretável) e LSTM (deep learning sequencial)
- **Motor de backtesting** com simulação realista de taxas, slippage, stop-loss e take-profit
- **Métricas financeiras** profissionais: Sharpe Ratio, Maximum Drawdown, Profit Factor
- **Cobertura de testes** com pytest e fixtures para dados sintéticos
- **Código PEP8** com docstrings completas e logging estruturado

---

## 🏗 Arquitetura

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AI TRADING BOT PIPELINE                       │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐
  │  Exchange    │    │  Data Ingestion   │    │ Technical Analysis   │
  │  (CCXT)      │───▶│  OHLCV Fetcher   │───▶│  RSI, MACD, BB,      │
  │  Binance     │    │  Rate Limiting    │    │  EMA, SMA, ATR,      │
  │  Kraken...   │    │  Error Handling   │    │  VWAP, Volume Ratio  │
  └──────────────┘    └──────────────────┘    └──────────┬───────────┘
                                                          │
                       ┌──────────────────────────────────▼──────────┐
                       │          Feature Engineering                  │
                       │  Returns % | Volume Anomaly | Target Label   │
                       └──────────────────────┬──────────────────────┘
                                              │
             ┌────────────────────────────────▼───────────────────────┐
             │                    AI Model Layer                        │
             │                                                          │
             │  ┌─────────────────────┐   ┌──────────────────────┐    │
             │  │   Random Forest     │   │        LSTM           │    │
             │  │  (Baseline, Fast)   │   │  (Sequential, Deep)   │    │
             │  │  Feature Importance │   │  Lookback Window: 60  │    │
             │  └─────────────────────┘   └──────────────────────┘    │
             │          TimeSeriesSplit — sem look-ahead bias           │
             └────────────────────────────┬───────────────────────────┘
                                          │
             ┌────────────────────────────▼───────────────────────────┐
             │                   Backtesting Engine                     │
             │  Capital Mgmt | Stop-Loss | Take-Profit | Slippage      │
             └────────────────────────────┬───────────────────────────┘
                                          │
             ┌────────────────────────────▼───────────────────────────┐
             │                    Results & Reporting                   │
             │   Sharpe Ratio | Max Drawdown | Win Rate | JSON + Logs  │
             └────────────────────────────────────────────────────────┘
```

### Fluxo de Dados

1. **DataIngestion** conecta à exchange via CCXT e busca N candles OHLCV com rate limiting automático
2. **TechnicalAnalysis** enriquece o DataFrame com 15+ indicadores usando Pandas (sem dependência de TA-Lib)
3. **Feature Engineering** gera retornos percentuais, anomalias de volume e o target label binário
4. **AI Model** treina com `TimeSeriesSplit` para respeitar a ordem temporal e evitar data leakage
5. **Backtester** simula operações no conjunto de teste com custos realistas e gestão de risco
6. **Reporting** salva métricas em JSON e logs estruturados para auditoria

---

## 📁 Estrutura de Pastas

```
ai_trading_bot/
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py      # Módulo de coleta de dados via CCXT
│   ├── technical_analysis.py  # Cálculo de indicadores técnicos
│   ├── ai_model.py            # Random Forest + LSTM (BaseModel interface)
│   └── backtesting.py         # Simulador de estratégias
│
├── tests/
│   ├── __init__.py
│   └── test_technical_analysis.py  # Testes unitários (pytest)
│
├── results/                   # JSONs de resultados (gerado automaticamente)
├── logs/                      # Arquivos de log (gerado automaticamente)
│
├── main.py                    # Orquestrador do pipeline (CLI)
├── requirements.txt
└── README.md
```

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.11+
- pip

### Passo a Passo

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/ai-trading-bot.git
cd ai-trading-bot

# 2. Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
# ou: venv\Scripts\activate  # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. (Opcional) Para usar o modelo LSTM
pip install tensorflow
```

---

## 💻 Uso

### Modo Dry-Run (recomendado para testes — sem API)

```bash
python main.py --dry-run
```

### Com dados reais (Binance, sem autenticação para dados públicos)

```bash
# BTC/USDT, timeframe de 1 hora, 500 candles
python main.py --symbol BTC/USDT --timeframe 1h --limit 500

# ETH/USDT, timeframe de 4 horas
python main.py --symbol ETH/USDT --timeframe 4h --limit 1000

# Com modelo LSTM (requer TensorFlow instalado)
python main.py --model lstm --limit 2000 --dry-run
```

### Executar Testes

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Argumentos da CLI

| Argumento    | Padrão         | Descrição                                |
|:-------------|:---------------|:-----------------------------------------|
| `--symbol`   | `BTC/USDT`     | Par de trading da exchange               |
| `--timeframe`| `1h`           | Intervalo dos candles                    |
| `--limit`    | `500`          | Número de candles históricos             |
| `--model`    | `random_forest`| Modelo de IA: `random_forest` ou `lstm` |
| `--dry-run`  | `False`        | Usa dados sintéticos (sem API)           |
| `--log-level`| `INFO`         | Nível de verbosidade dos logs            |
| `--output`   | `results/`     | Diretório para salvar resultados         |

---

## 📦 Módulos

### `src/data_ingestion.py` — DataIngestion
Conecta a 100+ exchanges via CCXT. Features:
- Busca candles OHLCV com retry automático
- Rate limiting integrado (`enableRateLimit=True`)
- Suporte a modo sandbox para testes seguros
- Busca multi-símbolo com tratamento de falhas parciais

### `src/technical_analysis.py` — TechnicalAnalysis
Calcula indicadores diretamente com Pandas/NumPy (zero dependência de TA-Lib):
- **Médias**: SMA (20/50/200), EMA (9/21)
- **Momentum**: RSI (14), MACD (12/26/9)
- **Volatilidade**: Bollinger Bands, ATR
- **Volume**: VWAP, Volume Ratio
- API fluente com method chaining: `ta.add_sma().add_rsi().add_macd()`

### `src/ai_model.py` — RandomForestModel / LSTMModel
Interface `BaseModel` abstrata (Strategy Pattern) com duas implementações:

| Aspecto | Random Forest | LSTM |
|:--------|:--------------|:-----|
| Dados mínimos | ~300 candles | ~2000 candles |
| Velocidade de treino | Segundos | Minutos |
| Interpretabilidade | ✅ Feature Importance | ❌ Caixa preta |
| Captura temporal | Limitada | Excelente |
| Recomendado para | Início e baseline | Experimentos avançados |

### `src/backtesting.py` — Backtester
Simula operações históricas com:
- Gestão de capital por percentual (`position_size_pct`)
- Custos reais: taxa Binance (0.1%) + slippage (0.05%)
- Stop-loss e take-profit por candle
- Filtro de confiança mínima do modelo (`min_confidence`)
- Métricas: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor

---

## 🧪 Decisões de Design e Conceitos Importantes

### Por que TimeSeriesSplit e não train_test_split aleatório?

Em séries temporais, usar split aleatório causa **data leakage** (vazamento de dados): o modelo
"vê" o futuro durante o treino, inflando artificialmente as métricas. O `TimeSeriesSplit` garante
que o modelo seja sempre treinado com dados do passado e avaliado com dados do futuro.

### Por que Random Forest como baseline?

Random Forest é resistente a overfitting, treina rapidamente e fornece **feature importance** —
essencial para entender quais indicadores técnicos realmente contribuem para as previsões.
É o ponto de partida ideal antes de escalar para modelos mais complexos.

### Por que simular slippage no backtesting?

Slippage é a diferença entre o preço esperado e o preço executado, comum em mercados com pouca
liquidez. Ignorá-lo produz backtests irrealisticamente otimistas (overfitting de estratégia).

---

## ⚠️ Disclaimer

> **Este projeto é estritamente educacional e de portfólio.**
>
> - Não constitui aconselhamento financeiro, de investimento ou recomendação de qualquer tipo.
> - Resultados de backtesting **não garantem** performance futura.
> - Trading de criptomoedas e outros ativos envolve **risco substancial de perda**.
> - Os autores não se responsabilizam por qualquer perda financeira decorrente do uso deste código.
> - Consulte um profissional financeiro certificado antes de tomar decisões de investimento.
>
> **Nunca use dinheiro que você não pode se dar ao luxo de perder.**

---

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

---

<div align="center">

Feito com 🧠 e ☕ — Para aprender, questionar e evoluir.

</div>
