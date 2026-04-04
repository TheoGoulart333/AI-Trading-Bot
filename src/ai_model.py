"""
ai_model.py
-----------
Módulo de Inteligência Artificial para previsão de tendência de mercado.

Contém dois modelos:
    1. RandomForestModel  — baseline rápido e interpretável.
    2. LSTMModel          — rede neural recorrente para padrões sequenciais.

Ambos seguem a mesma interface (fit / predict / evaluate) para
intercambialidade fácil no pipeline principal.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Interface Base (Strategy Pattern)                                        #
# ======================================================================= #

class BaseModel(ABC):
    """Interface abstrata para todos os modelos de IA."""

    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepara features (X) e targets (y) a partir do DataFrame."""
        pass

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Treina o modelo."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna previsões: 1 = Alta, 0 = Baixa."""
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Avalia o modelo e retorna métricas."""
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        return {"classification_report": report, "confusion_matrix": cm.tolist()}


# ======================================================================= #
#  Modelo 1: Random Forest (Baseline Recomendado)                          #
# ======================================================================= #

class RandomForestModel(BaseModel):
    """
    Classificador Random Forest para prever a direção do próximo candle.

    Vantagens:
        - Robusto a overfitting
        - Feature importance embutida
        - Treina rapidamente
        - Altamente interpretável

    Target: 1 se close[t+1] > close[t], senão 0.
    """

    FEATURE_COLUMNS = [
        "rsi_14",
        "macd_line", "macd_signal", "macd_histogram",
        "bb_width",
        "atr_14",
        "sma_20", "sma_50",
        "ema_9", "ema_21",
        # Features de momentum construídas
        "return_1", "return_3", "return_7",
        "volume_ratio",
    ]

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            min_samples_split=20,
            class_weight="balanced",  # Lida com desbalanceamento de classes
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Engenharia de features: adiciona retornos percentuais e ratio de volume.

        Args:
            df: DataFrame com indicadores técnicos calculados.

        Returns:
            Tupla (X, y) prontos para treinamento.
        """
        data = df.copy()

        # Features de retorno percentual
        data["return_1"] = data["close"].pct_change(1)
        data["return_3"] = data["close"].pct_change(3)
        data["return_7"] = data["close"].pct_change(7)

        # Volume relativo à média (anomalia de volume)
        data["volume_ratio"] = data["volume"] / data["volume"].rolling(20).mean()

        # Target: 1 se o preço SUBIU no próximo candle
        data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)

        data.dropna(inplace=True)

        # Seleciona apenas as features disponíveis no DataFrame
        available = [f for f in self.FEATURE_COLUMNS if f in data.columns]
        self.feature_names = available

        X = data[available].values
        y = data["target"].values
        return X, y

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Normaliza e treina o Random Forest."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True

        # Log de importância das features
        importances = pd.Series(
            self.model.feature_importances_, index=self.feature_names
        ).sort_values(ascending=False)
        logger.info(f"🌲 Top 5 features mais importantes:\n{importances.head()}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna previsão binária (0 ou 1)."""
        if not self.is_fitted:
            raise RuntimeError("Modelo não foi treinado. Execute fit() primeiro.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna a probabilidade de cada classe (útil para sizing de posição)."""
        if not self.is_fitted:
            raise RuntimeError("Modelo não foi treinado. Execute fit() primeiro.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, path: str) -> None:
        """Serializa o modelo e o scaler."""
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)
        logger.info(f"💾 Modelo salvo em: {path}")

    def load(self, path: str) -> None:
        """Carrega modelo e scaler serializados."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_fitted = True
        logger.info(f"📂 Modelo carregado de: {path}")

    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
        """
        Validação cruzada respeitando a ordem temporal (TimeSeriesSplit).
        Evita look-ahead bias — erro clássico em backtesting.

        Args:
            X, y: Features e targets completos.
            n_splits: Número de folds temporais.

        Returns:
            Dicionário com acurácias por fold.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            X_tr_scaled = self.scaler.fit_transform(X_tr)
            X_val_scaled = self.scaler.transform(X_val)

            self.model.fit(X_tr_scaled, y_tr)
            score = self.model.score(X_val_scaled, y_val)
            scores.append(score)
            logger.info(f"Fold {fold}: Acurácia = {score:.4f}")

        mean_acc = np.mean(scores)
        logger.info(f"✅ Acurácia média (CV): {mean_acc:.4f} ± {np.std(scores):.4f}")
        return {"scores": scores, "mean": mean_acc, "std": np.std(scores)}


# ======================================================================= #
#  Modelo 2: LSTM (Deep Learning — Opcional/Avançado)                      #
# ======================================================================= #

class LSTMModel(BaseModel):
    """
    Rede LSTM para capturar dependências temporais sequenciais.

    Requer TensorFlow/Keras instalado.
    Use quando o padrão de mercado tem memória temporal longa.

    Nota: Requer mais dados (~2000+ candles) e tempo de treino maior.
    """

    def __init__(self, lookback: int = 60, epochs: int = 50, batch_size: int = 32):
        """
        Args:
            lookback: Janela de candles passados como input da sequência.
            epochs: Épocas de treinamento.
            batch_size: Tamanho do batch.
        """
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = StandardScaler()

    def _build_model(self, input_shape: tuple) -> None:
        """Constrói a arquitetura LSTM."""
        try:
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.layers import (
                LSTM, BatchNormalization, Dense, Dropout, Input
            )
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError(
                "TensorFlow não encontrado. Instale com: pip install tensorflow"
            )

        self.model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(128, return_sequences=True),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(64, return_sequences=False),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation="relu"),
                Dense(1, activation="sigmoid"),  # Saída: prob. de alta
            ]
        )
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(f"🧠 Arquitetura LSTM:\n{self.model.summary()}")

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Converte features planas em sequências 3D para o LSTM."""
        Xs, ys = [], []
        for i in range(self.lookback, len(X)):
            Xs.append(X[i - self.lookback : i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Usa subconjunto de features numéricas para o LSTM."""
        data = df.copy()
        data["target"] = (data["close"].shift(-1) > data["close"]).astype(int)
        data.dropna(inplace=True)

        feature_cols = [c for c in data.columns if c != "target"]
        X = self.scaler.fit_transform(data[feature_cols].values)
        y = data["target"].values
        return self._create_sequences(X, y)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Treina o LSTM com early stopping para evitar overfitting."""
        from tensorflow.keras.callbacks import EarlyStopping

        self._build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Converte probabilidades para classificação binária (threshold=0.5)."""
        proba = self.model.predict(X, verbose=0)
        return (proba > 0.5).astype(int).flatten()


# --- Execução standalone para testes rápidos ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Gera dados sintéticos para demonstração
    np.random.seed(42)
    n = 500
    mock_features = {
        "rsi_14": np.random.uniform(20, 80, n),
        "macd_line": np.random.randn(n),
        "macd_signal": np.random.randn(n),
        "macd_histogram": np.random.randn(n),
        "bb_width": np.random.uniform(0.01, 0.1, n),
        "atr_14": np.random.uniform(100, 500, n),
        "sma_20": np.random.uniform(40000, 45000, n),
        "sma_50": np.random.uniform(39000, 44000, n),
        "ema_9": np.random.uniform(40000, 45000, n),
        "ema_21": np.random.uniform(39500, 44500, n),
        "close": np.cumsum(np.random.randn(n) * 100) + 40000,
        "volume": np.random.uniform(100, 1000, n),
    }
    df = pd.DataFrame(mock_features)

    rf = RandomForestModel()
    X, y = rf.prepare_features(df)

    # Split temporal (nunca use shuffle=True em séries temporais!)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    rf.fit(X_train, y_train)
    metrics = rf.evaluate(X_test, y_test)
    print("\n📊 Métricas do modelo:", metrics["classification_report"]["accuracy"])
