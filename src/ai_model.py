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

    sample_index: pd.Index

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
        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        logger.info(
            "\n%s",
            classification_report(y_test, y_pred, zero_division=0),
        )
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
        "macd_line",
        "macd_signal",
        "macd_histogram",
        "bb_width",
        "atr_14",
        "sma_20",
        "sma_50",
        "ema_9",
        "ema_21",
        # Features de momentum construídas
        "return_1",
        "return_3",
        "return_7",
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
        self.sample_index = pd.Index([])
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

        missing = set(self.FEATURE_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Features obrigatórias ausentes: {sorted(missing)}")

        # O último candle não possui futuro conhecido e não pode receber classe 0.
        next_close = data["close"].shift(-1)
        data["target"] = (next_close > data["close"]).where(next_close.notna())
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=[*self.FEATURE_COLUMNS, "target"], inplace=True)

        self.feature_names = self.FEATURE_COLUMNS.copy()
        self.sample_index = data.index.copy()

        X = data[self.feature_names].to_numpy(dtype=float)
        y = data["target"].to_numpy(dtype=int)
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

    def predict_up_probability(self, X: np.ndarray) -> np.ndarray:
        """Retorna P(classe=1), inclusive quando o treino contém uma classe só."""
        probabilities = self.predict_proba(X)
        classes = self.model.classes_
        if 1 in classes:
            positive_column = int(np.flatnonzero(classes == 1)[0])
            return probabilities[:, positive_column]
        return np.zeros(len(X), dtype=float)

    def save(self, path: str) -> None:
        """Serializa o modelo e o scaler."""
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            },
            path,
        )
        logger.info(f"💾 Modelo salvo em: {path}")

    def load(self, path: str) -> None:
        """Carrega modelo e scaler serializados."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data.get("feature_names", [])
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
        self.feature_names: list[str] = []
        self.sample_index = pd.Index([])
        self.is_fitted = False

    def _build_model(self, input_shape: tuple) -> None:
        """Constrói a arquitetura LSTM."""
        try:
            from tensorflow.keras.layers import (
                LSTM,
                BatchNormalization,
                Dense,
                Dropout,
                Input,
            )
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.optimizers import Adam
        except ImportError as error:
            raise ImportError(
                "TensorFlow não encontrado. Instale com: pip install tensorflow"
            ) from error

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
        for end in range(self.lookback - 1, len(X)):
            start = end - self.lookback + 1
            Xs.append(X[start : end + 1])
            ys.append(y[end])
        return np.array(Xs), np.array(ys)

    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Usa subconjunto de features numéricas para o LSTM."""
        data = df.copy()
        next_close = data["close"].shift(-1)
        data["target"] = (next_close > data["close"]).where(next_close.notna())
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        self.feature_names = [c for c in data.columns if c != "target"]
        X = data[self.feature_names].to_numpy(dtype=float)
        y = data["target"].to_numpy(dtype=int)
        X_sequences, y_sequences = self._create_sequences(X, y)
        self.sample_index = data.index[self.lookback - 1 :].copy()
        return X_sequences, y_sequences

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Treina o LSTM com early stopping para evitar overfitting."""
        from tensorflow.keras.callbacks import EarlyStopping

        if X_train.ndim != 3:
            raise ValueError(
                "O LSTM espera features no formato [amostras, tempo, atributos]."
            )

        n_samples, lookback, n_features = X_train.shape
        self.scaler.fit(X_train.reshape(-1, n_features))
        X_train_scaled = self.scaler.transform(X_train.reshape(-1, n_features)).reshape(
            n_samples, lookback, n_features
        )

        self._build_model(input_shape=(lookback, n_features))
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        self.model.fit(
            X_train_scaled,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1,
        )
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Converte probabilidades para classificação binária (threshold=0.5)."""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna a probabilidade de alta para cada sequência."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Modelo não foi treinado. Execute fit() primeiro.")
        n_samples, lookback, n_features = X.shape
        X_scaled = self.scaler.transform(X.reshape(-1, n_features)).reshape(
            n_samples, lookback, n_features
        )
        return self.model.predict(X_scaled, verbose=0).flatten()


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
