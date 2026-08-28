import numpy as np
import pandas as pd
import pytest

from src.ai_model import LSTMModel, RandomForestModel


def test_random_forest_target_excludes_unknown_future(feature_df: pd.DataFrame) -> None:
    model = RandomForestModel(n_estimators=10)
    X, y = model.prepare_features(feature_df)

    assert len(X) == len(y) == len(model.sample_index)
    assert model.sample_index[-1] < feature_df.index[-1]
    expected = (
        feature_df.loc[model.sample_index, "close"].shift(-1)
        > feature_df.loc[model.sample_index, "close"]
    )
    np.testing.assert_array_equal(y[:-1], expected.iloc[:-1].astype(int))


def test_random_forest_rejects_missing_features(feature_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Features obrigatórias ausentes"):
        RandomForestModel().prepare_features(feature_df.drop(columns="rsi_14"))


def test_random_forest_trains_predicts_and_persists(
    feature_df: pd.DataFrame, tmp_path
) -> None:
    model = RandomForestModel(n_estimators=20)
    X, y = model.prepare_features(feature_df)
    split = int(len(X) * 0.8)
    model.fit(X[:split], y[:split])

    predictions = model.predict(X[split:])
    probabilities = model.predict_proba(X[split:])
    up_probability = model.predict_up_probability(X[split:])
    assert set(np.unique(predictions)) <= {0, 1}
    assert probabilities.shape == (len(predictions), 2)
    np.testing.assert_allclose(up_probability, probabilities[:, 1])

    model_path = tmp_path / "model.pkl"
    model.save(str(model_path))
    restored = RandomForestModel(n_estimators=1)
    restored.load(str(model_path))
    np.testing.assert_array_equal(predictions, restored.predict(X[split:]))
    assert restored.feature_names == model.feature_names


def test_lstm_preparation_preserves_sequence_timestamps(
    feature_df: pd.DataFrame,
) -> None:
    model = LSTMModel(lookback=10, epochs=1)
    X, y = model.prepare_features(feature_df)

    assert X.ndim == 3
    assert X.shape[1] == 10
    assert len(X) == len(y) == len(model.sample_index)
    assert model.sample_index[-1] < feature_df.index[-1]
