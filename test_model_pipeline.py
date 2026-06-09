"""Tests unitaires minimaux du pipeline ML."""

import numpy as np
from model_pipeline import train_model, evaluate_model


def test_train_returns_fitted_model():
    """Le modele entraine doit savoir predire."""
    x = np.random.rand(50, 4)
    y = np.random.randint(0, 2, 50)
    model = train_model(x, y, n_estimators=10)
    assert hasattr(model, "predict")


def test_evaluate_returns_valid_accuracy():
    """L'accuracy doit etre comprise entre 0 et 1."""
    x = np.random.rand(40, 4)
    y = np.random.randint(0, 2, 40)
    model = train_model(x, y, n_estimators=10)
    acc = evaluate_model(model, x, y)
    assert 0.0 <= acc <= 1.0
