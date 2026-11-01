"""Synthetic checks for the INN-52 vial identification helpers. They test the logic, not the video."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from vial_identify import estimate_fractions, fit_mixture, rank_recipes, simplex_grid


def test_mixture_fit_recovers_per_tank_features_and_identifies_an_unseen_mix():
    W_true = np.array([[0.0, 0.0, 0.0], [0.1, 0.6, 1.2], [1.0, 0.3, 0.1]])   # any three "dyes"
    train = np.array([[1, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.7, 0.3], [0.2, 0.4, 0.4]])
    W = fit_mixture(train, train @ W_true)
    assert np.allclose(W, W_true)
    menu = {"a": np.array([0.6, 0.2, 0.2]), "b": np.array([0.2, 0.2, 0.6]), "c": np.array([0.2, 0.6, 0.2])}
    feat = menu["b"] @ W_true
    scale = np.ones(3)
    assert rank_recipes(feat, menu, W, scale)[0] == "b"
    assert np.allclose(estimate_fractions(feat, W, scale, simplex_grid()), menu["b"])


def test_simplex_grid_rows_sum_to_one():
    g = simplex_grid(4)
    assert len(g) == 15
    assert np.allclose(g.sum(axis=1), 1)
