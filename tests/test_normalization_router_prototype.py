import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.downstream.normalization_router import (
    ROUTER_FEATURES,
    RouterArtifact,
    artifact_from_sklearn_pipeline,
)
from src.downstream.router_experiment import (
    _make_model,
    _select_conservative_threshold,
)


def test_exported_router_matches_sklearn_pipeline(tmp_path: Path):
    rng = np.random.default_rng(2026)
    matrix = rng.normal(size=(80, len(ROUTER_FEATURES)))
    labels = (matrix[:, 0] + 0.5 * matrix[:, 1] > 0).astype(int)
    frame = pd.DataFrame(matrix, columns=ROUTER_FEATURES)
    model = _make_model(2026)
    model.fit(frame, labels)
    artifact = artifact_from_sklearn_pipeline(
        model, threshold=0.61, metadata={"test": True}
    )
    path = tmp_path / "router.json"
    artifact.save(path)
    loaded = RouterArtifact.load(path)
    expected = model.predict_proba(frame)[:, 1]
    observed = loaded.score(frame.to_dict(orient="records"))
    np.testing.assert_allclose(observed, expected, atol=1e-10)
    assert loaded.threshold == 0.61
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "wolof_tnt_sentence_router_v1"


def test_threshold_selection_respects_clean_harm_constraint():
    frame = pd.DataFrame(
        {
            "condition": ["REAL"] * 4 + ["M1_HEAVY"] * 4,
            "correct_before": [1, 1, 1, 1, 0, 0, 0, 1],
            "correct_after": [0, 1, 1, 1, 1, 1, 0, 0],
        }
    )
    probabilities = np.asarray([0.9, 0.1, 0.1, 0.1, 0.95, 0.8, 0.2, 0.7])
    threshold, sweep = _select_conservative_threshold(
        frame, probabilities, max_clean_accuracy_drop=0.0
    )
    selected = sweep.loc[np.isclose(sweep["threshold"], threshold)].iloc[0]
    assert bool(selected["eligible"])
    assert selected["clean_accuracy_drop"] <= 0.0
    decisions = probabilities >= threshold
    assert not decisions[0]


def test_router_feature_contract_contains_no_oracle_fields():
    assert ROUTER_FEATURES
    assert all(not name.startswith("oracle_") for name in ROUTER_FEATURES)
    assert "synthetic_operation_count" not in ROUTER_FEATURES
    assert "gold_confidence_before" not in ROUTER_FEATURES

