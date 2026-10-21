"""Serializable sentence-level acceptance gate for frozen TNT candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.downstream.routing_features import (
    DEPLOYABLE_SURFACE_FEATURES,
    DEPLOYABLE_TNT_FEATURES,
    TntSignalExtractor,
    load_formal_vocabulary,
    surface_features,
)
from src.downstream.text_error_analysis import pair_text_metrics


ROUTER_FEATURES = tuple(DEPLOYABLE_SURFACE_FEATURES + DEPLOYABLE_TNT_FEATURES)


@dataclass(frozen=True)
class RouterArtifact:
    feature_names: tuple[str, ...]
    imputer_statistics: tuple[float, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    threshold: float
    metadata: dict

    @classmethod
    def load(cls, path: str | Path) -> "RouterArtifact":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(raw["feature_names"]),
            imputer_statistics=tuple(map(float, raw["imputer_statistics"])),
            scaler_mean=tuple(map(float, raw["scaler_mean"])),
            scaler_scale=tuple(map(float, raw["scaler_scale"])),
            coefficients=tuple(map(float, raw["coefficients"])),
            intercept=float(raw["intercept"]),
            threshold=float(raw["threshold"]),
            metadata=dict(raw.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "format": "wolof_tnt_sentence_router_v1",
                    "feature_names": list(self.feature_names),
                    "imputer_statistics": list(self.imputer_statistics),
                    "scaler_mean": list(self.scaler_mean),
                    "scaler_scale": list(self.scaler_scale),
                    "coefficients": list(self.coefficients),
                    "intercept": self.intercept,
                    "threshold": self.threshold,
                    "metadata": self.metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def score(self, rows: Sequence[Mapping]) -> np.ndarray:
        matrix = np.asarray(
            [[float(row.get(name, np.nan)) for name in self.feature_names] for row in rows],
            dtype=float,
        )
        imputer = np.asarray(self.imputer_statistics)
        matrix = np.where(np.isfinite(matrix), matrix, imputer)
        scale = np.asarray(self.scaler_scale)
        scale = np.where(scale == 0, 1.0, scale)
        standardized = (matrix - np.asarray(self.scaler_mean)) / scale
        logits = standardized @ np.asarray(self.coefficients) + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))

    def decisions(self, rows: Sequence[Mapping]) -> np.ndarray:
        return self.score(rows) >= self.threshold


def deployable_features(
    input_text: str,
    normalized_candidate: str,
    tnt_signals: Mapping[str, float],
    formal_vocabulary: set[str],
) -> dict[str, float]:
    row = dict(surface_features(input_text, formal_vocabulary))
    metrics = pair_text_metrics(input_text, normalized_candidate)
    row.update(
        {
            "tnt_output_char_edits": float(metrics["char_edits"]),
            "tnt_output_char_edit_rate": float(metrics["cer"]),
            "tnt_output_token_edits": float(metrics["word_edits"]),
            "tnt_output_token_edit_rate": float(metrics["wer"]),
            "tnt_output_chrf": float(metrics["chrf"]),
            "tnt_output_changed": float(not metrics["exact"]),
        }
    )
    row.update({key: float(value) for key, value in tnt_signals.items()})
    missing = set(ROUTER_FEATURES).difference(row)
    if missing:
        raise RuntimeError(f"Missing deployable router features: {sorted(missing)}")
    return row


class FrozenSentenceNormalizationRouter:
    """Run TNT, then accept or reject its complete sentence candidate."""

    def __init__(
        self,
        artifact_path: str | Path,
        tnt_checkpoint: str | Path,
        formal_vocabulary_path: str | Path,
        *,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.artifact = RouterArtifact.load(artifact_path)
        self.formal_vocabulary = load_formal_vocabulary(formal_vocabulary_path)
        self.tnt = TntSignalExtractor(
            tnt_checkpoint, device=device, batch_size=batch_size
        )

    def route(self, texts: Sequence[str]) -> list[dict]:
        texts = [str(text) for text in texts]
        candidates = self.tnt.normalize(texts)
        signals = self.tnt.extract(texts)
        feature_rows = [
            deployable_features(text, candidate, signal, self.formal_vocabulary)
            for text, candidate, signal in zip(texts, candidates, signals)
        ]
        scores = self.artifact.score(feature_rows)
        decisions = scores >= self.artifact.threshold
        return [
            {
                "input_text": text,
                "normalized_candidate": candidate,
                "routing_score": float(score),
                "threshold": self.artifact.threshold,
                "decision": "NORMALIZE" if decision else "SKIP",
                "selected_text": candidate if decision else text,
            }
            for text, candidate, score, decision in zip(
                texts, candidates, scores, decisions
            )
        ]


def artifact_from_sklearn_pipeline(
    pipeline,
    *,
    threshold: float,
    metadata: Mapping | None = None,
) -> RouterArtifact:
    imputer = pipeline.named_steps["impute"]
    scaler = pipeline.named_steps["scale"]
    model = pipeline.named_steps["model"]
    if model.coef_.shape != (1, len(ROUTER_FEATURES)):
        raise ValueError("Unexpected logistic-regression coefficient shape")
    return RouterArtifact(
        feature_names=ROUTER_FEATURES,
        imputer_statistics=tuple(map(float, imputer.statistics_)),
        scaler_mean=tuple(map(float, scaler.mean_)),
        scaler_scale=tuple(map(float, scaler.scale_)),
        coefficients=tuple(map(float, model.coef_[0])),
        intercept=float(model.intercept_[0]),
        threshold=float(threshold),
        metadata=dict(metadata or {}),
    )

