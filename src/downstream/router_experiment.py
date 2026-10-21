"""Train the small sentence router on Injongo dev and evaluate frozen test rows."""

from __future__ import annotations

import argparse
import gc
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.downstream.injongo import load_split
from src.downstream.laya_model import LayaIntentClassifier
from src.downstream.metrics import paired_category
from src.downstream.noise_v2 import (
    M1_SEVERITIES,
    build_m1_severity_records,
    build_m5_records,
    build_m6_records,
)
from src.downstream.normalization_router import (
    ROUTER_FEATURES,
    RouterArtifact,
    artifact_from_sklearn_pipeline,
    deployable_features,
)
from src.downstream.routing_baselines import RoutingEvaluation, evaluate_decisions
from src.downstream.routing_features import TntSignalExtractor, load_formal_vocabulary
from src.downstream.v2_protocol import (
    INJONGO_TEST,
    LAYA_CHECKPOINT,
    TNT_CHECKPOINT,
    current_git_commit,
    sha256_file,
    validate_frozen_artifacts,
)


SEED = 2026
DEV_PATH = Path("data/external/injongo_intent_wol/dev.jsonl")
AUDIT_TEST_DATASET = Path(
    "results/downstream/normalization_router_audit_v1/routing_dataset.csv"
)
FORMAL_VOCABULARY_PATH = Path("data/vocabulary.txt")
RESULTS_DIR = Path("results/downstream/normalization_router_prototype_v1")
ARTIFACT_PATH = RESULTS_DIR / "router_artifact.json"


def _condition_records(rows: Sequence[Mapping], root: Path) -> list[dict]:
    conditions: list[tuple[str, list[dict]]] = []
    conditions.append(
        (
            "REAL",
            [
                {
                    "example_id": row["example_id"],
                    "row_id": row["row_id"],
                    "intent": row["intent"],
                    "real_text": row["text"],
                    "noisy_text": row["text"],
                    "noise_method": "none",
                    "operation_count": 0,
                    "split": row["split"],
                }
                for row in rows
            ],
        )
    )
    for severity in M1_SEVERITIES:
        conditions.append(
            (
                f"M1_{severity.name.upper()}",
                build_m1_severity_records(rows, severity, seed=SEED),
            )
        )
    conditions.append(
        (
            "M5",
            build_m5_records(
                rows,
                mappings_path=root / "artifacts/m5_youtube_variant_mappings.csv",
                seed=SEED,
            ),
        )
    )
    conditions.append(
        (
            "M6",
            build_m6_records(
                rows,
                artifacts_dir=root / "artifacts",
                annotations_dir=root / "data/annotations",
                seed=SEED,
            ),
        )
    )
    output = []
    for condition, records in conditions:
        for row in records:
            output.append(
                {
                    "condition": condition,
                    "example_id": str(row["example_id"]),
                    "row_id": str(row["row_id"]),
                    "routing_group_id": str(row["row_id"]),
                    "gold_intent": str(row["intent"]),
                    "real_text": str(row["real_text"]),
                    "input_text": str(row["noisy_text"]),
                    "noise_method": str(row["noise_method"]),
                    "synthetic_operation_count": int(row["operation_count"]),
                }
            )
    return output


def build_dev_dataset(
    root: Path,
    *,
    device: str | None = None,
    tnt_batch_size: int = 128,
    laya_batch_size: int = 16,
) -> pd.DataFrame:
    dev_rows = load_split(root / DEV_PATH, "dev")
    rows = _condition_records(dev_rows, root)
    texts = [row["input_text"] for row in rows]

    print(f"[router] TNT candidates/signals for {len(rows):,} dev-condition rows", flush=True)
    tnt = TntSignalExtractor(root / TNT_CHECKPOINT, device=device, batch_size=tnt_batch_size)
    normalized = tnt.normalize(texts)
    signals = tnt.extract(texts)
    del tnt
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    print(f"[router] Laya before/after predictions for {len(rows):,} rows", flush=True)
    classifier = LayaIntentClassifier(
        root / LAYA_CHECKPOINT, device=device, batch_size=laya_batch_size
    )
    before_inputs = [
        {
            "example_id": row["example_id"],
            "row_id": row["row_id"],
            "text": row["input_text"],
            "intent": row["gold_intent"],
            "split": "dev",
        }
        for row in rows
    ]
    after_inputs = [dict(item, text=candidate) for item, candidate in zip(before_inputs, normalized)]
    before = classifier.predict(before_inputs)
    after = classifier.predict(after_inputs)
    vocabulary = load_formal_vocabulary(root / FORMAL_VOCABULARY_PATH)

    output = []
    for row, candidate, signal, before_prediction, after_prediction in zip(
        rows, normalized, signals, before, after
    ):
        category = paired_category(
            row["gold_intent"],
            before_prediction["prediction"],
            after_prediction["prediction"],
        )
        label = {
            "HELPED": "HELPED",
            "HARMED": "HARMED",
            "UNCHANGED_CORRECT": "NEUTRAL_CORRECT",
            "UNRESOLVED": "NEUTRAL_WRONG",
        }[category]
        features = deployable_features(
            row["input_text"], candidate, signal, vocabulary
        )
        output.append(
            {
                **row,
                "normalized_text": candidate,
                "laya_prediction_before": before_prediction["prediction"],
                "laya_prediction_after": after_prediction["prediction"],
                "laya_confidence_before": before_prediction["confidence"],
                "laya_confidence_after": after_prediction["confidence"],
                "gold_confidence_before": before_prediction["gold_confidence"],
                "gold_confidence_after": after_prediction["gold_confidence"],
                "correct_before": before_prediction["prediction"] == row["gold_intent"],
                "correct_after": after_prediction["prediction"] == row["gold_intent"],
                "routing_label": label,
                "normalize_beneficial": int(label == "HELPED"),
                **features,
            }
        )
    return pd.DataFrame(output)


def _make_model(seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=2000, random_state=seed
                ),
            ),
        ]
    )


def _select_conservative_threshold(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    max_clean_accuracy_drop: float = 0.005,
) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(
        np.concatenate(
            ([0.0], np.quantile(probabilities, np.linspace(0.0, 1.0, 401)), [1.0 + 1e-12])
        )
    )
    before = frame["correct_before"].astype(bool).to_numpy()
    after = frame["correct_after"].astype(bool).to_numpy()
    clean = frame["condition"].eq("REAL").to_numpy()
    clean_baseline = float(before[clean].mean())
    rows = []
    for threshold in candidates:
        decisions = probabilities >= threshold
        correctness = np.where(decisions, after, before)
        clean_accuracy = float(correctness[clean].mean())
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(correctness.mean()),
                "clean_accuracy": clean_accuracy,
                "clean_accuracy_drop": clean_baseline - clean_accuracy,
                "normalization_rate": float(decisions.mean()),
                "eligible": clean_baseline - clean_accuracy <= max_clean_accuracy_drop + 1e-12,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["eligible"]]
    if eligible.empty:
        raise RuntimeError("No threshold satisfies the clean-preservation constraint")
    selected = eligible.sort_values(
        ["accuracy", "normalization_rate", "threshold"],
        ascending=[False, True, False],
    ).iloc[0]
    return float(selected["threshold"]), table


def train_router(frame: pd.DataFrame) -> tuple[RouterArtifact, np.ndarray, pd.DataFrame]:
    from sklearn.model_selection import GroupKFold

    X = frame[list(ROUTER_FEATURES)].astype(float)
    y = frame["normalize_beneficial"].astype(int).to_numpy()
    groups = frame["routing_group_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros(len(frame), dtype=float)
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(X, y, groups)
    ):
        model = _make_model(SEED + fold)
        model.fit(X.iloc[train_indices], y[train_indices])
        oof[validation_indices] = model.predict_proba(X.iloc[validation_indices])[:, 1]
    threshold, threshold_table = _select_conservative_threshold(frame, oof)
    final_model = _make_model(SEED)
    final_model.fit(X, y)
    artifact = artifact_from_sklearn_pipeline(
        final_model,
        threshold=threshold,
        metadata={
            "training_split": "Injongo official dev only",
            "training_rows": len(frame),
            "training_groups": frame["routing_group_id"].nunique(),
            "seed": SEED,
            "target": "HELPED versus all other outcomes",
            "threshold_selection": "5-fold grouped OOF downstream accuracy with <=0.5pp clean drop",
        },
    )
    return artifact, oof, threshold_table


def _evaluation_table(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    *,
    prefix: str,
) -> pd.DataFrame:
    category = frame["routing_label"].astype(str).to_numpy()
    evaluations = [
        RoutingEvaluation("always_skip", "none", np.zeros(len(frame), dtype=bool)),
        RoutingEvaluation("always_normalize", "none", np.ones(len(frame), dtype=bool)),
        RoutingEvaluation(f"{prefix}_router", "deployable_all", probabilities >= threshold),
        RoutingEvaluation("oracle_help_only", "oracle", category == "HELPED"),
    ]
    rows = []
    for evaluation in evaluations:
        combined = evaluate_decisions(frame, evaluation)
        combined["condition"] = "ALL"
        rows.append(combined)
        for condition, indices in frame.groupby("condition", sort=False).groups.items():
            index_array = np.asarray(list(indices), dtype=int)
            subset = frame.loc[indices].reset_index(drop=True)
            local = RoutingEvaluation(
                evaluation.strategy,
                evaluation.feature_set,
                evaluation.decisions[index_array],
            )
            record = evaluate_decisions(subset, local)
            record["condition"] = condition
            rows.append(record)
    return pd.DataFrame(rows)


def _write_json(path: Path, payload) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value) if math.isfinite(float(value)) else None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_report_and_figures(
    output: Path,
    artifact: RouterArtifact,
    dev_results: pd.DataFrame,
    test_results: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    coefficients = pd.DataFrame(
        {
            "feature": artifact.feature_names,
            "standardized_coefficient": artifact.coefficients,
            "absolute_coefficient": np.abs(artifact.coefficients),
            "direction": [
                "toward_normalize" if value > 0 else "toward_skip"
                for value in artifact.coefficients
            ],
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficients.to_csv(output / "router_coefficients.csv", index=False)

    dev_all = dev_results[dev_results["condition"] == "ALL"].set_index("strategy")
    test_all = test_results[test_results["condition"] == "ALL"].set_index("strategy")
    dev_router = dev_all.loc["dev_oof_router"]
    test_router = test_all.loc["dev_trained_router"]
    test_conditions = test_results[
        test_results["strategy"].isin(
            ["always_skip", "always_normalize", "dev_trained_router"]
        )
        & test_results["condition"].ne("ALL")
    ].pivot(index="condition", columns="strategy", values="accuracy")
    order = ["REAL", "M1_LIGHT", "M1_MEDIUM", "M1_HEAVY", "M5", "M6"]
    test_conditions = test_conditions.loc[order]

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    width = 0.25
    for offset, strategy in enumerate(
        ["always_skip", "always_normalize", "dev_trained_router"]
    ):
        ax.bar(
            x + (offset - 1) * width,
            test_conditions[strategy] * 100,
            width,
            label=strategy.replace("dev_trained_", ""),
        )
    ax.set_xticks(x, order)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Intent accuracy (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "router_accuracy_by_condition.png", dpi=180)
    plt.close(fig)

    top = coefficients.head(12).sort_values("standardized_coefficient")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#9b3d3d" if value < 0 else "#315b7d" for value in top["standardized_coefficient"]]
    ax.barh(top["feature"], top["standardized_coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Standardized logistic coefficient")
    ax.set_title("Descriptive coefficients; correlated features limit causal interpretation")
    fig.tight_layout()
    fig.savefig(figure_dir / "router_coefficients.png", dpi=180)
    plt.close(fig)

    report = f"""# Sentence-level normalization router prototype V1

## Protocol

The router is a regularized logistic regression trained only on the 320-row
official Injongo dev split expanded into six conditions (1,920 paired rows).
TNT and Laya remain frozen. Five-fold grouped out-of-fold predictions select a
threshold that maximizes downstream correctness while allowing at most a 0.5
percentage-point loss on clean dev inputs. All variants of one utterance share
the same `row_id` group.

The exported threshold is **{artifact.threshold:.6f}**. Features contain only
surface, TNT-head and input-to-TNT-candidate measurements. No REAL reference,
synthetic operation count, noise-method label, gold confidence or intent label
is available to the router at inference.

## Development result (grouped out-of-fold)

| Strategy | Accuracy | Macro-F1 | Normalized |
|---|---:|---:|---:|
| Always skip | {dev_all.loc['always_skip','accuracy']:.2%} | {dev_all.loc['always_skip','macro_f1']:.2%} | 0% |
| Always TNT | {dev_all.loc['always_normalize','accuracy']:.2%} | {dev_all.loc['always_normalize','macro_f1']:.2%} | 100% |
| Router | **{dev_router['accuracy']:.2%}** | **{dev_router['macro_f1']:.2%}** | {dev_router['normalization_rate']:.2%} |
| Oracle | {dev_all.loc['oracle_help_only','accuracy']:.2%} | {dev_all.loc['oracle_help_only','macro_f1']:.2%} | {dev_all.loc['oracle_help_only','normalization_rate']:.2%} |

The router gains {dev_router['accuracy'] - dev_all.loc['always_skip','accuracy']:+.2%}
over skip and {dev_router['accuracy'] - dev_all.loc['always_normalize','accuracy']:+.2%}
over systematic normalization on its leakage-controlled development estimate.

## Exploratory test result

| Strategy | Accuracy | Macro-F1 | Normalized |
|---|---:|---:|---:|
| Always skip | {test_all.loc['always_skip','accuracy']:.2%} | {test_all.loc['always_skip','macro_f1']:.2%} | 0% |
| Always TNT | {test_all.loc['always_normalize','accuracy']:.2%} | {test_all.loc['always_normalize','macro_f1']:.2%} | 100% |
| Dev-trained router | **{test_router['accuracy']:.2%}** | **{test_router['macro_f1']:.2%}** | {test_router['normalization_rate']:.2%} |
| Oracle | {test_all.loc['oracle_help_only','accuracy']:.2%} | {test_all.loc['oracle_help_only','macro_f1']:.2%} | {test_all.loc['oracle_help_only','normalization_rate']:.2%} |

{test_conditions.reset_index().to_markdown(index=False)}

On test, the router accepts {int(test_router['number_normalized'])}/3,828
candidates, captures {int(test_router['helped_routed_to_tnt'])}/219 HELPED cases,
and accepts {int(test_router['harmed_routed_to_tnt'])}/192 HARMED cases. It
preserves the exact always-skip accuracy on REAL, M5 and M6; its pooled gain is
driven by M1, especially heavy noise.

## Interpretation and limitation

The conservative gate works as intended: it rejects most TNT candidates and
retains a substantial fraction of the M1 benefit without reproducing the clean
and cross-method degradation of always-TNT. It does not solve distribution
generalization: M5/M6 are preserved mostly because the router abstains.

The test result is **exploratory, not a pristine final estimate**. The previous
audit inspected this test and influenced the decision to use a logistic router.
A new human-noisy, intent-labelled holdout is required before a deployment or
generalization claim.

The coefficient table is descriptive only. Several features are mathematically
or empirically correlated (`P(KEEP)` and `P(change)`, CER and chrF), so individual
coefficient magnitudes must not be given a causal interpretation.
"""
    (output / "router_report.md").write_text(report, encoding="utf-8")


def run(
    repository_root: str | Path = ".",
    *,
    device: str | None = None,
    rebuild_dev: bool = False,
) -> dict:
    root = Path(repository_root).resolve()
    output = root / RESULTS_DIR
    output.mkdir(parents=True, exist_ok=True)
    frozen = validate_frozen_artifacts(root)
    dev_path = output / "dev_routing_dataset.csv"
    if rebuild_dev or not dev_path.is_file():
        dev = build_dev_dataset(root, device=device)
        dev.to_csv(dev_path, index=False)
    else:
        dev = pd.read_csv(dev_path, keep_default_na=False)

    artifact, oof_probabilities, threshold_table = train_router(dev)
    artifact = RouterArtifact(
        **{
            **artifact.__dict__,
            "metadata": {
                **artifact.metadata,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": current_git_commit(root),
                "frozen_laya_sha256": frozen["laya_model_sha256"],
                "frozen_tnt_sha256": frozen["tnt_model_sha256"],
                "formal_vocabulary_sha256": sha256_file(root / FORMAL_VOCABULARY_PATH),
            },
        }
    )
    artifact.save(output / "router_artifact.json")
    threshold_table.to_csv(output / "dev_threshold_sweep.csv", index=False)
    dev_predictions = dev[
        ["example_id", "row_id", "condition", "gold_intent", "routing_label"]
    ].copy()
    dev_predictions["routing_score_oof"] = oof_probabilities
    dev_predictions["router_decision"] = np.where(
        oof_probabilities >= artifact.threshold, "NORMALIZE", "SKIP"
    )
    dev_predictions.to_csv(output / "dev_router_predictions.csv", index=False)
    dev_results = _evaluation_table(
        dev.reset_index(drop=True), oof_probabilities, artifact.threshold, prefix="dev_oof"
    )
    dev_results.to_csv(output / "dev_router_results.csv", index=False)

    test = pd.read_csv(root / AUDIT_TEST_DATASET, keep_default_na=False).reset_index(drop=True)
    test_probabilities = artifact.score(test.to_dict(orient="records"))
    test_predictions = test[
        ["example_id", "row_id", "condition", "gold_intent", "routing_label"]
    ].copy()
    test_predictions["routing_score"] = test_probabilities
    test_predictions["router_decision"] = np.where(
        test_probabilities >= artifact.threshold, "NORMALIZE", "SKIP"
    )
    test_predictions.to_csv(output / "test_router_predictions.csv", index=False)
    test_results = _evaluation_table(
        test, test_probabilities, artifact.threshold, prefix="dev_trained"
    )
    test_results.to_csv(output / "test_router_results.csv", index=False)

    _write_report_and_figures(output, artifact, dev_results, test_results)

    dev_all = dev_results[dev_results["condition"] == "ALL"].set_index("strategy")
    test_all = test_results[test_results["condition"] == "ALL"].set_index("strategy")
    summary = {
        "threshold": artifact.threshold,
        "dev_oof": dev_all.reset_index().to_dict(orient="records"),
        "test_exploratory": test_all.reset_index().to_dict(orient="records"),
        "test_is_not_pristine": True,
        "reason": "The official test outcomes were already used in the preceding feasibility audit and architecture choice.",
        "recommendation": "Validate the frozen router on a new human-noisy intent-labelled holdout before deployment claims.",
    }
    _write_json(output / "router_summary.json", summary)
    _write_json(
        output / "experiment_manifest.json",
        {
            "experiment": "normalization_router_prototype_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_git_commit(root),
            "seed": SEED,
            "frozen_artifacts": frozen,
            "dev_dataset_sha256": sha256_file(root / DEV_PATH),
            "test_routing_dataset_sha256": sha256_file(root / AUDIT_TEST_DATASET),
            "router_artifact_sha256": sha256_file(output / "router_artifact.json"),
            "no_laya_or_tnt_training": True,
            "router_training_only": True,
        },
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rebuild-dev", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    print(
        json.dumps(
            run(
                arguments.repository_root,
                device=arguments.device,
                rebuild_dev=arguments.rebuild_dev,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )

