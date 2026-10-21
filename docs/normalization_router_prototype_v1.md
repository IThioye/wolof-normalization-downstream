# Sentence-level normalization router prototype V1

The prototype is an acceptance gate for the complete TNT candidate. It does
not alter or retrain TNT or Laya.

## Train and evaluate

The first run generates the six frozen conditions for Injongo dev, runs TNT
and Laya, fits the router, and performs the explicitly exploratory test:

```powershell
python -m src.downstream.router_experiment --device cuda --rebuild-dev
```

Later runs reuse the cached dev paired outputs:

```powershell
python -m src.downstream.router_experiment
```

The JSON artifact contains the imputer, scaler, logistic coefficients and the
conservative threshold. It does not require scikit-learn for inference.

## Use the exported gate

```python
from src.downstream.normalization_router import FrozenSentenceNormalizationRouter
from src.downstream.v2_protocol import TNT_CHECKPOINT

router = FrozenSentenceNormalizationRouter(
    "results/downstream/normalization_router_prototype_v1/router_artifact.json",
    TNT_CHECKPOINT,
    "data/vocabulary.txt",
    device="cuda",
)

result = router.route(["dama bg dem dakar suba"])[0]
print(result["decision"], result["selected_text"], result["routing_score"])
```

The gate runs TNT to obtain both its candidate and internal scores. It reduces
overcorrection but does not save TNT inference compute.

## Scientific status

The router is fitted on official dev only. Its test result is exploratory
because the preceding feasibility audit already inspected the official test.
Before deployment claims, freeze this artifact and evaluate it on a new
human-noisy intent-labelled holdout.
