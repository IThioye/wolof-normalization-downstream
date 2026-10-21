# Wolof normalization for downstream intent classification

This repository studies whether text normalization recovers intent-classification
performance lost when standardized Wolof is transformed into synthetic informal
orthography.

The experiment uses one Laya classifier fine-tuned only on the official
`InjongoIntent/wol` training split. The frozen classifier is evaluated on three
paired forms of every official test example:

```text
REAL       original standardized Wolof
NOISY      the same text corrupted with a synthetic noise method
NORMALIZED the NOISY text processed by the frozen TNT Wolof normalizer
```

The fine-tuned classifier is available at
[`IbrahimaThioye/laya-injongo-wolof-intent`](https://huggingface.co/IbrahimaThioye/laya-injongo-wolof-intent).

## Main results

### V1: M1-heavy stress test

| Condition | Accuracy | Macro-F1 |
|---|---:|---:|
| REAL | 80.88% | 81.18% |
| NOISY M1-heavy | 48.59% | 45.97% |
| NORMALIZED M1-heavy | 60.34% | 60.87% |

Normalization recovered 11.76 accuracy points and 14.90 macro-F1 points, but
this is a distribution-matched synthetic stress test: TNT was itself pretrained
on M1-like noise.

### V2: severity and clean preservation

| Input | Before TNT | After TNT | Accuracy delta |
|---|---:|---:|---:|
| REAL/clean | 80.88% | 77.90% | -2.98 pp |
| M1-light | 74.29% | 72.26% | -2.04 pp |
| M1-medium | 63.95% | 66.77% | +2.82 pp |
| M1-heavy | 48.59% | 60.34% | +11.76 pp |

TNT is therefore not presented as a universal preprocessing step. It was useful
mainly under sufficiently severe, M1-like corruption and harmed clean or lightly
corrupted text.

## Frozen protocol

| Purpose | Official split | Rows |
|---|---|---:|
| Fine-tuning | `train.jsonl` | 2,240 |
| Selection and calibration | `dev.jsonl` | 320 |
| Final evaluation | `test.jsonl` | 638 |

No new random split was created. The Laya checkpoint never saw NOISY examples
during fine-tuning. The same checkpoint was used for REAL, NOISY and NORMALIZED.

Pinned sources:

- InjongoIntent repository revision: `1f8be590da6699aee3dc23de6f63e801e2352eff`
- Laya multilingual revision: `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`
- Laya API: `laya==0.3.21`
- experiment seed: `2026`

## Repository contents

```text
src/downstream/   data preparation, training, evaluation and routing analysis
src/generation/   frozen synthetic-noise generators used by the study
src/modeling/     TNT loading and shared text metrics
notebooks/        executable fine-tuning and evaluation notebooks
results/          aggregate metrics and figures without redistributed text
tests/            protocol and determinism checks
docs/             detailed V1/V2 and routing documentation
```

The repository intentionally does not redistribute Injongo utterances or the
text-level prediction exports. The loader downloads the pinned official files
from their upstream repository.

## Installation

Python 3.11 was used for the reported experiments.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Use the fine-tuned classifier

```python
import laya

INTENTS = (
    "alarm", "balance", "bill_balance", "book_flight", "book_hotel",
    "calendar_update", "cancel_reservation", "car_rental",
    "confirm_reservation", "cook_time", "exchange_rate", "food_last",
    "freeze_account", "ingredients_list", "interest_rate",
    "international_visa", "make_call", "meal_suggestion", "min_payment",
    "pay_bill", "pin_change", "play_music", "plug_type", "recipe",
    "restaurant_reservation", "restaurant_reviews", "restaurant_suggestion",
    "share_location", "shopping_list_update", "spending_history", "text",
    "time", "timezone", "transactions", "transfer", "translate",
    "travel_notification", "travel_suggestion", "update_playlist", "weather",
)

agent = laya.load("IbrahimaThioye/laya-injongo-wolof-intent")
questions = {
    "intent": {
        "type": "choice",
        "instructions": "Choose the intent that best describes the Wolof utterance.",
        "criteria": {intent: "" for intent in INTENTS},
    }
}
result = agent.predict({"utterance": "Ban waxtu la?"}, questions)
print(result["answers"]["intent"]["choice"])
```

Keep the same 40 options and instruction used during fine-tuning. This model is
an intent classifier; it does not contain TNT and does not normalize text by
itself.

## Reproducing the study

Start with the protocol checks:

```bash
python -m pytest tests/test_downstream_normalization.py -q
python -m pytest tests/test_downstream_analysis_v2.py -q
```

Then open:

1. `notebooks/laya_injongo_wolof_finetuning.ipynb`
2. `notebooks/laya_normalization_benchmark.ipynb`
3. `notebooks/laya_downstream_analysis_v2.ipynb`

The normalization benchmark additionally requires the frozen TNT Hybrid V3
checkpoint documented in `docs/downstream_laya_normalization_v1.md`. It is kept
separate from the Laya classifier because they are different models.

## Limitations

- M1 is synthetic noise, not a substitute for naturally informal Wolof.
- The strongest V1 gain is distribution-matched to TNT's M1 pretraining.
- Applying TNT to clean text reduced accuracy by 2.98 percentage points.
- M5 and M6 did not show a positive downstream normalization gain.
- The 40-way choice task is larger than Laya's recommended choice-set size.
- Dev was used for both checkpoint selection and temperature calibration.
- Router test results are exploratory because the official test outcomes had
  already informed the preceding feasibility audit.

## Licence

Code and the released fine-tuned checkpoint are provided under the Apache
License 2.0. Upstream datasets and models retain their own licences and require
their own attribution.

