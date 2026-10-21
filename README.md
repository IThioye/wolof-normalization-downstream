# Does Wolof text normalization improve intent classification?

This repository contains a controlled downstream experiment on Wolof. It asks
whether normalizing informal or noisy writing helps a model understand the
speaker's intent.

The short answer is: **sometimes, but not always**. Normalization substantially
helped under severe synthetic noise similar to the normalizer's training data.
It harmed clean text and lightly corrupted text. The main conclusion is
therefore not “always normalize”, but “normalization needs a reliable decision
mechanism and evaluation on naturally informal Wolof”.

The fine-tuned intent classifier is published at
[`IbrahimaThioye/laya-injongo-wolof-intent`](https://huggingface.co/IbrahimaThioye/laya-injongo-wolof-intent).

## What is in this study?

| Name | Role |
|---|---|
| **Wolof** | A major language of Senegal and neighbouring countries. Informal online writing often differs from standardized orthography. |
| **InjongoIntent** | A multilingual intent-classification dataset. Its official Wolof split provides standardized utterances and 40 intent labels. |
| **Laya Multilingual** | The downstream classifier. It reads a Wolof utterance and chooses one of the 40 intents. |
| **M1** | A deterministic, Beqi-inspired rule set that transforms standardized Wolof into synthetic noisy Wolof. It is not human-written informal data. |
| **TNT Hybrid V3** | A character-level Wolof normalizer with word and spacing decisions. It attempts to convert noisy text back to standardized Wolof. |

The Laya classifier and TNT normalizer are two separate frozen models. Laya was
fine-tuned only on original standardized Wolof. It never saw M1-noisy examples
during training.

## Experimental design

Every official test example keeps the same ID and Gold intent across all
conditions:

```text
Official Injongo Wolof test utterance
                 │
                 ├──────────────────────────────► REAL
                 │
                 ▼
       synthetic corruption method
                 │
                 ├──────────────────────────────► NOISY
                 │
                 ▼
       frozen TNT Wolof normalizer
                 │
                 └──────────────────────────────► NORMALIZED

REAL, NOISY and NORMALIZED are evaluated by the same frozen Laya checkpoint.
```

This isolates the effect of changing the input text. No classifier retraining
takes place between conditions.

### Official split policy

| Purpose | InjongoIntent file | Rows | Use |
|---|---|---:|---|
| Classifier training | `train.jsonl` | 2,240 | Optimization on REAL text only |
| Selection and calibration | `dev.jsonl` | 320 | Best epoch and temperature |
| Final evaluation | `test.jsonl` | 638 | Frozen paired benchmark |

No examples were moved between the official splits and no additional random
split was created.

## Main results

### V1: severe M1 stress test

| Test condition | Accuracy | Macro-F1 |
|---|---:|---:|
| REAL standardized Wolof | 80.88% | 81.18% |
| M1-heavy NOISY | 48.59% | 45.97% |
| M1-heavy after TNT | 60.34% | 60.87% |

Under heavy M1 corruption, TNT recovered **11.76 accuracy points** and 14.90
macro-F1 points. This is a distribution-matched result: TNT was pretrained on
M1-style errors, so it should not be generalized to every form of informal
Wolof.

### V2: clean preservation and noise severity

| Input to Laya | Without TNT | With TNT | Accuracy change |
|---|---:|---:|---:|
| REAL clean text | 80.88% | 77.90% | -2.98 pp |
| M1-light, about 2 operations/sentence | 74.29% | 72.26% | -2.04 pp |
| M1-medium, about 5 operations/sentence | 63.95% | 66.77% | +2.82 pp |
| M1-heavy, about 9.31 operations/sentence | 48.59% | 60.34% | +11.76 pp |

The sign changes with severity: TNT is harmful when little correction is
needed, then becomes useful as M1 corruption increases.

### Generalization beyond M1

| Noise method | Sentences changed | NOISY accuracy | After TNT | Change |
|---|---:|---:|---:|---:|
| M1 | 638 | 48.59% | 60.34% | +11.76 pp |
| M5, YouTube-mined variants | 46 | 81.03% | 77.74% | -3.29 pp |
| M6, Gold-informed linguistic generator | 576 | 77.74% | 75.71% | -2.04 pp |

These results do not show cross-method generalization. M5 also changed only 46
of 638 sentences, so its aggregate score primarily reflects unchanged inputs.

### Exploratory normalization router

A small sentence-level router was tested to decide whether TNT should run. On
the already-analysed combined test conditions, it normalized 12.5% of inputs
and reached 72.75% accuracy, compared with 71.08% for never normalizing and
71.79% for always normalizing.

This is exploratory rather than a final deployment result: official test
outcomes had already informed the feasibility analysis. A new human-noisy
holdout is required for an unbiased router evaluation.

## What is released?

```text
src/downstream/   Injongo loading, Laya fine-tuning, evaluation and routing
src/generation/   M1/M5/M6 corruption functions used in the experiments
src/modeling/     shared metrics and TNT architecture/loading code
notebooks/        guided training and evaluation workflows
results/          aggregate metrics, per-class tables and figures
tests/            split, determinism and protocol checks
docs/             detailed V1, V2 and router protocols
```

The repository does **not** redistribute Injongo utterances or text-level
prediction files. The code downloads the official pinned dataset files from
the upstream Masakhane repository. The public results contain aggregate or
class-level statistics only.

The Hugging Face repository contains the fine-tuned Laya classifier. It does
not contain TNT. Exact reproduction of the NORMALIZED condition additionally
requires the frozen TNT Hybrid V3 checkpoint documented in
[`docs/downstream_laya_normalization_v1.md`](docs/downstream_laya_normalization_v1.md).

## Quick start: use the intent classifier

Python 3.11 and `laya==0.3.21` were used for the released model.

```bash
python -m pip install laya==0.3.21
```

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
answer = result["answers"]["intent"]
print(answer["choice"], answer["confidence"])
```

The released model expects the same instruction and complete 40-label inventory
used during training. It performs classification only; it does not normalize
the input.

## Reproduce the classifier training

Clone the repository and install its dependencies:

```bash
git clone https://github.com/IThioye/wolof-normalization-downstream.git
cd wolof-normalization-downstream
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the lightweight checks first:

```bash
python -m pytest tests/test_downstream_normalization.py -q
python -m pytest tests/test_downstream_analysis_v2.py -q
```

Then use the notebooks in this order:

1. `notebooks/laya_injongo_wolof_finetuning.ipynb` downloads and validates the
   official splits, prepares Laya examples and fine-tunes the classifier.
2. `notebooks/laya_normalization_benchmark.ipynb` evaluates the frozen V1
   REAL/NOISY/NORMALIZED conditions.
3. `notebooks/laya_downstream_analysis_v2.ipynb` runs clean-preservation,
   severity, cross-method and detailed error analyses.

Long training and final-test cells are disabled by default. Read the notebook
switches before enabling them.

## Reproducibility references

- InjongoIntent source: [Masakhane NLU](https://github.com/masakhane-io/masakhane-nlu/tree/main/InjongoIntent/wol)
- pinned Injongo revision: `1f8be590da6699aee3dc23de6f63e801e2352eff`
- Laya Multilingual base: [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
- pinned Laya revision: `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`
- model weight SHA-256: `9f3aa8e8040e83db2dcb564e0645e73fd19025a0376b0aa09d8a838bac1396ce`
- experiment seed: `2026`

## Important limitations

- All evaluated noise is synthetic. The study does not establish the same
  effects on naturally informal human-written Wolof.
- The strongest result is matched to TNT's M1 pretraining distribution.
- TNT over-normalized some clean or mildly corrupted inputs.
- The classifier's 40-option task exceeds the choice-set size generally
  recommended for Laya.
- The development split was used both for model selection and temperature
  calibration.
- M2 and M3 were not applicable because they require a paired French sentence;
  M4 is a fixed external dataset rather than a corruption function.
- The router needs validation on a new, untouched human-noisy dataset.

## Licence and attribution

The code and released fine-tuned weights use the Apache License 2.0. The base
model, InjongoIntent and other upstream resources retain their own licences and
attribution requirements. See [`NOTICE`](NOTICE) for attribution.

