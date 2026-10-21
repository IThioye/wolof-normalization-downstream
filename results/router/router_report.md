# Sentence-level normalization router prototype V1

## Protocol

The router is a regularized logistic regression trained only on the 320-row
official Injongo dev split expanded into six conditions (1,920 paired rows).
TNT and Laya remain frozen. Five-fold grouped out-of-fold predictions select a
threshold that maximizes downstream correctness while allowing at most a 0.5
percentage-point loss on clean dev inputs. All variants of one utterance share
the same `row_id` group.

The exported threshold is **0.714769**. Features contain only
surface, TNT-head and input-to-TNT-candidate measurements. No REAL reference,
synthetic operation count, noise-method label, gold confidence or intent label
is available to the router at inference.

## Development result (grouped out-of-fold)

| Strategy | Accuracy | Macro-F1 | Normalized |
|---|---:|---:|---:|
| Always skip | 68.18% | 69.37% | 0% |
| Always TNT | 65.78% | 66.76% | 100% |
| Router | **69.11%** | **70.19%** | 13.54% |
| Oracle | 73.18% | 73.93% | 5.00% |

The router gains +0.94%
over skip and +3.33%
over systematic normalization on its leakage-controlled development estimate.

## Exploratory test result

| Strategy | Accuracy | Macro-F1 | Normalized |
|---|---:|---:|---:|
| Always skip | 71.08% | 71.83% | 0% |
| Always TNT | 71.79% | 72.85% | 100% |
| Dev-trained router | **72.75%** | **73.55%** | 12.49% |
| Oracle | 76.80% | 77.26% | 5.72% |

| condition   |   always_normalize |   always_skip |   dev_trained_router |
|:------------|-------------------:|--------------:|---------------------:|
| REAL        |           0.778997 |      0.808777 |             0.808777 |
| M1_LIGHT    |           0.722571 |      0.742947 |             0.746082 |
| M1_MEDIUM   |           0.667712 |      0.639498 |             0.653605 |
| M1_HEAVY    |           0.603448 |      0.485893 |             0.568966 |
| M5          |           0.777429 |      0.810345 |             0.810345 |
| M6          |           0.757053 |      0.777429 |             0.777429 |

On test, the router accepts 478/3,828
candidates, captures 87/219 HELPED cases,
and accepts 23/192 HARMED cases. It
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
