# Can a security encoder learn CVSS 4.0 metrics?

This reproducible experiment fine-tunes [SecureBERT 2.0](https://huggingface.co/cisco-ai/SecureBERT2.0-base) to predict the eleven CVSS 4.0 Base metrics from vulnerability descriptions.

It supports an article about a hybrid architecture for generating CVSS vectors. The production-oriented API and deterministic calculator live in [CVSS OSS Vulnerability Assessor API](https://github.com/samhermeli/cvss-oss-vulnerability-assessor-api).

> This is an educational proof of feasibility, not a production model. A predicted vector still needs deterministic CVSS validation and security review.

## Why an encoder?

This is a fixed multi-label classification task: one description maps to eleven metrics with small, known label sets. A specialized encoder predicts every metric in one local forward pass and returns a confidence per metric.

```text
vulnerability description
          │
          ▼
SecureBERT multi-head classifier
          │
          ├── sufficiently confident ──► deterministic CVSS 4.0 calculator
          │
          └── uncertain ───────────────► semantic fallback / human review
```

This experiment does **not** establish that an encoder is universally better than an LLM. It shows why an encoder is worth evaluating for this bounded stage: predictable local inference, fixed labels, and an explicit confidence gate. A production decision needs a blind, like-for-like comparison on the same dataset.

## Dataset

Training uses the exact public NVD feed included in this directory: `nvdcve-2.0-2025.json.gz`. It is the compressed version of the 2025 JSON file from the [Mendeley Data publication](https://data.mendeley.com/datasets/5343z8zfxg/1) *Fragmentation of CVSS Scores in the NVD*. The publication is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [DATASET.md](./DATASET.md) for attribution, provenance, and an integrity hash.

The script inspects 29,320 CVE entries and retains 8,658 records with both an English description and a complete CVSS 4.0 Base vector. The data is used as published: the experiment does not alter descriptions or labels.

## Contents

```text
cvss4-securebert-experiment/
├── DATASET.md                  # attribution and integrity
├── README.md                   # experiment guide
├── nvdcve-2.0-2025.json.gz     # public NVD 2025 feed used for training
├── securebert_cvss4_demo.py    # inspection, training, calibration and inference
├── request_example.json        # input contract example
└── prediction_example.json     # recorded inference example
```

The generated checkpoint is intentionally excluded. It is large, run-specific, and can be regenerated with the training command.

## Reproduce

### Inspect the data

This step only needs the Python standard library:

```bash
cd cvss4-securebert-experiment
python3 securebert_cvss4_demo.py inspect
```

Expected headline result:

```text
CVE total:                      29320
Eligible records with CVSS 4.0:  8658
```

### Train the short configuration

The demonstration configuration fixes the seed, uses 2,200 records, and trains for one epoch.

```bash
uv run \
  --with torch \
  --with "transformers>=4.48" \
  --with safetensors \
  python securebert_cvss4_demo.py train --quick
```

The script selects CUDA, Apple Silicon MPS, or CPU automatically. It writes `securebert_cvss4_demo.pt` beside the script; that generated file is ignored by Git.

### Run inference

```bash
uv run \
  --with torch \
  --with "transformers>=4.48" \
  --with safetensors \
  python securebert_cvss4_demo.py predict --request-file request_example.json
```

The output includes the predicted vector, per-metric confidence, local forward-pass latency, and an illustrative decision to accept the encoder or use a semantic fallback.

## Method

- **Base model:** `cisco-ai/SecureBERT2.0-base`.
- **Architecture:** a shared encoder and eleven independent classification heads.
- **Class imbalance:** weighted cross-entropy, with capped class weights calculated from the training split.
- **Split:** 80% training, 10% validation, 10% held-out test; seed `42`.
- **Calibration:** temperature scaling per metric, fitted only on validation and applied unchanged to test.
- **Scoring:** the model predicts vector components; deterministic CVSS 4.0 logic validates and scores the vector.

## Recorded quick-run results

The recorded run used 2,200 records: 1,760 for training, 220 for validation, and 220 held out for test. It trained for one epoch with sequence length 160 and batch size 8.

| Measure | Held-out test result |
|---|---:|
| Mean accuracy across the eleven metrics | 0.767 |
| Mean macro-F1 across the eleven metrics | 0.577 |
| Mean expected calibration error | 0.056 |

Attack Complexity was the strongest metric in this run (accuracy `0.959`, macro-F1 `0.775`). Privileges Required was the weakest (accuracy `0.509`, macro-F1 `0.442`). The included [prediction example](./prediction_example.json) records a 9-of-11 metric match and routes to fallback because several confidences are below the educational `0.70` threshold.

## Limits

- Mean per-metric accuracy is not exact vector accuracy.
- The split is random and small; temporal or family-aware splits would be stronger.
- The recorded latency is one local forward-pass measurement, not a service latency SLO.
- The threshold is illustrative; a real policy needs calibration, risk-aware fallback cost, and out-of-distribution handling.
- This experiment has not yet run a blind, like-for-like comparison with an LLM baseline.

## Licence

The experiment code is covered by this repository's [MIT License](../LICENSE). The included dataset is separately attributed under CC BY 4.0 in [DATASET.md](./DATASET.md).
