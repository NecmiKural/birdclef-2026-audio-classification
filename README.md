# BirdCLEF 2026 — Audio Species Classification

Multi-class acoustic species identification on the [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) dataset (birds, amphibians, mammals, reptiles, and insects recorded in the Brazilian Pantanal). Built as a university deep-learning assignment (COMPX525).

**Research question:** *Can AudioLDM2 synthetic text-to-audio enrichment of rare classes improve classification?* The answer turns out to depend on the model family — and the result is the interesting part.

> **Note on data:** The competition audio and metadata are governed by Kaggle's competition rules and are **not** redistributed here. This repo contains code, the report, figures, and per-class result tables. Download the data from the competition page to run anything.

## Headline result

Evaluated on a fixed 80/20 split with the Kaggle skip-empty **macro ROC-AUC**:

| Pipeline | Without synthetic | With synthetic | Δ |
|---|---|---|---|
| BirdNET embeddings + ML | 0.8869 | **0.8940** | **+0.0071** |
| From-scratch CompactAudioCNN | 0.8028 | 0.7645 | −0.0383 |

Synthetic enrichment **helps** the embedding pipeline (it gains on 24 rare-class targets, +0.032 mean ΔAUC) but **hurts** the from-scratch CNN once the real-data baseline is given enough training budget to learn the rare classes on its own. The report shows this is training-budget-sensitive: at 10 epochs the gap is small; at 20 epochs the unsynthetic baseline keeps climbing while the synthetic run plateaus and early-stops.

## Approach

Two model families, each run on original data (A) and synthetic-enriched data (B), per the assignment's 2×2 design:

- **BirdNET embeddings → classical ML.** Audio is chunked and passed through BirdNET to extract clip-level embeddings (extraction was parallelised across 16 Kaggle kernels to stay inside GPU limits — see `kaggle_kernels/`), then a classifier is trained on the embeddings.
- **CompactAudioCNN from scratch.** A small CNN trained directly on mel-spectrograms.
- **Synthetic enrichment (B).** AudioLDM2 generates synthetic clips for rare classes, with BirdNET used to filter low-quality generations.

Dataset: 234 taxonomy labels, 206 covered by `train.csv`, highly imbalanced (29.5× ratio between the most frequent class and the rare-class floor; 4 classes have a single real example). Singleton classes are forced into validation.

## Repository layout

```
REPORT.md / REPORT.pdf     Full report: split, methodology, network diagrams, per-class results
figures/                   Class distribution, learning curves, experiment flow, example spectrograms
docs/
  kaggle_runbook.md         How the chunked Kaggle pipeline was run
  strategy.md               Experimental strategy
  per_class_*.csv           Per-class A2-vs-B2 AUC breakdowns
  study_guide_tr.md         Study notes (Turkish)
notebooks/
  birdclef-roc-auc.ipynb    ROC-AUC evaluation notebook
kaggle_kernels/             All Kaggle kernels used (notebooks + metadata, outputs cleared):
                            BirdNET embedding extraction (16 chunks), neural baselines,
                            synthetic generation, submission notebooks
```

## Tech

Python · PyTorch · BirdNET · AudioLDM2 · librosa · scikit-learn · Kaggle GPU kernels.

---
*University coursework (COMPX525 Deep Learning). Shared as a portfolio sample. Competition data not included.*
