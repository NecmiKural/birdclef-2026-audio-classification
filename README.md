# BirdCLEF 2026: Audio Species Classification

Multi-class acoustic species identification on the [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) dataset (birds, amphibians, mammals, reptiles, and insects recorded in the Brazilian Pantanal). Built as a university deep-learning assignment (COMPX525).

**Research question:** *Can AudioLDM2 synthetic text-to-audio enrichment of rare classes improve classification?* The answer turns out to depend on the model family, and that split is the interesting part.

> **Note on data:** The competition audio and metadata are governed by Kaggle's competition rules and are **not** redistributed here. This repo contains code, the report, figures, and per-class result tables. Download the data from the competition page to run anything.

A short summary that anyone can rerun on Kaggle, with one rare-class spectrogram and the rare-class results charted: [BirdCLEF 2026 synthetic enrichment summary](https://www.kaggle.com/code/necmikural/birdclef-2026-synthetic-enrichment-summary) (source in `kaggle_kernels/summary/`).

## Headline result

Validation results on one fixed 80/20 split of `train_audio` with one seed, scored with the Kaggle skip-empty **macro ROC-AUC**. These are not Kaggle leaderboard scores, and the soundscapes were not used in the reportable experiments.

| Pipeline | Without synthetic | With synthetic | Δ |
|---|---|---|---|
| BirdNET embeddings + ML | 0.8869 | **0.8940** | **+0.0071** |
| From-scratch CompactAudioCNN | 0.8028 | 0.7645 | −0.0383 |

BirdNET embeddings + ML reach 0.8940 validation macro ROC-AUC with synthetic enrichment and 0.8869 without, on one fixed 80/20 split of train_audio with one seed. This is not a Kaggle leaderboard score.

Synthetic enrichment **helps** the embedding pipeline overall (+0.0071 macro ROC-AUC). On its 24 rare-class targets the mean ΔAUC is +0.032, with 11 improved and 13 worsened, most of them with 1 to 4 positive validation examples, so that per-class picture is noisy. It **hurts** the from-scratch CNN once the real-data baseline is given enough training budget to learn the rare classes on its own. The report shows this is training-budget-sensitive: at 10 epochs the gap is small; at 20 epochs the unsynthetic baseline keeps climbing while the synthetic run plateaus and early-stops.

## Approach

Two model families, each run on original data (A) and synthetic-enriched data (B), per the assignment's 2×2 design:

- **BirdNET embeddings → classical ML.** Audio is chunked and passed through BirdNET to extract clip-level embeddings (extraction was parallelised across 16 Kaggle kernels to stay inside GPU limits; see `kaggle_kernels/`), then a classifier is trained on the embeddings.
- **CompactAudioCNN from scratch.** A small CNN trained directly on mel-spectrograms.
- **Synthetic enrichment (B).** AudioLDM2 generates synthetic clips for rare classes. Every clip was embedded with BirdNET as a sanity check; no classifier-based quality filter was applied (the report explains why and recommends one for the bird classes).

Dataset: 234 taxonomy labels, 206 covered by `train.csv`, highly imbalanced (29.5× ratio between the most frequent class and the rare-class floor; 4 classes have a single real example). Singleton classes are forced into validation.

## Repository layout

```
REPORT.md / REPORT.pdf     Full report: split, methodology, network diagrams, per-class results
figures/                   Class distribution, learning curves, experiment flow, example spectrograms
                           (the spectrograms are derived from BirdCLEF+ 2026 competition audio)
docs/
  kaggle_runbook.md         How the chunked Kaggle pipeline was run
  per_class_*.csv           Per-class A2-vs-B2 AUC breakdowns
  rare_class_*.csv          Rare-class AUC breakdowns
notebooks/
  birdclef-roc-auc.ipynb    ROC-AUC evaluation notebook
kaggle_kernels/             All Kaggle kernels used (notebooks + metadata, outputs cleared):
                            BirdNET embedding extraction (16 chunks), neural baselines,
                            synthetic generation, submission notebooks
```

## Tech

Python · PyTorch · BirdNET · AudioLDM2 · librosa · scikit-learn · Kaggle GPU kernels.

---
## License

The code is MIT licensed (see `LICENSE`). The licence does not cover the competition data or the models this project uses: BirdNET is CC BY-NC-SA 4.0 and AudioLDM2 (`cvssp/audioldm2`) is CC BY-NC-SA 4.0.

---
*University coursework (COMPX525 Deep Learning). Shared as a portfolio sample. Competition data not included.*
