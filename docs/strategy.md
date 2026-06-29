# COMPX525 Assignment 2 Full-Mark Strategy

## Grading Target

The assignment has four equally important experiment blocks:

1. Original data + BirdNET embeddings + ML classifier.
2. Original data + neural model or BirdNET fine-tuning.
3. Synthetic text-to-audio enrichment + BirdNET embeddings + ML classifier.
4. Synthetic text-to-audio enrichment + neural model or BirdNET fine-tuning.

The report must make the comparison fair: same deterministic validation split,
same metric, same label order, and clear evidence for what changed between A and
B.

## Current Experiment Results

All four required experiment blocks now have reportable Kaggle results:

| Experiment | Data | Model family | Validation macro AUC |
| --- | --- | --- | ---: |
| A1 | Original | BirdNET embeddings + ML classifier | 0.8869 |
| A2 | Original | CompactAudioCNN neural baseline | 0.6451 |
| B1 | Synthetic-enriched | BirdNET embeddings + ML classifier | 0.8940 |
| B2 | Synthetic-enriched | CompactAudioCNN neural baseline | 0.6494 |

Synthetic enrichment improved both matched comparisons:

- B1 versus A1: +0.0071 macro AUC.
- B2 versus A2: +0.0044 macro AUC.

This gives the report a clean assignment story: the validation split stays
fixed, the model family stays fixed inside each A/B pair, and the only intended
change is adding accepted synthetic rare-class training audio.

## 110526 Lecture Update

The instructor later clarified that the assignment should use only
`train_audio` and ignore soundscapes because soundscapes are too hard for this
assignment scope. Therefore the reportable synthetic experiments use only
synthetic clips whose labels exist in `train.csv`.

The earlier soundscape-inclusive synthetic run generated 430 clips and reached
0.9035/0.6572 for B1/B2, but it is now treated as superseded exploratory work.
The current reportable train-audio-only synthetic set has 150 clips across 25
underrepresented `train_audio` classes.

## Non-Negotiables

- Use the Kaggle label order from `sample_submission.csv` / `taxonomy.csv`.
- Evaluate with macro ROC-AUC while skipping validation classes with no positive
  examples, matching the Kaggle metric.
- Keep an 80/20 split from `train.csv`.
- Force singleton classes into validation, then put their generated synthetic
  examples into training for the synthetic-data experiments.
- Ignore soundscape audio/labels in the reportable experiments, per the
  110526 lecture update.
- Log every run into `/kaggle/working/a2_run.log` and save JSON summaries.

## Current Baseline Pipeline

The working pipeline is:

1. Build deterministic manifests:
   - `train_audio_split_manifest.csv`
2. Extract BirdNET embeddings in chunks on Kaggle GPU.
3. Aggregate 3-second BirdNET segment embeddings to clip features:
   - mean
   - standard deviation
   - max
   - resulting dimension: 3072
4. Train the experiment A1 ML classifier:
   - `StandardScaler`
   - `OneVsRestClassifier`
   - `SGDClassifier(loss="log_loss", class_weight="balanced")`
5. Record validation AUC and per-run summaries.

## Why Chunked BirdNET

Measured Kaggle benchmark:

- 128 files -> 1450 segments.
- 207.7 seconds total.
- 1.62 seconds per file.

Full `train_audio` extraction is therefore too large for a single clean run.
We use 16 chunks, save compact `.npz` feature files, and delete raw BirdNET CSV
and DB outputs after aggregation.

## Experiment A1: BirdNET + ML

Goal: reliable, defensible baseline.

Implementation:

- Extract all `train_audio` BirdNET chunk features.
- Train ML classifier on train split.
- Validate on val split.
- Include secondary labels in target vectors when available.

Report evidence:

- Data counts and rare-class summary.
- Split policy and singleton handling.
- Feature aggregation diagram.
- Validation macro AUC.
- Short error analysis for rare classes and missing train-positive classes.

## Experiment A2: Original Data Neural Model

Goal: independent neural baseline that is not just an ML head on BirdNET.

Recommended path:

- Use PyTorch for all neural experiments, because Kaggle GPU notebooks support
  it well and it keeps the training loop, augmentations, and inference code
  explicit for the report.
- Start with a compact CNN/EfficientNet-style mel-spectrogram model.
- Use 5-second or 10-second crops at 32 kHz for train audio.
- Use BCEWithLogitsLoss for multi-label targets.
- Add simple augmentations:
  - random crop
  - gain/noise
  - time/frequency masking
  - mixup only after the plain baseline is stable

Report evidence:

- Architecture diagram.
- Training curve.
- Validation AUC.
- Compare with A1, including why BirdNET may outperform or underperform.

## Experiment B1: Synthetic-Balanced BirdNET + ML

Goal: show whether synthetic text-to-audio augmentation improves rare-class
coverage.

Synthetic policy:

- Target rare classes first, especially count <= 5.
- For singleton classes, keep real example in validation and add generated
  examples only to training.
- Store a manifest with:
  - target label
  - prompt
  - generation seed
  - generated file path
  - duration
  - filtering decision
  - BirdNET confidence sanity check if available

Training:

- Re-run BirdNET feature extraction for accepted synthetic audio.
- Merge real-train features + synthetic-train features.
- Keep the original validation split unchanged.

Report evidence:

- Before/after class-count table.
- Synthetic filtering criteria.
- B1 AUC versus A1.
- Rare-class AUC or qualitative rare-class analysis.

## Experiment B2: Synthetic-Balanced Neural Model

Goal: test whether the neural model benefits from the same enriched training
distribution.

Implementation:

- Same architecture and validation split as A2.
- Train with real-train + accepted synthetic-train audio.
- Keep all hyperparameters as close as practical to A2, so the main changed
  variable is the data.

Report evidence:

- B2 AUC versus A2.
- Training curve comparison.
- Short discussion of overfitting or synthetic artifacts.

## Submission Strategy

- Do not spend Kaggle submission attempts until local validation has a credible
  model.
- First public submission should be a conservative inference notebook from the
  best validation model.
- Use submission attempts only for:
  - validating inference formatting
  - testing the current best model
  - one final ensemble or calibration pass

## Execution Status

1. Finished BirdNET feature extraction for all 16 real-data chunks.
2. Ran A1 ML validation on all real BirdNET features.
3. Ran A2 neural baseline with the original data split.
4. Designed the reportable train-audio-only rare-class synthetic plan with 150
   accepted clips.
5. Extracted BirdNET features for accepted synthetic clips.
6. Ran B1 and B2 with the unchanged validation split.

Next work is report writing, figure/table preparation, and optionally a
competition inference/submission notebook from the strongest validation model.
