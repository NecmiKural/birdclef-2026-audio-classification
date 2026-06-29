# COMPX525 Assignment 2 Kaggle Runbook

## Files

- `assignment_2/notebooks/compx525_a2_birdclef_kaggle.ipynb`
  - Upload this to Kaggle.
- `assignment_2/notebooks/compx525_a2_birdclef_kaggle.py`
  - Editable notebook source with cell markers.
- `assignment_2/scripts/py_to_ipynb.py`
  - Regenerates the Kaggle notebook from the editable Python source.
- `assignment_2/scripts/prepare_birdnet_chunk_kernel.py`
  - Creates repeatable Kaggle kernel packages for BirdNET chunk extraction.

## Kaggle Setup

1. Create or open a Kaggle notebook.
2. Add the BirdCLEF+ 2026 competition dataset as input.
3. Use GPU only when running neural experiments. Stage 0 does not need GPU.
4. For BirdNET smoke test, enable internet unless `birdnet-analyzer` is attached as an offline dataset/package.

## CLI Setup

The CLI package folder is:

```bash
assignment_2/kaggle_kernel_stage0
```

It contains the notebook and `kernel-metadata.json` with:

- `competition_sources: ["birdclef-2026"]`
- `enable_gpu: false`
- `enable_internet: true`
- private notebook id: `necmikural/compx525-a2-birdclef-stage0`

Run or update the Kaggle notebook from the local workspace:

```bash
kaggle kernels push -p assignment_2/kaggle_kernel_stage0 -t 1800
kaggle kernels status necmikural/compx525-a2-birdclef-stage0
kaggle kernels output necmikural/compx525-a2-birdclef-stage0 -p assignment_2/kaggle_outputs/stage0 -o
```

This does not submit to the competition and does not spend daily submission attempts.

## First Run

Leave these flags as-is:

```python
RUN_BIRDNET_SMOKE = False
RUN_BIRDNET_FULL = False
RUN_NEURAL_SMOKE = False
```

Run the full notebook once. This should not load audio files. It should produce:

- `/kaggle/working/a2_run.log`
- `/kaggle/working/eda_summary.json`
- `/kaggle/working/split_summary.json`
- `/kaggle/working/audio_path_summary.json`
- `/kaggle/working/train_audio_split_manifest.csv`
- `/kaggle/working/train_soundscape_split_manifest.csv`

Send back the visible notebook output plus the log/JSON files.

## BirdNET Smoke Run

After Stage 0 is clean, set:

```python
RUN_BIRDNET_SMOKE = True
RUN_BIRDNET_FULL = False
RUN_NEURAL_SMOKE = False
```

Run again. Send back:

- `/kaggle/working/a2_run.log`
- `/kaggle/working/birdnet_smoke_manifest.csv`
- the first rows and column names of `/kaggle/working/birdnet_smoke_embeddings.csv`, if created

Do not enable `RUN_BIRDNET_FULL` until the smoke output format is confirmed.

Validated setup:

```python
pip install birdnet-analyzer perch-hoplite==0.1.3
```

`perch-hoplite` must be pinned to `0.1.3`; newer `1.0.x` releases renamed an
internal class and break BirdNET Analyzer 2.4.0 embedding extraction.

BirdNET embedding CSV format:

- `file_path`
- `start`
- `end`
- `embedding` as a comma-separated 1024-dimensional vector string

The notebook aggregates segment embeddings to clip-level features using
`mean + std + max`, producing 3072-dimensional clip features.

## BirdNET Benchmark

The benchmark kernel package is:

```bash
assignment_2/kaggle_kernel_birdnet_benchmark
```

Run:

```bash
kaggle kernels push -p assignment_2/kaggle_kernel_birdnet_benchmark -t 2400
kaggle kernels status necmikural/compx525-a2-birdnet-benchmark
kaggle kernels output necmikural/compx525-a2-birdnet-benchmark -p assignment_2/kaggle_outputs/birdnet_benchmark_v1 -o
```

Observed result for 128 files:

- 1450 embedding segments
- 207.7 seconds total
- 1.62 seconds/file
- 0.143 seconds/segment

This implies full `train_audio` extraction is too large for one run. Use chunked
extraction instead.

## BirdNET Chunk Extraction

The chunk test package is:

```bash
assignment_2/kaggle_kernel_birdnet_chunk_test
```

The notebook supports:

```python
RUN_BIRDNET_CHUNK = True
BIRDNET_CHUNK_INDEX = 0
BIRDNET_NUM_CHUNKS = 16
BIRDNET_CHUNK_SPLIT = "all"
BIRDNET_CHUNK_MAX_FILES = None
KEEP_RAW_BIRDNET_CSV = False
```

Each chunk emits compact reusable artifacts:

- `birdnet_<split>_chunkXX_ofYY_clip_features.npz`
- `birdnet_<split>_chunkXX_ofYY_clip_metadata.csv`
- `birdnet_<split>_chunkXX_ofYY_segments.csv`
- `birdnet_<split>_chunkXX_ofYY_summary.json`
- `birdnet_<split>_chunkXX_ofYY_manifest.csv`

Raw BirdNET CSV and DB files are deleted by default after aggregation to keep
Kaggle outputs smaller.

Kaggle currently allows only 2 concurrent GPU batch sessions for this account.
If a push fails with:

```text
Kernel push error: Maximum batch GPU session count of 2 reached.
```

wait for one running chunk to complete, then retry the failed chunk package.

If a retry for a never-created slug returns:

```text
Kernel push error: Notebook not found
```

create the same chunk with a fresh kernel slug such as `chunk-02-r1`. The
notebook output names remain correct because they are controlled by
`BIRDNET_CHUNK_INDEX`, not the Kaggle slug.

Prepare a real chunk kernel package:

```bash
python assignment_2/scripts/prepare_birdnet_chunk_kernel.py \
  --out-dir assignment_2/kaggle_kernel_birdnet_chunk_01 \
  --kernel-id necmikural/compx525-a2-birdnet-chunk-01 \
  --title 'COMPX525 A2 BirdNET Chunk 01' \
  --chunk-index 1 \
  --num-chunks 16 \
  --batch-size 32
```

Push and monitor:

```bash
kaggle kernels push -p assignment_2/kaggle_kernel_birdnet_chunk_01 -t 7200
kaggle kernels status necmikural/compx525-a2-birdnet-chunk-01
kaggle kernels output necmikural/compx525-a2-birdnet-chunk-01 \
  -p assignment_2/kaggle_outputs/birdnet_chunk_01_v1 -o
```

Chunk `00/16` was pushed as:

```bash
https://www.kaggle.com/code/necmikural/compx525-a2-birdnet-chunk-00
```

## BirdNET ML Baseline

After chunk outputs are attached to a Kaggle notebook or downloaded locally, set:

```python
RUN_BIRDNET_ML_BASELINE = True
```

The notebook scans `/kaggle/input` and `/kaggle/working` for
`birdnet_*_chunk*_clip_features.npz`, joins them to the deterministic split
manifest, and trains:

- `StandardScaler`
- `OneVsRestClassifier`
- `SGDClassifier(loss="log_loss", class_weight="balanced")`

Outputs:

- `birdnet_ml_baseline_summary.json`
- `birdnet_ml_val_predictions.csv`
- `birdnet_ml_feature_metadata.csv`

Partial smoke package using chunks `00` and `01`:

```bash
python assignment_2/scripts/prepare_birdnet_ml_kernel.py \
  --out-dir assignment_2/kaggle_kernel_birdnet_ml_partial_00_01 \
  --kernel-id necmikural/compx525-a2-birdnet-ml-partial-00-01 \
  --title 'COMPX525 A2 BirdNET ML Partial 00 01' \
  --kernel-source necmikural/compx525-a2-birdnet-chunk-00 \
  --kernel-source necmikural/compx525-a2-birdnet-chunk-01 \
  --classifier-max-iter 10

kaggle kernels push -p assignment_2/kaggle_kernel_birdnet_ml_partial_00_01 -t 1800
```

This is not the final A1 score. It only verifies that attached chunk outputs are
discoverable and the ML baseline path trains end-to-end on Kaggle.

Observed partial result with chunks `00` and `01`:

- feature rows joined: 4363
- train rows: 3478
- validation rows: 885
- feature dimension: 3072
- classifier max iterations: 10
- fit time: 16.4 seconds
- validation macro AUC: 0.7791

This score is useful only as a pipeline smoke test because the training feature
set covers a subset of classes.

Full A1 result with all 16 BirdNET chunks:

- feature rows joined: 35179
- train rows: 28133
- validation rows: 7046
- feature dimension: 3072
- classifier max iterations: 20
- fit time: 187.2 seconds
- train positive classes: 202
- validation positive classes: 205
- validation macro AUC: 0.8869

This is the current reportable result for experiment A1: original data,
BirdNET embeddings, and a classical ML classifier.

## Neural Smoke Run

After path checks are clean, set:

```python
RUN_BIRDNET_SMOKE = False
RUN_BIRDNET_FULL = False
RUN_NEURAL_SMOKE = True
```

Use GPU for this run. This is only a tiny sanity check for audio loading,
mel-spectrogram creation, model forward/backward pass, and metric dimensions.

Observed neural smoke result:

- Kaggle preinstalled `torch=2.10.0+cu128` failed on the assigned GPU with
  `CUDA error: no kernel image is available for execution on the device`.
- Installing `torch==2.5.1+cu121` and `torchaudio==2.5.1+cu121` fixed CUDA
  compatibility.
- CUDA conv self-test passed.
- Smoke train batches completed on tensors shaped `(8, 1, 128, 501)`.
- Validation tensors had shape `y_true=(64, 234)` and `y_pred=(64, 234)`.
- Smoke macro AUC was `0.5444`, which is only a dimension/path sanity check.

Observed neural baseline v1 result:

- model: `CompactAudioCNN`
- train rows: 4096
- validation rows: 2048
- epochs: 3
- batch size: 24
- input: 5 seconds at 32 kHz, 128 mel bins
- best epoch: 2
- best validation macro AUC: 0.5798

This was the first reportable A2 neural result, but it was much weaker than
the A1 BirdNET+ML baseline. The follow-up run increased train and validation
coverage and trained for one additional epoch.

Observed neural baseline v2 result:

- model: `CompactAudioCNN`
- train rows: 12000
- validation rows: 4096
- epochs: 4
- batch size: 24
- input: 5 seconds at 32 kHz, 128 mel bins
- best epoch: 4
- best validation macro AUC: 0.6451
- epoch AUCs: 0.5678, 0.5740, 0.6218, 0.6451

This is the current reportable A2 neural result. It improves over v1 by
0.0653 validation macro AUC, and the best checkpoint occurs at the final
epoch, suggesting that more data/training continues to help this architecture.

## Synthetic Enrichment Plan

The assignment's experiment B requires synthetic audio enrichment before
re-running the BirdNET+ML and neural experiments. The README/lecture wording
uses `AudioLM2`, but the reproducible public text-to-audio implementation used
here is `AudioLDM2` through `cvssp/audioldm2`. The plan is generated by:

```bash
python scripts/create_synthetic_plan.py --target-floor 10
```

Original soundscape-inclusive plan summary:

- target floor: 10 training examples per class
- taxonomy classes: 234
- classes receiving synthetic samples: 53
- total synthetic samples planned: 430
- train-audio rare-class samples planned: 150
- soundscape-only class samples planned: 280
- taxonomy-only class samples planned: 0

This keeps the original validation split unchanged. Singleton real `train_audio`
examples remain in validation, and their generated samples are assigned only to
the synthetic training set.

110526 lecture update:

- The instructor clarified that reportable experiments should use only
  `train_audio` and ignore soundscapes.
- The soundscape-inclusive 430-sample run is therefore exploratory/superseded.
- The current reportable synthetic plan is:
  - target floor: 10 training examples per underrepresented `train_audio` class
  - classes receiving synthetic samples: 25
  - total synthetic samples used: 150
  - soundscape-only class samples used: 0
  - plan file: `synthetic/synthetic_plan_floor10_train_audio_only.csv`

Synthetic generation uses AudioLDM2 through Hugging Face Diffusers with prompts
recorded in `synthetic/synthetic_plan_floor10.csv`. The first Kaggle run is a
small smoke test:

```bash
python scripts/prepare_synthetic_generation_kernel.py \
  --out-dir kaggle_kernel_synthetic_smoke_v1 \
  --kernel-id necmikural/compx525-a2-synthetic-smoke-v1 \
  --title 'COMPX525 A2 Synthetic Smoke V1' \
  --start 0 \
  --limit 6 \
  --steps 25 \
  --install-torch-cu121

kaggle kernels push -p assignment_2/kaggle_kernel_synthetic_smoke_v1 -t 7200
```

Synthetic smoke notes:

- v1 failed because Kaggle's preinstalled `torchvision` was incompatible with
  the installed cu121 `torch` package during AudioLDM2/CLAP imports.
- v2 fixed that by uninstalling `torchvision`, but the packaged plan CSV was
  not visible from the notebook working directory.
- v3 embedded the plan CSV into the notebook and loaded the model, but failed
  because the `cvssp/audioldm2` repository's language-model component needs to
  be loaded as `GPT2LMHeadModel` with current Diffusers/Transformers.
- v4 generated 6/6 smoke samples successfully with `cvssp/audioldm2`, 25
  inference steps, 5-second audio, and zero generation failures.
- v5 generated 2/2 PCM smoke samples successfully. Output WAV files are mono,
  16 kHz, exactly 5.0 seconds, and 16-bit PCM.

The generation script now writes PCM 16-bit WAV at 16 kHz. Generate the full
430 samples in five chunks of 86 planned rows:

```text
chunk 00: start 0,   limit 86
chunk 01: start 86,  limit 86
chunk 02: start 172, limit 86
chunk 03: start 258, limit 86
chunk 04: start 344, limit 86
```

After all chunks finish, merge manifests/audio, then filter accepted audio with
BirdNET before the B1/B2 training runs.

Observed full synthetic generation result:

- chunks completed: 5/5
- planned rows: 430
- generated WAV files: 430
- generation failures: 0
- duplicate synthetic IDs: 0
- output format: mono, 16 kHz, 5.0 seconds, 16-bit PCM
- local merged manifest:
  `synthetic/merged_floor10/synthetic_generated_manifest.csv`
- local merged summary:
  `synthetic/merged_floor10/synthetic_generated_summary.json`

Only the 150 generated clips whose labels are present in `train.csv` are used
for the current reportable B1/B2 experiments.

Observed synthetic BirdNET filter/embedding result:

- attached generation chunks: 5/5
- manifest rows discovered: 430
- audio files discovered: 430
- prepared for embedding: 430
- accepted after BirdNET embedding sanity check: 430
- BirdNET embedding segments: 860
- synthetic clip features: 430
- synthetic clip feature dimension: 3072
- output feature file:
  `kaggle_outputs/synthetic_birdnet_v1/synthetic_birdnet_clip_features.npz`
- accepted manifest:
  `kaggle_outputs/synthetic_birdnet_v1/synthetic_birdnet_filtered_manifest.csv`

Observed B1 synthetic BirdNET+ML result, train_audio-only:

- real BirdNET feature rows: 35179
- real train rows: 28133
- real validation rows: 7046
- accepted synthetic train rows: 150
- total train rows: 28283
- train positive classes: 206
- validation positive classes: 205
- classifier: `StandardScaler + OneVsRestClassifier(SGDClassifier log_loss)`
- classifier max iterations: 20
- fit time: 182.1 seconds
- validation macro AUC: 0.8940
- A1 comparison: 0.8869 -> 0.8940, delta +0.0071
- synthetic added 4 singleton train-positive classes that were missing from
  the A1 real-train split.

This is the current reportable result for experiment B1: synthetic-enriched
training data, BirdNET embeddings, and a classical ML classifier.

Observed B2 synthetic neural result, train_audio-only:

- model: `CompactAudioCNN`
- real/synthetic training rows: 12150
- accepted synthetic train rows: 150
- validation rows: 4096
- epochs: 4
- batch size: 24
- input: 5 seconds at 32 kHz, 128 mel bins
- train positive classes: 206
- validation positive classes: 206
- best epoch: 3
- best validation macro AUC: 0.6494
- epoch AUCs: 0.6025, 0.6348, 0.6494, 0.6086
- A2 comparison: 0.6451 -> 0.6494, delta +0.0044

This is the current reportable result for experiment B2: synthetic-enriched
training data and the same compact PyTorch neural architecture used for A2.
The validation split and main hyperparameters were kept aligned with A2, so the
main changed variable is the 150 accepted train-audio-only synthetic rare-class
training clips.

Superseded soundscape-inclusive B results:

- B1 with 430 synthetic clips including soundscape-only labels: 0.9035
- B2 with 430 synthetic clips including soundscape-only labels: 0.6572
- These are not the current reportable results after the 110526 lecture update.

## Kaggle Submission Smoke

The assignment is graded through the Moodle PDF report, not the Kaggle
leaderboard. A minimal CPU notebook submission was still run to verify the
Kaggle submission path:

- kernel: `necmikural/compx525-a2-submission-smoke-v1`
- output file: `submission.csv`
- submission time: 2026-05-07 09:12:54.987000
- status: complete
- public score: 0.500

This was a format/path smoke test based on `sample_submission.csv`, not an
inference notebook for the validation-best models.
