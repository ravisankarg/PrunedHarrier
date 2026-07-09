# PrunedHarrier

PrunedHarrier is a local, resumable pipeline for compressing `microsoft/harrier-oss-v1-0.6b` into a smaller `12` layer, `384` hidden-size embedding student.

The workflow is designed for a small local GPU. The default plan keeps the trainable student on CUDA and the larger teacher on CPU, uses repo-local caches, saves restartable checkpoints, and runs a cheap MTEB smoke check every checkpoint interval.

## What This Builds

- Teacher: `microsoft/harrier-oss-v1-0.6b`
- Student: `12` layers, `hidden_size=384`, `intermediate_size=1536`
- Initialization: deterministic tensor slicing from the teacher
- Layer map:
  - student `0` <- teacher `0`
  - student `1` <- teacher `2`
  - student `2` <- teacher `5`
  - student `3` <- teacher `7`
  - student `4` <- teacher `10`
  - student `5` <- teacher `12`
  - student `6` <- teacher `15`
  - student `7` <- teacher `17`
  - student `8` <- teacher `20`
  - student `9` <- teacher `22`
  - student `10` <- teacher `25`
  - student `11` <- teacher `27`

## Losses

Phase 1 uses unlabeled text distillation:

```text
similarity_weight * final embedding similarity-matrix MSE
+ cosine_weight * final embedding cosine loss
+ layerwise_weight * hidden-state layerwise MSE
```

Because the student is `384d` and the teacher is `1024d`, training uses temporary projection heads:

```text
final embedding projection: 384 -> 1024
layerwise hidden projection: 384 -> 1024
```

The saved student remains a `384d` model. Projection heads are stored only in `trainer_state.pt` so training can resume.

Phase 2 adds task-aware contrastive loss over prompted `query` / `positive` pairs:

```text
teacher_weight * distillation_loss
+ contrastive_weight * symmetric in-batch contrastive loss
```

## Setup

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

export HF_HOME=$PWD/.cache/huggingface
export HF_DATASETS_CACHE=$PWD/.cache/huggingface/datasets
export MTEB_CACHE=$PWD/.cache/mteb
export PYTHON=$PWD/.venv/bin/python

# Optional, but recommended for rate limits and gated/private resources.
export HF_TOKEN='your_huggingface_token'
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

Check the environment:

```bash
python scripts/00_check_env.py
```

## Step 1: Prepare Teacher And Student

Downloads the teacher snapshot and creates the sliced `12L/384d` student.

```bash
python scripts/01_prepare_models.py \
  --teacher microsoft/harrier-oss-v1-0.6b \
  --student-config configs/student_12l_384.json \
  --out-dir models/harrier_student_12l_384_sliced
```

Resume behavior: Hugging Face downloads resume automatically. If the student already has `config.json`, `model.safetensors`, and `slice_manifest.json`, the script skips the rebuild. Add `--force` to rebuild.

## Step 2: Download Unlabeled Multilingual Text

Downloads `25K` bounded Wikipedia passages per language for the configured top-20 language list.

```bash
python scripts/02_download_unlabeled_text.py \
  --languages configs/top20_languages.json \
  --per-language 25000 \
  --max-chars 2000 \
  --out data/processed/unlabeled_top20_25k.jsonl
```

Resume behavior: rerun the same command after a power or network interruption. It skips languages that already have `25000` rows.

## Step 3: Phase 1 Distillation

The first debug run showed that `1024 -> 384` width slicing produces a very weak raw STS embedding space, and that heavy projected/layerwise loss did not repair it. Prefer a raw-embedding similarity-first run before using projected cosine or layerwise losses heavily.

Recommended restart for phase 1:

```bash
python scripts/03_train_distill.py \
  --student models/harrier_student_12l_384_sliced \
  --teacher models/teacher_harrier_0_6b \
  --train-jsonl data/processed/unlabeled_top20_25k.jsonl \
  --output-dir runs/phase1_similarity_only \
  --max-length 512 \
  --batch-size 8 \
  --grad-accum 4 \
  --lr 2e-5 \
  --max-steps 5000 \
  --similarity-weight 1.0 \
  --cosine-weight 0.0 \
  --layerwise-weight 0.0 \
  --student-device cuda \
  --student-dtype bf16 \
  --teacher-device cuda \
  --teacher-dtype bf16 \
  --eval-every 500 \
  --save-every 500 \
  --mteb-slice-config configs/mteb_smoke.json \
  --resume
```

If STS starts moving in the right direction, extend it. If not, move directly to task-aware contrastive training.

This command uses the latest requested large microbatch settings:

```text
batch size: 16
gradient accumulation: 4
effective rows per optimizer step: 64
```

It requests BF16 explicitly.

```bash
python scripts/03_train_distill.py \
  --student models/harrier_student_12l_384_sliced \
  --teacher models/teacher_harrier_0_6b \
  --train-jsonl data/processed/unlabeled_top20_25k.jsonl \
  --output-dir runs/phase1_unlabeled_distill \
  --max-length 512 \
  --batch-size 16 \
  --grad-accum 4 \
  --lr 2e-5 \
  --max-steps 20000 \
  --similarity-weight 1.0 \
  --cosine-weight 0.25 \
  --layerwise-weight 0.25 \
  --layer-map-config configs/student_12l_384.json \
  --student-device cuda \
  --student-dtype bf16 \
  --teacher-device cpu \
  --teacher-dtype bf16 \
  --eval-every 1000 \
  --save-every 1000 \
  --mteb-slice-config configs/mteb_smoke.json \
  --resume
```

If 4 GB VRAM runs out of memory, first reduce to:

```text
--batch-size 1 --grad-accum 32
```

or lower sequence length:

```text
--max-length 256
```

Resume behavior: `--resume` loads `runs/phase1_unlabeled_distill/checkpoint-latest`, including optimizer, scheduler, scaler, and projection-head state when available.

## Step 4: Download Task-Aware Contrastive Data

This step first tries compatible MTEB mirror datasets for real task data, then fills missing language/task buckets from the unlabeled corpus. Retrieval uses fallback by default because joining MIRACL qrels to the streamed corpus is slow; pass `--hf-retrieval` only if you explicitly want to try that expensive path.

```bash
python scripts/04_download_task_contrastive_data.py \
  --mode hf-then-fallback \
  --languages configs/top20_languages.json \
  --sources configs/task_data_sources.json \
  --per-task-language 5000 \
  --unlabeled-fallback data/processed/unlabeled_top20_25k.jsonl \
  --out data/processed/task_contrastive_top20.jsonl
```

Resume behavior: rerun the same command. It skips language/task buckets that already have the requested count. Use `--force` to rebuild the file.

To use only local prompted fallback data:

```bash
python scripts/04_download_task_contrastive_data.py \
  --mode fallback-only \
  --languages configs/top20_languages.json \
  --sources configs/task_data_sources.json \
  --per-task-language 5000 \
  --unlabeled-fallback data/processed/unlabeled_top20_25k.jsonl \
  --out data/processed/task_contrastive_top20.jsonl
```

## Step 5: Phase 2 Task-Aware Training

```bash
python scripts/03_train_distill.py \
  --student runs/phase1_unlabeled_distill/checkpoint-latest \
  --teacher models/teacher_harrier_0_6b \
  --train-jsonl data/processed/task_contrastive_top20.jsonl \
  --output-dir runs/phase2_task_contrastive \
  --max-length 512 \
  --batch-size 2 \
  --grad-accum 16 \
  --lr 1e-5 \
  --max-steps 20000 \
  --contrastive-weight 1.0 \
  --teacher-weight 0.5 \
  --layerwise-weight 0.10 \
  --layer-map-config configs/student_12l_384.json \
  --student-device cuda \
  --student-dtype bf16 \
  --teacher-device cpu \
  --teacher-dtype bf16 \
  --eval-every 1000 \
  --save-every 1000 \
  --mteb-slice-config configs/mteb_smoke.json \
  --resume
```

Resume behavior: `--resume` loads `runs/phase2_task_contrastive/checkpoint-latest`.

## Step 6: MTEB Checks

Cheap smoke check:

```bash
python scripts/05_eval_mteb_slice.py \
  --model runs/phase2_task_contrastive/checkpoint-latest \
  --config configs/mteb_smoke.json \
  --output-dir eval_results/phase2_mteb_smoke
```

Larger multilingual slice:

```bash
python scripts/05_eval_mteb_slice.py \
  --model runs/phase2_task_contrastive/checkpoint-latest \
  --config configs/mteb_slice.json \
  --output-dir eval_results/phase2_mteb_slice
```

`configs/mteb_smoke.json` is intended for frequent checkpoint checks. `configs/mteb_slice.json` includes classification, clustering, retrieval, and STS tasks and may download or evaluate much more data.

## Compare Phase 1 Checkpoints On CPU

Use this when you want to decide which phase-1 checkpoint should feed phase 2. It evaluates selected checkpoints on CPU and writes:

```text
checkpoint_scores.csv
checkpoint_scores_summary.json
```

Broad comparison over selected checkpoints:

```bash
python scripts/06_eval_checkpoints_cpu.py \
  --run-dir runs/phase1_unlabeled_distill \
  --config configs/mteb_slice.json \
  --steps 2000,8000,10000,20000 \
  --output-dir eval_results/phase1_checkpoint_cpu
```

Cheap STS-only smoke:

```bash
python scripts/06_eval_checkpoints_cpu.py \
  --run-dir runs/phase1_unlabeled_distill \
  --config configs/mteb_smoke.json \
  --steps 2000 \
  --output-dir eval_results/cpu_checkpoint_smoke
```

List available checkpoints without running evaluation:

```bash
python scripts/06_eval_checkpoints_cpu.py \
  --run-dir runs/phase1_unlabeled_distill \
  --steps 2000,8000,10000 \
  --list-only
```

## Validation Run

The scripts have been smoke-tested in `.venv` with:

```bash
python -m py_compile scripts/*.py
python scripts/00_check_env.py
python scripts/03_train_distill.py --max-steps 1 --layerwise-weight 0.25 ...
python scripts/05_eval_mteb_slice.py --config configs/mteb_smoke.json ...
```

The full unlabeled corpus download was also verified locally:

```text
500000 valid JSONL rows
25000 rows for each configured language
```
