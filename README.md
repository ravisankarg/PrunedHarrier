# PrunedHarrier

PrunedHarrier is a local, resumable pipeline for compressing `microsoft/harrier-oss-v1-0.6b` into a smaller `12` layer, `384` hidden-size embedding student.

The workflow is designed for a small local GPU. The default plan keeps the trainable student on CUDA and the larger teacher on CPU, uses repo-local caches, and saves restartable checkpoints. MTEB evaluation is disabled during training by default so checkpoint evaluation cannot cause an out-of-memory failure.

## Encoder Distillation Experiment Ledger

This is the consolidated record of the encoder experiments completed so far. Decoder-only work is intentionally excluded. “Baseline” means the student before distillation unless the row explicitly says teacher or projection control. Scores are not directly interchangeable across runs: STS is a sentence-level semantic score, while cosine, MSE, retrieval, clustering, and classification measure different geometries or task behaviors.

| Experiment | Student base architecture and compression strategy | Student baseline KPI | Distillation strategy and revisions | KPI result | First-principles verdict / failure reason |
|---|---|---|---|---|---|
| **HG2: 270M pruning control** | Harrier 270M body, 12 copied layers, full `d=640`, `FFN=2048`, `4Q/1KV`, `head_dim=256`; then vocabulary-prioritized pruning to `82,993` rows for the ~120M target. | Teacher: STS `0.8359`, MRR `0.9962`, KNN `0.4242`, NMI `0.5222`. 12-layer/full-vocab student: STS `0.2321`, MRR `0.1900`, KNN `0.0985`, NMI `0.0495`. Vocab-pruned control: STS `0.0676`. | Exact layer copy + vocabulary pruning; phase-1 final-embedding similarity/cosine was planned, with task-contrastive phase 2 and tokenizer-safe loss guards. | Fresh `1,379`-text pruned check reached STS `-0.0046`; no promoted post-distillation checkpoint. | Layer deletion changes the computation at every depth; vocabulary pruning also changes tokenization. A final embedding loss cannot reconstruct missing intermediate transformations or a missing token inventory. |
| **HG2: PCA-projected 12L/640 → 384** | 12-layer, full-width `d=640` vocabulary-pruned student with a `640→384` output projection. | Fresh compressed STS `-0.0046`; PCA teacher target STS `0.8410`. | Teacher `1024→384` PCA targets; MSE + cosine + similarity-matrix losses; random-projection control. | Direct STS `0.2684` at checkpoint 2,000 and `0.2736` at 2,500; training target cosine `0.485` at step 2,572. | The target was learnable, but the run was short and optimized a projected target without enough task-aware signal. It showed movement, not recovery of the original task geometry. |
| **HG2: 0.6B → 12L/384 direct slice** | 12 layers, `d=384`, `FFN=1536`, `6Q/3KV`, `head_dim=64`; direct tensor slicing from the 0.6B teacher. | Raw student-to-teacher target cosine was about `0.001`. An apparent STS `0.0483` result was discarded because the export reported the wrong embedding dimension. | Similarity-only MSE: final `0.0752` at 5,000 steps; similarity + layerwise loss: `0.0905` at 2,000, then `0.0663` at 11,000. | Training alignment improved, but no trustworthy external KPI recovery. | Width slicing is not a neutral compression: it selects a coordinate subspace from a basis learned at `d=1024`. Projecting the loss does not make the sliced student’s native basis semantic. |
| **HG2: PCA-aligned 384 + task contrastive** | Same direct-sliced 12L/384 student, trained against a validated PCA teacher target. | PCA teacher STS `0.840966`; target/student alignment cosine `0.99995` before the schedule probes. | Original schedule, high learning rate, constant learning rate, pairwise similarity, then balanced task-contrastive data; corrected corpus was tested separately. | Direct STS: `0.0307` (original), `0.1633` (high LR), `0.1917` (constant), `0.2025` (pairwise). Task pilot: STS `0.2995`, 20NG NMI `0.1219` at step 500. Corrected-data pilot: STS `0.2044` at step 100. | The teacher target was valid, so target construction was not the bottleneck. The student basis/capacity, optimizer schedule, and early task-data mixture were; the task pilot started before the representation was stable and never reached a final multi-task checkpoint. |
| **Guru: geometric PCA/SVD initialization** | 12-layer bidirectional Gemma3-style encoder, `d=384`, `6Q/1KV`, `head_dim=64`, `FFN=2048`, full vocabulary; PCA/SVD transformer initialization, about `133.1M` parameters. | Teacher STS `0.8111`; fresh initialization STS `0.4013`. Token-table PCA retained `0.7626` variance with row cosine `0.8896`. | Frozen PCA table + rank-32 embedding LoRA + identity adapter; MSE, cosine, and PCA-Gram geometry objectives; MSE/geometry/mixed phase-2 ablations. | STS rose to `0.4260` at step 500, then fell to `0.2627` at step 5,000; phase-2 external KPIs were incomplete. | Preserving table geometry is not preserving sentence geometry. A frozen low-rank adaptation path did not have enough freedom to repair the nonlinear encoder’s mismatch, and the geometry losses encouraged the wrong invariant. |
| **RG / UniGemma vector-only** | Gemma3 270M, 18 layers, `d=640`, frozen shared vocabulary/body, mean pooling, `640→768` output projection. | Vector MSE about `0.00170` at step 80; no external student KPI. | Vector regression only. | No task KPI; experiment stopped as a diagnostic. | A small coordinate MSE says the student matches the teacher’s vectors numerically, not that neighborhoods or sentence semantics are preserved. Without task/evaluation pressure, the objective was underdetermined. |
| **RG / UniGemma main 640** | Same 18L/640 body, LoRA `r=16`, mean pooling, LayerNorm + `640→768` projection; the body remained full-size and mostly frozen. | No separately reported compressed baseline; this was a quality-control run, not a true architectural compression. | 5M examples; `0.25` vector + `0.50` cosine + `0.25` pairwise loss; fixed-resume continuation. | Cosine `0.8461` at step 7,240 and `0.8643` at step 10,460. MTEB STS: Spearman `0.8014`, Pearson `0.8091`. | Best completed sentence KPI, but it does not prove compression: the expensive 270M encoder remains. Only STS was measured, so retrieval, clustering, and classification preservation remain unknown. |
| **RG / UniGemma compact 384** | 12 layers, `d=384`, `FFN=1536`, 262K vocabulary; base tensor-slice initialization, mean pooling, `384→768` projection, LoRA. | No trustworthy external baseline was retained for the raw compact slice. | Same vector/cosine/pairwise family as the 640 run. | Cosine `0.7404` at step 11,540; no external MTEB result. | The smaller student inherits a basis and capacity mismatch from direct slicing. The lower training cosine is evidence that the projection head cannot compensate for a body whose representational subspace was never trained at `d=384`. |
| **Ramana: token-table projection control** | Gemma `262,144×640` token table projected to `384` with PCA; no transformer-body distillation. | PCA variance retained `0.8148`; pairwise row-cosine Pearson `0.9918`. | PCA only; no sentence-level distillation. | Same table-geometry KPIs; no sentence KPI. | Excellent row geometry does not establish sentence embedding quality. This isolates the vocabulary-table problem but cannot answer whether the encoder’s compositional computation survives. |
| **Granite: architecture baseline** | ModernBERT, 97M parameters, 12 layers, `d=384`, `FFN=1536`, 12 heads, vocab 180K, CLS pooling, local attention 128. | STS `0.7794`, 20NG NMI `0.4318`, Banking77 `0.7616`, SciFact `0.6760`. Local attention 512 produced the same STS `0.7794`. | LoRA contrastive trainer was proposed but not run; this is an architecture/control baseline rather than a completed distillation. | No distillation KPI. | It demonstrates that a native 384-wide encoder can be strong, but it does not show that slicing a 640/1024-wide trained encoder produces the same basis. |

### What the ledger says

- The best completed sentence result is RG/UniGemma at STS Spearman `0.8014`, but that model keeps the full encoder body and is therefore a quality benchmark, not a compression win.
- The strongest actual compressed pilot is HG2 PCA + task contrastive at STS `0.2995`; it is promising only as a direction because it stopped at step 500 and has no final multi-task report.
- The repeated failure is structural: layer pruning, width slicing, vocabulary pruning, and projection change different parts of the function. A loss on the final vector cannot replace capacity that was removed upstream.
- The clean next mission is the 270M→120M path in [`docs/harrier_270m_to_120m_mission.md`](docs/harrier_270m_to_120m_mission.md): preserve `d=640`, prune layers and vocabulary deliberately, validate fresh baselines, then add task-aware distillation.

See the visual experiment dashboard at [ravisankarg.github.io/PrunedHarrier](https://ravisankarg.github.io/PrunedHarrier/).

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

Background top-50 expansion, defaulting to `50K` rows per language:

```bash
bash scripts/start_top50_unlabeled_download.sh
```

The launcher writes `data/processed/unlabeled_top50_50k.jsonl`, seeds it from the existing top-20 file if available, then continues in the background. Override defaults with environment variables such as `PER_LANGUAGE=25000` or `OUT=data/processed/unlabeled_top50_25k.jsonl`.

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
  --run-mteb-during-training \
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
  --run-mteb-during-training \
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

For the larger real-data phase-2 corpus, use the balanced builder. It combines score-filtered STS-MT, Amazon Reviews, Mr. TyDi, TyDi QA, MLDR, and MIRACL. It writes exactly `50K` semantic-similarity and retrieval rows, `50,004` classification and clustering rows, and uses marked real-row repetition only where a retrieval language has fewer unique examples than its equal quota. It never adds prompted fallback rows.

```bash
HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN" \
  .venv/bin/python scripts/19_build_balanced_task_contrastive.py \
  --out data/processed/task_contrastive_balanced_50k_corrected.jsonl \
  --target-per-task 50000 \
  --force \
  --cache-dir .cache
```

The builder keeps STS-MT training pairs with `similarity_score >= 3.0`, removes exact self-pairs, and uses reverse query/document orientation only when needed to reach the quota. It records the source score and writes a manifest. Validate every rebuilt corpus before training:

```bash
.venv/bin/python scripts/20_validate_task_contrastive_data.py \
  --jsonl data/processed/task_contrastive_balanced_50k_corrected.jsonl \
  --target-per-task 50000
```

The builder globally shuffles the assembled rows before writing. This is required because the trainer's bounded shuffle buffer cannot mix a 50K-row semantic block with the later classification, clustering, and retrieval blocks. The validator rejects long single-task runs and a task-homogeneous prefix.

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
  --student runs/pca_teacher_384_distill_phase2_pairwise_0p20_from_002500/checkpoint-001500 \
  --teacher models/teacher_harrier_0_6b \
  --train-jsonl data/processed/task_contrastive_balanced_50k_corrected.jsonl \
  --output-dir runs/phase2_task_contrastive \
  --max-length 512 \
  --batch-size 16 \
  --grad-accum 8 \
  --lr 2e-6 \
  --lr-schedule constant \
  --warmup-steps 100 \
  --max-steps 500 \
  --contrastive-weight 1.0 \
  --contrastive-temperature 0.05 \
  --teacher-weight 0.0 \
  --similarity-weight 0.0 \
  --cosine-weight 0.0 \
  --layerwise-weight 0.0 \
  --student-device cuda \
  --student-dtype bf16 \
  --teacher-device cpu \
  --teacher-dtype bf16 \
  --save-every 100 \
  --max-running-loss 8.0
```

MTEB is intentionally not run during training. Evaluate a saved checkpoint separately in Step 6 after training completes. The `--run-mteb-during-training` flag is available only for an explicit opt-in smoke run.

Resume behavior: `--resume` loads `runs/phase2_task_contrastive/checkpoint-latest`.

The displayed loss is normalized per microbatch and is comparable across different `--grad-accum` values. Do not interpret the old pre-fix progress display as a raw loss: it summed the microbatch losses and made `grad-accum 8` appear approximately eight times larger.

## Data and Training Guardrails

The previous task run exposed two failures that are now guarded:

- The original STS builder ignored `similarity_score`, so unrelated sentence pairs were treated as positives. The corrected builder filters scores and rejects missing or weak STS metadata.
- The original balanced file was task-block ordered. A bounded training shuffle therefore created an unintended curriculum: semantic similarity first, then classification/clustering/retrieval. The corrected builder globally shuffles rows and validates task mixing.
- The trainer did not persist normalized contrastive loss, and its progress display summed losses across gradient accumulation. The trainer now writes `metrics.jsonl`, reports normalized loss, supports `--max-running-loss`, and keeps MTEB disabled unless explicitly requested.

Before any long run, validate the corpus, run a `500`-step pilot, and inspect normalized loss plus the saved checkpoint. Stop and investigate if normalized rolling loss rises above the configured threshold for `100` steps, becomes NaN, or the checkpoint loses the baseline task scores. Do not resume from a checkpoint produced after a triggered loss guard; restart from the last known-good checkpoint with a fresh optimizer.

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
