# Harrier 270M To 120M Mission

Goal: prune `microsoft/harrier-oss-v1-270m` to about `120M` parameters while preserving multilingual embedding quality across retrieval, STS, clustering, and classification.

## Design Decision

Use layer pruning plus vocabulary pruning. Do not slice hidden size.

The local config check for `microsoft/harrier-oss-v1-270m` reports:

- `model_type`: `gemma3_text`
- `vocab_size`: `262144`
- `hidden_size`: `640`
- `num_hidden_layers`: `18`
- `intermediate_size`: `2048`
- `num_attention_heads`: `4`
- `num_key_value_heads`: `1`
- `head_dim`: `256`

Full vocabulary alone is about `262144 * 640 = 167.8M` embedding parameters, so a `120M` target cannot keep the full tokenizer. The first target is therefore a full-width `12` layer body plus the largest language-prioritized vocabulary that fits the parameter budget.

## Mission Stages

1. Establish teacher baseline on the exact multilingual MTEB slice.
2. Build language-prioritized token IDs from the support-language corpus.
3. Prepare a `12L/640d` student from exact copied layers and pruned vocab rows.
4. Smoke-test SentenceTransformer export and STS before any training.
5. Distill on multilingual unlabeled text with final-embedding similarity/cosine loss first.
6. Add task-aware contrastive training for retrieval, STS, classification, and clustering.
7. Run checkpoint-by-checkpoint MTEB and only promote checkpoints that preserve the task mix.

## Commands

Set caches:

```bash
export HF_HOME=$PWD/.cache/huggingface
export HF_DATASETS_CACHE=$PWD/.cache/huggingface/datasets
export MTEB_CACHE=$PWD/.cache/mteb
export PYTHON=$PWD/.venv/bin/python
```

Download multilingual text after replacing `configs/top20_languages.json` with the language list you want:

```bash
.venv/bin/python scripts/02_download_unlabeled_text.py \
  --languages configs/top20_languages.json \
  --per-language 25000 \
  --max-chars 2000 \
  --out data/processed/harrier270_unlabeled_langs_25k.jsonl
```

Build ordered vocabulary priority from that corpus:

```bash
.venv/bin/python scripts/13_select_vocab_for_languages.py \
  --tokenizer microsoft/harrier-oss-v1-270m \
  --corpus data/processed/harrier270_unlabeled_langs_25k.jsonl \
  --languages configs/top20_languages.json \
  --out data/processed/harrier270_vocab_priority.json
```

Prepare the 120M target model:

```bash
.venv/bin/python scripts/14_prepare_harrier_270m_120m.py \
  --teacher microsoft/harrier-oss-v1-270m \
  --plan configs/harrier_270m_120m.json \
  --vocab-priority data/processed/harrier270_vocab_priority.json \
  --out-dir models/harrier_270m_12l_120m_vocab_pruned
```

Smoke-evaluate teacher and fresh student:

```bash
.venv/bin/python scripts/05_eval_mteb_slice.py \
  --model microsoft/harrier-oss-v1-270m \
  --config configs/mteb_smoke.json \
  --output-dir eval_results/harrier270_teacher_smoke

.venv/bin/python scripts/05_eval_mteb_slice.py \
  --model models/harrier_270m_12l_120m_vocab_pruned \
  --config configs/mteb_smoke.json \
  --output-dir eval_results/harrier270_120m_fresh_smoke
```

Run phase 1 similarity distillation:

```bash
.venv/bin/python scripts/03_train_distill.py \
  --student models/harrier_270m_12l_120m_vocab_pruned \
  --teacher microsoft/harrier-oss-v1-270m \
  --train-jsonl data/processed/harrier270_unlabeled_langs_25k.jsonl \
  --output-dir runs/harrier270_120m_phase1_similarity \
  --max-length 512 \
  --batch-size 8 \
  --grad-accum 4 \
  --lr 2e-5 \
  --max-steps 10000 \
  --similarity-weight 1.0 \
  --cosine-weight 1.0 \
  --layerwise-weight 0.0 \
  --layer-map-config configs/harrier_270m_120m.json \
  --student-device cuda \
  --student-dtype bf16 \
  --teacher-device cuda \
  --teacher-dtype bf16 \
  --eval-every 1000 \
  --save-every 1000 \
  --mteb-slice-config configs/mteb_smoke.json \
  --resume
```

Layerwise token MSE is intentionally off for the vocab-pruned model because the pruned BPE tokenizer can split text differently from the teacher tokenizer. Final pooled embedding losses remain valid because teacher and student encode the same raw text with their own tokenizers.

Build task-aware contrastive rows:

```bash
.venv/bin/python scripts/04_download_task_contrastive_data.py \
  --mode hf-then-fallback \
  --languages configs/top20_languages.json \
  --sources configs/task_data_sources.json \
  --per-task-language 5000 \
  --unlabeled-fallback data/processed/harrier270_unlabeled_langs_25k.jsonl \
  --out data/processed/harrier270_task_contrastive.jsonl
```

Run phase 2 task-aware training:

```bash
.venv/bin/python scripts/03_train_distill.py \
  --student runs/harrier270_120m_phase1_similarity/checkpoint-latest \
  --teacher microsoft/harrier-oss-v1-270m \
  --train-jsonl data/processed/harrier270_task_contrastive.jsonl \
  --output-dir runs/harrier270_120m_phase2_task \
  --max-length 512 \
  --batch-size 8 \
  --grad-accum 4 \
  --lr 1e-5 \
  --max-steps 10000 \
  --similarity-weight 0.5 \
  --cosine-weight 0.5 \
  --contrastive-weight 1.0 \
  --student-device cuda \
  --student-dtype bf16 \
  --teacher-device cuda \
  --teacher-dtype bf16 \
  --eval-every 1000 \
  --save-every 1000 \
  --mteb-slice-config configs/mteb_smoke.json \
  --resume
```

Run the multilingual task slice on selected checkpoints:

```bash
.venv/bin/python scripts/05_eval_mteb_slice.py \
  --model runs/harrier270_120m_phase2_task/checkpoint-latest \
  --config configs/mteb_multilingual_v2_core.json \
  --output-dir eval_results/harrier270_120m_phase2_multilingual_core
```

## Promotion Gate

Promote a checkpoint only if:

- SentenceTransformer loads and reports `640` embedding dimensions.
- Fresh pruning does not catastrophically collapse STS smoke.
- Phase 1 improves the fresh pruned baseline.
- Phase 2 improves or preserves at least three of the four task families.
- Retrieval is judged separately from STS because retrieval often needs task-aware prompting and contrastive rows.
