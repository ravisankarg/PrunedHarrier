import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_jsonl(path, shuffle_buffer=0, seed=13):
    rng = random.Random(seed)
    if shuffle_buffer <= 1:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    buf = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            buf.append(json.loads(line))
            if len(buf) >= shuffle_buffer:
                rng.shuffle(buf)
                while buf:
                    yield buf.pop()
    rng.shuffle(buf)
    while buf:
        yield buf.pop()


def append_jsonl(path, rows):
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def last_token_pool(last_hidden_state, attention_mask):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]


def encode_transformer(model, tokenizer, texts, device, max_length):
    batch = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
    return F.normalize(pooled, p=2, dim=1)


def pick_dtype(device, force_fp32=False, requested="auto"):
    if force_fp32:
        return torch.float32
    if requested == "fp32":
        return torch.float32
    if requested == "fp16":
        if device == "cpu":
            raise ValueError("fp16 is only supported for this script on CUDA")
        return torch.float16
    if requested == "bf16":
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA is available but this GPU/PyTorch build does not report BF16 support")
        return torch.bfloat16
    if device == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def set_cache_env(base_dir=".cache"):
    base = Path(base_dir)
    os.environ["HF_HOME"] = str(base / "huggingface")
    os.environ["HF_DATASETS_CACHE"] = str(base / "huggingface" / "datasets")
    os.environ["MTEB_CACHE"] = str(base / "mteb")
