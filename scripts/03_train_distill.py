import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from common import encode_transformer, ensure_dir, iter_jsonl, last_token_pool, pick_dtype, read_json, set_cache_env, write_json


def make_batches(path, batch_size, shuffle_buffer, seed):
    rows = iter_jsonl(path, shuffle_buffer=shuffle_buffer, seed=seed)
    batch = []
    for row in rows:
        if "query" in row and "positive" in row:
            batch.append(row)
        elif "text" in row:
            batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def endless_batches(path, batch_size, shuffle_buffer, seed):
    epoch = 0
    while True:
        yield from make_batches(path, batch_size, shuffle_buffer, seed + epoch)
        epoch += 1


def batch_texts(rows):
    if rows and "query" in rows[0] and "positive" in rows[0]:
        queries = [r["query"] for r in rows]
        positives = [r["positive"] for r in rows]
        return queries + positives, True
    return [r["text"] for r in rows], False


def distill_loss(student_emb, teacher_emb, projected_student_emb=None):
    s_sim = student_emb @ student_emb.T
    t_sim = teacher_emb @ teacher_emb.T
    sim = F.mse_loss(s_sim, t_sim)
    if student_emb.shape[-1] == teacher_emb.shape[-1]:
        cosine = 1.0 - F.cosine_similarity(student_emb, teacher_emb, dim=-1).mean()
        return cosine + sim
    if projected_student_emb is not None:
        projected_student_emb = F.normalize(projected_student_emb, p=2, dim=-1)
        cosine = 1.0 - F.cosine_similarity(projected_student_emb, teacher_emb, dim=-1).mean()
        return cosine + sim
    return sim


def contrastive_loss(student_emb, pair_batch_size, temperature=0.05):
    q = student_emb[:pair_batch_size]
    p = student_emb[pair_batch_size:]
    logits = (q @ p.T) / temperature
    labels = torch.arange(pair_batch_size, device=student_emb.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5


def encode_with_hidden_states(model, tokenizer, texts, device, max_length):
    batch = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch, output_hidden_states=True)
    pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
    return F.normalize(pooled, p=2, dim=1), outputs.hidden_states


def layerwise_mse_loss(student_hidden, teacher_hidden, layer_map, projector):
    losses = []
    for student_i, teacher_i in enumerate(layer_map):
        s = student_hidden[student_i + 1].float()
        t = teacher_hidden[teacher_i + 1].to(s.device).float()
        if projector is not None:
            s = projector(s)
        losses.append(F.mse_loss(F.normalize(s, p=2, dim=-1), F.normalize(t, p=2, dim=-1)))
    return torch.stack(losses).mean()


def save_checkpoint(student, tokenizer, output_dir, step, manifest):
    ckpt = Path(output_dir) / f"checkpoint-{step:06d}"
    ensure_dir(ckpt)
    student.save_pretrained(ckpt, safe_serialization=True)
    tokenizer.save_pretrained(ckpt)
    write_json(ckpt / "train_manifest.json", manifest)
    return ckpt


def save_training_state(output_dir, step, optimizer, scheduler, scaler, projection, layer_projection):
    state_path = Path(output_dir) / f"checkpoint-{step:06d}" / "trainer_state.pt"
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if projection is not None:
        state["projection"] = projection.state_dict()
    if layer_projection is not None:
        state["layer_projection"] = layer_projection.state_dict()
    torch.save(state, state_path)


def update_latest(output_dir, ckpt):
    latest = Path(output_dir) / "checkpoint-latest"
    tmp = Path(output_dir) / "checkpoint-latest.tmp"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    rel = os.path.relpath(ckpt, latest.parent)
    tmp.symlink_to(rel, target_is_directory=True)
    os.replace(tmp, latest)


def maybe_run_mteb(script_path, ckpt, output_dir, step, cfg):
    if not cfg:
        return
    eval_dir = Path(output_dir) / "mteb_slice" / f"step-{step:06d}"
    cmd = [
        os.environ.get("PYTHON", sys.executable),
        str(script_path),
        "--model",
        str(ckpt),
        "--config",
        cfg,
        "--output-dir",
        str(eval_dir),
    ]
    print("Running MTEB slice:", " ".join(cmd))
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", default="microsoft/harrier-oss-v1-0.6b")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--layerwise-weight", type=float, default=0.0)
    parser.add_argument("--layer-map-config", default="configs/student_12l_384.json")
    parser.add_argument("--teacher-device", default="cpu")
    parser.add_argument("--student-device", default="cuda")
    parser.add_argument("--student-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--shuffle-buffer", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--mteb-slice-config")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--force-fp32", action="store_true")
    parser.add_argument("--teacher-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument("--resume", action="store_true", help="Resume from output-dir/checkpoint-latest if present.")
    args = parser.parse_args()

    set_cache_env(args.cache_dir)
    if args.student_device == "cuda" and not torch.cuda.is_available():
        args.student_device = "cpu"

    ensure_dir(args.output_dir)
    torch.manual_seed(args.seed)

    start_step = 0
    resume_ckpt = Path(args.output_dir) / "checkpoint-latest"
    if args.resume and resume_ckpt.exists():
        print(f"Resuming student weights from {resume_ckpt}")
        args.student = str(resume_ckpt)

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student_dtype = pick_dtype(args.student_device, args.force_fp32, args.student_dtype)
    print(f"Loading student on {args.student_device} dtype={student_dtype}")
    student = AutoModel.from_pretrained(args.student, torch_dtype=student_dtype).to(args.student_device)
    student.train()
    student.config.use_cache = False
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    teacher_dtype = pick_dtype(args.teacher_device, args.force_fp32, args.teacher_dtype)
    print(f"Loading teacher on {args.teacher_device} dtype={teacher_dtype}")
    teacher = AutoModel.from_pretrained(args.teacher, torch_dtype=teacher_dtype).to(args.teacher_device)
    teacher.eval()
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)

    projection = None
    if student.config.hidden_size != teacher.config.hidden_size:
        projection = nn.Linear(student.config.hidden_size, teacher.config.hidden_size, bias=False).to(args.student_device)
        print(f"Using train-time projection head {student.config.hidden_size}->{teacher.config.hidden_size}")

    layer_projection = None
    layer_map = None
    if args.layerwise_weight > 0:
        cfg = read_json(args.layer_map_config)
        layer_map = cfg["layer_map"]
        if len(layer_map) != student.config.num_hidden_layers:
            raise ValueError("layer_map length must match student num_hidden_layers")
        if student.config.hidden_size != teacher.config.hidden_size:
            layer_projection = nn.Linear(student.config.hidden_size, teacher.config.hidden_size, bias=False).to(args.student_device)
            print(f"Using layerwise projection head {student.config.hidden_size}->{teacher.config.hidden_size}")

    params = (
        list(student.parameters())
        + ([] if projection is None else list(projection.parameters()))
        + ([] if layer_projection is None else list(layer_projection.parameters()))
    )
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.student_device == "cuda" and student_dtype == torch.float16))
    state_path = Path(args.student) / "trainer_state.pt"
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        if projection is not None and "projection" in state:
            projection.load_state_dict(state["projection"])
        if layer_projection is not None and "layer_projection" in state:
            layer_projection.load_state_dict(state["layer_projection"])
        start_step = int(state.get("step", 0))
        print(f"Loaded trainer state at step {start_step}")
    elif args.resume and resume_ckpt.exists():
        name = Path(args.student).resolve().name
        if name.startswith("checkpoint-"):
            try:
                start_step = int(name.split("-")[-1])
            except ValueError:
                start_step = 0
        print(f"No trainer_state.pt found; continuing weights-only from step {start_step}")

    data_iter = endless_batches(args.train_jsonl, args.batch_size, args.shuffle_buffer, args.seed)
    progress = tqdm(range(start_step + 1, args.max_steps + 1), desc="train")
    optimizer.zero_grad(set_to_none=True)

    running = []
    script_path = Path(__file__).parent / "05_eval_mteb_slice.py"
    manifest = vars(args).copy()

    for step in progress:
        step_loss = 0.0
        for _ in range(args.grad_accum):
            rows = next(data_iter)
            texts, is_pair_batch = batch_texts(rows)
            with torch.no_grad():
                if args.layerwise_weight > 0:
                    teacher_emb, teacher_hidden = encode_with_hidden_states(teacher, tokenizer, texts, args.teacher_device, args.max_length)
                    teacher_hidden = tuple(h.to(args.student_device) for h in teacher_hidden)
                else:
                    teacher_emb = encode_transformer(teacher, tokenizer, texts, args.teacher_device, args.max_length)
                    teacher_hidden = None
                teacher_emb = teacher_emb.to(args.student_device)

            with torch.cuda.amp.autocast(enabled=(args.student_device == "cuda"), dtype=student_dtype):
                if args.layerwise_weight > 0:
                    student_emb, student_hidden = encode_with_hidden_states(student, tokenizer, texts, args.student_device, args.max_length)
                else:
                    student_emb = encode_transformer(student, tokenizer, texts, args.student_device, args.max_length)
                    student_hidden = None
                projected = projection(student_emb) if projection is not None else None
                projected = projected.float() if projected is not None else None
                loss = args.teacher_weight * distill_loss(student_emb.float(), teacher_emb.float(), projected)
                if args.layerwise_weight > 0:
                    loss = loss + args.layerwise_weight * layerwise_mse_loss(
                        student_hidden,
                        teacher_hidden,
                        layer_map,
                        layer_projection,
                    )
                if is_pair_batch and args.contrastive_weight > 0:
                    loss = loss + args.contrastive_weight * contrastive_loss(student_emb.float(), len(rows))
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            step_loss += float(loss.detach().cpu()) * args.grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running.append(step_loss)
        if len(running) > 100:
            running.pop(0)
        progress.set_postfix(loss=f"{sum(running) / len(running):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt = save_checkpoint(student, tokenizer, args.output_dir, step, manifest)
            save_training_state(args.output_dir, step, optimizer, scheduler, scaler, projection, layer_projection)
            update_latest(args.output_dir, ckpt)
            print(f"Saved {ckpt}")
            if step % args.eval_every == 0:
                maybe_run_mteb(script_path, ckpt, args.output_dir, step, args.mteb_slice_config)

    write_json(Path(args.output_dir) / "done.json", {"status": "complete", "max_steps": args.max_steps})


if __name__ == "__main__":
    main()
