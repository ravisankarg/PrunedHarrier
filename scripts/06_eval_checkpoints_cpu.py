import argparse
import csv
import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

from common import ensure_dir, read_json, set_cache_env


def checkpoint_step(path):
    match = re.search(r"checkpoint-(\d+)$", Path(path).name)
    return int(match.group(1)) if match else -1


def find_checkpoints(run_dir, steps):
    run_dir = Path(run_dir)
    checkpoints = sorted(run_dir.glob("checkpoint-[0-9]*"), key=checkpoint_step)
    if steps:
        wanted = {int(step) for step in steps}
        checkpoints = [ckpt for ckpt in checkpoints if checkpoint_step(ckpt) in wanted]
    return checkpoints


def load_status(output_dir):
    status_path = Path(output_dir) / "mteb_slice_status.json"
    if not status_path.exists():
        return None
    with status_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_rows(checkpoint, output_dir):
    rows = []
    for path in Path(output_dir).rglob("*.json"):
        if path.name in {"model_meta.json", "mteb_slice_status.json"}:
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        task_name = data.get("task_name") or path.stem
        scores = data.get("scores", {})
        for split, entries in scores.items():
            if isinstance(entries, dict):
                entries = [entries]
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                main_score = entry.get("main_score")
                if main_score is None:
                    continue
                rows.append(
                    {
                        "checkpoint": str(checkpoint),
                        "step": checkpoint_step(checkpoint),
                        "task": task_name,
                        "split": split,
                        "subset": entry.get("hf_subset", ""),
                        "languages": ",".join(entry.get("languages", []) or []),
                        "main_score": main_score,
                        "cosine_spearman": entry.get("cosine_spearman", ""),
                        "accuracy": entry.get("accuracy", ""),
                        "v_measure": entry.get("v_measure", ""),
                        "ndcg_at_10": entry.get("ndcg_at_10", ""),
                    }
                )
    return rows


def summarize(rows):
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(float(row["main_score"]))
    summary = {}
    for task, values in sorted(by_task.items()):
        summary[task] = sum(values) / len(values)
    if summary:
        summary["mean_all_tasks"] = sum(summary.values()) / len(summary)
    return summary


def write_summary(out_dir, all_rows):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    csv_path = out_dir / "checkpoint_scores.csv"
    json_path = out_dir / "checkpoint_scores_summary.json"

    fields = [
        "checkpoint",
        "step",
        "task",
        "split",
        "subset",
        "languages",
        "main_score",
        "cosine_spearman",
        "accuracy",
        "v_measure",
        "ndcg_at_10",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    by_step = {}
    for row in all_rows:
        by_step.setdefault(str(row["step"]), []).append(row)
    summary = {step: summarize(rows) for step, rows in sorted(by_step.items(), key=lambda item: int(item[0]))}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return csv_path, json_path, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/phase1_unlabeled_distill")
    parser.add_argument("--config", default="configs/mteb_slice.json")
    parser.add_argument("--output-dir", default="eval_results/phase1_checkpoint_cpu")
    parser.add_argument("--steps", default="", help="Comma-separated checkpoint steps, for example 2000,8000,10000.")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    set_cache_env(args.cache_dir)
    cfg = read_json(args.config)
    steps = [item.strip() for item in args.steps.split(",") if item.strip()]
    checkpoints = find_checkpoints(args.run_dir, steps)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {args.run_dir} for steps={steps or 'all'}")

    print("Selected checkpoints:")
    for ckpt in checkpoints:
        print(f"  step {checkpoint_step(ckpt)}: {ckpt}")
    if args.list_only:
        return

    import mteb

    tasks = mteb.get_tasks(tasks=cfg["tasks"], languages=cfg.get("languages"))
    if len(tasks) == 0:
        raise ValueError(f"No MTEB tasks resolved for tasks={cfg['tasks']} languages={cfg.get('languages')}")
    print("Resolved tasks:", ", ".join(task.metadata.name for task in tasks))

    all_rows = []
    output_root = Path(args.output_dir)
    ensure_dir(output_root)

    for ckpt in checkpoints:
        step = checkpoint_step(ckpt)
        ckpt_out = output_root / f"step-{step:06d}"
        status = load_status(ckpt_out)
        if status and status.get("ok") and not args.overwrite:
            print(f"step {step}: existing successful eval, reusing {ckpt_out}")
            all_rows.extend(score_rows(ckpt, ckpt_out))
            continue

        print(f"step {step}: loading {ckpt} on CPU")
        model = SentenceTransformer(str(ckpt), device="cpu")
        ensure_dir(ckpt_out)
        try:
            evaluation = mteb.MTEB(tasks=tasks)
            evaluation.run(
                model,
                output_folder=str(ckpt_out),
                overwrite_results=True,
                raise_error=False,
                verbosity=cfg.get("verbosity", 1),
            )
            status = {
                "ok": True,
                "model": str(ckpt),
                "device": "cpu",
                "tasks": cfg["tasks"],
                "languages": cfg.get("languages"),
                "resolved_tasks": [task.metadata.name for task in tasks],
            }
        except Exception as exc:
            status = {"ok": False, "model": str(ckpt), "device": "cpu", "error": repr(exc)}
            print(f"step {step}: eval failed: {exc}")
        with (ckpt_out / "mteb_slice_status.json").open("w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
            f.write("\n")
        all_rows.extend(score_rows(ckpt, ckpt_out))

    csv_path, json_path, summary = write_summary(output_root, all_rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    for step, values in summary.items():
        metric = values.get("mean_all_tasks")
        if metric is not None:
            print(f"step {step}: mean_all_tasks={metric:.6f} {values}")


if __name__ == "__main__":
    main()
