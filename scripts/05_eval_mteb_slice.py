import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from common import ensure_dir, read_json, set_cache_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/mteb_slice.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=".cache")
    args = parser.parse_args()

    set_cache_env(args.cache_dir)
    cfg = read_json(args.config)
    ensure_dir(args.output_dir)

    try:
        import mteb

        model = SentenceTransformer(args.model, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
        tasks = mteb.get_tasks(tasks=cfg["tasks"], languages=cfg.get("languages"))
        if len(tasks) == 0:
            raise ValueError(f"No MTEB tasks resolved for tasks={cfg['tasks']} languages={cfg.get('languages')}")
        evaluation = mteb.MTEB(tasks=tasks)
        evaluation.run(
            model,
            output_folder=args.output_dir,
            overwrite_results=True,
            raise_error=False,
            verbosity=cfg.get("verbosity", 1),
        )
        status = {
            "ok": True,
            "model": args.model,
            "tasks": cfg["tasks"],
            "languages": cfg.get("languages"),
            "resolved_tasks": [task.metadata.name for task in tasks],
        }
    except Exception as exc:
        status = {"ok": False, "model": args.model, "error": repr(exc)}
        print(f"MTEB slice failed: {exc}")

    with open(Path(args.output_dir) / "mteb_slice_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
