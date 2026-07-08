import argparse
import gc
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from common import ensure_dir, read_json, set_cache_env


def clean_text(text, max_chars):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) < 80:
        return None
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def wikipedia_config(lang):
    return f"20231101.{lang}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="configs/top20_languages.json")
    parser.add_argument("--per-language", type=int, default=25000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()

    set_cache_env(args.cache_dir)
    langs = read_json(args.languages)
    out = Path(args.out)
    ensure_dir(out.parent)

    existing = set()
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    existing.add((row.get("lang"), row.get("text")))
                except json.JSONDecodeError:
                    continue

    with out.open("a", encoding="utf-8") as f:
        for lang in langs:
            code = lang["code"]
            already = sum(1 for item_lang, _ in existing if item_lang == code)
            if already >= args.per_language:
                print(f"{code}: already has {already}, skipping")
                continue

            needed = args.per_language - already
            print(f"{code}: downloading {needed} examples from {args.dataset}/{wikipedia_config(code)}")
            try:
                ds = load_dataset(args.dataset, wikipedia_config(code), split=args.split, streaming=True)
            except Exception as exc:
                print(f"{code}: failed to open dataset: {exc}")
                continue

            count = 0
            pbar = tqdm(total=needed, desc=code)
            for ex in ds:
                text = clean_text(ex.get("text") or ex.get("title") or "", args.max_chars)
                if not text:
                    continue
                key = (code, text)
                if key in existing:
                    continue
                row = {
                    "text": text,
                    "lang": code,
                    "source": args.dataset,
                    "task": "unlabeled",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing.add(key)
                count += 1
                pbar.update(1)
                if count >= needed:
                    break
            pbar.close()
            del ds
            gc.collect()
            print(f"{code}: wrote {count}")


if __name__ == "__main__":
    main()
