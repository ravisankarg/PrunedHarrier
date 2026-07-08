import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from common import ensure_dir, read_json, set_cache_env


def text_value(ex, candidates):
    for key in candidates:
        val = ex.get(key)
        if isinstance(val, str) and len(val.strip()) > 0:
            return val.strip()
    return None


TASK_PROMPTS = {
    "retrieval": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "semantic_similarity": "Instruct: Retrieve semantically similar text\nQuery: ",
    "classification": "Instruct: Classify the sentiment or topic of the text\nQuery: ",
    "clustering": "Instruct: Group texts with similar topic or label\nQuery: ",
}


def write_row(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def add_universal_fallback_from_unlabeled(out_f, unlabeled_path, per_task_language, langs, seed, done):
    if not unlabeled_path or not Path(unlabeled_path).exists():
        return
    rng = random.Random(seed)
    by_lang = {lang["code"]: [] for lang in langs}
    with open(unlabeled_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lang = row.get("lang")
            if lang in by_lang and len(by_lang[lang]) < per_task_language * 2:
                by_lang[lang].append(row["text"])

    for lang, texts in by_lang.items():
        rng.shuffle(texts)
        for task, prompt in TASK_PROMPTS.items():
            existing_count = sum(1 for item in done if item[0] == lang and item[1] == task)
            if existing_count >= per_task_language:
                print(f"{lang}/{task}: already has {existing_count}, skipping fallback", flush=True)
                continue
            for idx, text in enumerate(texts[:per_task_language]):
                positive = text
                if task in {"classification", "clustering"} and len(texts) > 1:
                    positive = texts[(idx + 1) % len(texts)]
                row = {
                    "task": task,
                    "lang": lang,
                    "query": prompt + text,
                    "positive": positive,
                    "source": "unlabeled_prompted_fallback",
                }
                key = (row["lang"], row["task"], row["query"], row["positive"])
                if key in done:
                    continue
                write_row(out_f, row)
                done.add(key)


def try_sts(out_f, lang, spec, limit):
    configs = [lang, f"{lang}-{lang}", "en"]
    written = 0
    for cfg in configs:
        try:
            ds = load_dataset(spec["dataset"], cfg, split="train", streaming=True)
            break
        except Exception:
            ds = None
    if ds is None:
        return 0
    for ex in ds:
        a = text_value(ex, ["sentence1", "sent1", "text1", "query"])
        b = text_value(ex, ["sentence2", "sent2", "text2", "positive"])
        if not a or not b:
            continue
        write_row(out_f, {
            "task": "semantic_similarity",
            "lang": lang,
            "query": spec["query_prompt"] + a,
            "positive": b,
            "source": spec["dataset"],
        })
        written += 1
        if written >= limit:
            break
    return written


def try_label_dataset(out_f, lang, task, spec, limit):
    configs = [lang, f"{lang}_Latn", "en"]
    written = 0
    for cfg in configs:
        try:
            ds = load_dataset(spec["dataset"], cfg, split="train", streaming=True)
            break
        except Exception:
            ds = None
    if ds is None:
        return 0
    label_buckets = {}
    for ex in ds:
        text = text_value(ex, ["text", "sentence", "content", "review_body", "title"])
        label = ex.get("label") if "label" in ex else ex.get("labels")
        if text is None or label is None:
            continue
        label_buckets.setdefault(str(label), []).append(text)
        if sum(len(v) for v in label_buckets.values()) >= limit * 3:
            break
    for label, texts in label_buckets.items():
        for i in range(0, len(texts) - 1, 2):
            write_row(out_f, {
                "task": task,
                "lang": lang,
                "query": spec["query_prompt"] + texts[i],
                "positive": texts[i + 1],
                "label": label,
                "source": spec["dataset"],
            })
            written += 1
            if written >= limit:
                return written
    return written


def try_retrieval(out_f, lang, spec, limit):
    # MIRACL dataset packaging has changed across versions; this reader is deliberately
    # conservative and skips a language if the expected query-positive fields are absent.
    written = 0
    configs = [lang, f"{lang}/train", "en"]
    for cfg in configs:
        try:
            ds = load_dataset(spec["dataset"], cfg, split="train", streaming=True)
            break
        except Exception:
            ds = None
    if ds is None:
        return 0
    for ex in ds:
        query = text_value(ex, ["query", "question", "text"])
        positive = text_value(ex, ["positive", "positive_passage", "passage", "document"])
        if not query or not positive:
            continue
        write_row(out_f, {
            "task": "retrieval",
            "lang": lang,
            "query": spec["query_prompt"] + query,
            "positive": positive,
            "source": spec["dataset"],
        })
        written += 1
        if written >= limit:
            break
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default="configs/top20_languages.json")
    parser.add_argument("--sources", default="configs/task_data_sources.json")
    parser.add_argument("--per-task-language", type=int, default=5000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--unlabeled-fallback", default="data/processed/unlabeled_top20_25k.jsonl")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["fallback-only", "hf-then-fallback"],
        default="fallback-only",
        help="fallback-only is robust with datasets>=5; hf-then-fallback also tries configured HF sources.",
    )
    args = parser.parse_args()

    set_cache_env(args.cache_dir)
    langs = read_json(args.languages)
    sources = read_json(args.sources)
    out = Path(args.out)
    ensure_dir(out.parent)
    done = set()
    mode = "w" if args.force or not out.exists() else "a"
    if out.exists() and not args.force:
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((row.get("lang"), row.get("task"), row.get("query"), row.get("positive")))

    with out.open(mode, encoding="utf-8") as f:
        if args.mode == "hf-then-fallback":
            for lang in langs:
                code = lang["code"]
                for task, spec in sources.items():
                    existing_count = sum(1 for item in done if item[0] == code and item[1] == task)
                    if existing_count >= args.per_task_language:
                        print(f"{code}/{task}: already has {existing_count}, skipping", flush=True)
                        continue
                    print(f"{code}/{task}: downloading up to {args.per_task_language}", flush=True)
                    if task == "semantic_similarity":
                        n = try_sts(f, code, spec, args.per_task_language)
                    elif task in {"classification", "clustering"}:
                        n = try_label_dataset(f, code, task, spec, args.per_task_language)
                    elif task == "retrieval":
                        n = try_retrieval(f, code, spec, args.per_task_language)
                    else:
                        n = 0
                    print(f"{code}/{task}: wrote {n}", flush=True)
        print("Adding prompted fallback examples from unlabeled text", flush=True)
        add_universal_fallback_from_unlabeled(f, args.unlabeled_fallback, args.per_task_language, langs, args.seed, done)


if __name__ == "__main__":
    main()
