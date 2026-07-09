import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from common import ensure_dir, read_json, set_cache_env


DONE_KEYS = None


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

TOP20_TO_STS17 = {
    "en": ["en-en"],
    "zh": [],
    "hi": [],
    "es": ["es-es", "es-en"],
    "ar": ["ar-ar", "en-ar"],
    "fr": ["fr-en"],
    "bn": [],
    "pt": [],
    "ru": [],
    "ur": [],
    "id": [],
    "de": ["en-de"],
    "ja": [],
    "sw": [],
    "mr": [],
    "te": [],
    "tr": ["en-tr"],
    "ta": [],
    "vi": [],
    "ko": ["ko-ko"],
}

TOP20_TO_AMAZON = {
    "en": "en",
    "zh": "zh",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "ja": "ja",
}

TOP20_TO_MASAKHA = {
    "en": "eng",
    "fr": "fra",
    "sw": "swa",
}

TOP20_TO_MIRACL = {
    "en": "en",
    "zh": "zh",
    "hi": "hi",
    "es": "es",
    "ar": "ar",
    "fr": "fr",
    "bn": "bn",
    "ru": "ru",
    "id": "id",
    "de": "de",
    "ja": "ja",
    "sw": "sw",
    "te": "te",
    "ko": "ko",
}


def write_row(f, row):
    global DONE_KEYS
    key = (row.get("lang"), row.get("task"), row.get("query"), row.get("positive"))
    if DONE_KEYS is not None:
        if key in DONE_KEYS:
            return
        DONE_KEYS.add(key)
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
            needed = per_task_language - existing_count
            written = 0
            for idx, text in enumerate(texts):
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
                written += 1
                if written >= needed:
                    break
            print(f"{lang}/{task}: fallback wrote {written}", flush=True)


def print_output_summary(path):
    by_source = {}
    by_task = {}
    by_bucket = {}
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            source = row.get("source", "")
            task = row.get("task", "")
            lang = row.get("lang", "")
            by_source[source] = by_source.get(source, 0) + 1
            by_task[task] = by_task.get(task, 0) + 1
            by_bucket[(lang, task)] = by_bucket.get((lang, task), 0) + 1

    print("Final output summary", flush=True)
    print(f"  rows: {sum(by_source.values())}", flush=True)
    print(f"  bad_json_rows: {bad}", flush=True)
    print(f"  buckets: {len(by_bucket)}", flush=True)
    if by_bucket:
        print(f"  bucket_min: {min(by_bucket.values())}", flush=True)
        print(f"  bucket_max: {max(by_bucket.values())}", flush=True)
    print("  by_source:", flush=True)
    for source, count in sorted(by_source.items(), key=lambda item: (-item[1], item[0])):
        print(f"    {source}: {count}", flush=True)
    print("  by_task:", flush=True)
    for task, count in sorted(by_task.items()):
        print(f"    {task}: {count}", flush=True)


def try_sts(out_f, lang, spec, limit):
    written = 0
    configs = TOP20_TO_STS17.get(lang, [])
    if not configs and spec["dataset"] == "mteb/sts17-crosslingual-sts":
        return 0
    if not configs:
        configs = [lang, f"{lang}-{lang}", "en"]
    for cfg in configs:
        try:
            split = "test" if spec["dataset"].startswith("mteb/") else "train"
            ds = load_dataset(spec["dataset"], cfg, split=split, streaming=True)
        except Exception as exc:
            print(f"{lang}/semantic_similarity/{cfg}: skip {exc}", flush=True)
            continue
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
                "subset": cfg,
            })
            written += 1
            if written >= limit:
                return written
    return written


def try_label_dataset(out_f, lang, task, spec, limit):
    if spec["dataset"] == "mteb/AmazonReviewsClassification":
        cfg = TOP20_TO_AMAZON.get(lang)
        if not cfg:
            return 0
        configs = [cfg]
        splits = ["validation", "test"]
    elif spec["dataset"] == "mteb/masakhanews":
        cfg = TOP20_TO_MASAKHA.get(lang)
        if not cfg:
            return 0
        configs = [cfg]
        splits = ["test"]
    else:
        configs = [lang, f"{lang}_Latn", "en"]
        splits = ["train"]
    written = 0
    label_buckets = {}
    for cfg in configs:
        for split in splits:
            try:
                ds = load_dataset(spec["dataset"], cfg, split=split, streaming=True)
            except Exception as exc:
                print(f"{lang}/{task}/{cfg}/{split}: skip {exc}", flush=True)
                continue
            for ex in ds:
                text = text_value(ex, ["text", "sentence", "content", "review_body", "headline", "title"])
                label = ex.get("label") if "label" in ex else ex.get("labels")
                if label is None:
                    label = ex.get("category")
                if text is None or label is None:
                    continue
                label_buckets.setdefault(str(label), []).append(text)
                if sum(len(v) for v in label_buckets.values()) >= limit * 3:
                    break
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
    written = 0
    if spec["dataset"] == "mteb/MIRACLRetrieval":
        cfg = TOP20_TO_MIRACL.get(lang)
        if not cfg:
            return 0
        try:
            queries = load_dataset(spec["dataset"], f"{cfg}-queries", split="dev", streaming=True)
            qrels = load_dataset(spec["dataset"], f"{cfg}-qrels", split="dev", streaming=True)
            corpus = load_dataset(spec["dataset"], f"{cfg}-corpus", split="dev", streaming=True)
        except Exception as exc:
            print(f"{lang}/retrieval/{cfg}: skip {exc}", flush=True)
            return 0

        needed_query_ids = {}
        needed_doc_ids = set()
        for ex in qrels:
            if ex.get("score", 0) <= 0:
                continue
            qid = str(ex.get("query-id"))
            did = str(ex.get("corpus-id"))
            if qid and did and qid not in needed_query_ids:
                needed_query_ids[qid] = did
                needed_doc_ids.add(did)
            if len(needed_query_ids) >= limit:
                break

        query_texts = {}
        for ex in queries:
            qid = str(ex.get("_id"))
            if qid in needed_query_ids:
                query_texts[qid] = ex.get("text", "")
            if len(query_texts) >= len(needed_query_ids):
                break

        doc_texts = {}
        for ex in corpus:
            did = str(ex.get("_id"))
            if did in needed_doc_ids:
                title = ex.get("title") or ""
                text = ex.get("text") or ""
                doc_texts[did] = (title + "\n" + text).strip()
            if len(doc_texts) >= len(needed_doc_ids):
                break

        for qid, did in needed_query_ids.items():
            query = query_texts.get(qid)
            positive = doc_texts.get(did)
            if not query or not positive:
                continue
            write_row(out_f, {
                "task": "retrieval",
                "lang": lang,
                "query": spec["query_prompt"] + query,
                "positive": positive,
                "source": spec["dataset"],
                "subset": cfg,
            })
            written += 1
            if written >= limit:
                return written
        return written

    configs = [lang, f"{lang}/train", "en"]
    for cfg in configs:
        try:
            ds = load_dataset(spec["dataset"], cfg, split="train", streaming=True)
        except Exception:
            continue
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
                return written
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
    parser.add_argument(
        "--hf-retrieval",
        action="store_true",
        help="Also try real retrieval datasets. MIRACL corpus joins can be slow in streaming mode.",
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
    global DONE_KEYS
    DONE_KEYS = done

    with out.open(mode, encoding="utf-8") as f:
        if args.mode == "hf-then-fallback":
            for lang in langs:
                code = lang["code"]
                for task, spec in sources.items():
                    existing_count = sum(1 for item in done if item[0] == code and item[1] == task)
                    if existing_count >= args.per_task_language:
                        print(f"{code}/{task}: already has {existing_count}, skipping", flush=True)
                        continue
                    if task == "retrieval" and not args.hf_retrieval:
                        print(f"{code}/{task}: skipping HF retrieval; fallback will fill this bucket", flush=True)
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
    print_output_summary(out)


if __name__ == "__main__":
    main()
