import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoConfig

from common import ensure_dir


def repair_one(path, template):
    path = Path(path)
    template = Path(template)
    config = AutoConfig.from_pretrained(path)

    for name in ["modules.json", "config_sentence_transformers.json"]:
        src = template / name
        if src.exists() and src.resolve() != (path / name).resolve():
            shutil.copy2(src, path / name)

    pooling_src = template / "1_Pooling"
    if pooling_src.exists() and pooling_src.resolve() != (path / "1_Pooling").resolve():
        shutil.copytree(pooling_src, path / "1_Pooling", dirs_exist_ok=True)
    ensure_dir(path / "1_Pooling")
    pooling_config = path / "1_Pooling" / "config.json"
    if pooling_config.exists():
        with pooling_config.open("r", encoding="utf-8") as f:
            pooling = json.load(f)
    else:
        pooling = {
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": False,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False,
            "pooling_mode_lasttoken": True,
            "include_prompt": True,
        }
    pooling["word_embedding_dimension"] = config.hidden_size
    with pooling_config.open("w", encoding="utf-8") as f:
        json.dump(pooling, f, ensure_ascii=False, indent=2)
        f.write("\n")

    normalize_src = template / "2_Normalize"
    if normalize_src.exists() and normalize_src.resolve() != (path / "2_Normalize").resolve():
        shutil.copytree(normalize_src, path / "2_Normalize", dirs_exist_ok=True)
    else:
        ensure_dir(path / "2_Normalize")
    print(f"repaired {path} with hidden_size={config.hidden_size}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="models/harrier_student_12l_384_sliced")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        repair_one(path, args.template)


if __name__ == "__main__":
    main()
