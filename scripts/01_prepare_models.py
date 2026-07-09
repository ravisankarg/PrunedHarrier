import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer

from common import ensure_dir, read_json, set_cache_env, write_json


def slice_copy(target_tensor, source_tensor):
    slices = tuple(slice(0, min(t, s)) for t, s in zip(target_tensor.shape, source_tensor.shape))
    target_tensor.zero_()
    target_tensor[slices].copy_(source_tensor[slices].to(dtype=target_tensor.dtype))


def copy_module_sliced(target, source):
    source_state = source.state_dict()
    with torch.no_grad():
        for name, target_tensor in target.state_dict().items():
            source_tensor = source_state.get(name)
            if source_tensor is None:
                continue
            if target_tensor.shape == source_tensor.shape:
                target_tensor.copy_(source_tensor.to(dtype=target_tensor.dtype))
            else:
                slice_copy(target_tensor, source_tensor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="microsoft/harrier-oss-v1-0.6b")
    parser.add_argument("--student-config", default="configs/student_12l_384.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_cache_env(args.cache_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    done_files = [out_dir / "config.json", out_dir / "model.safetensors", out_dir / "slice_manifest.json"]
    if not args.force and all(path.exists() for path in done_files):
        print(f"Student already prepared at {out_dir}; use --force to rebuild")
        return

    print(f"Downloading teacher snapshot metadata/files for {args.teacher}")
    teacher_snapshot = snapshot_download(
        args.teacher,
        local_dir=Path("models") / "teacher_harrier_0_6b",
        resume_download=True,
    )
    print(f"Teacher snapshot: {teacher_snapshot}")

    print("Loading teacher config/model on CPU")
    teacher_config = AutoConfig.from_pretrained(args.teacher)
    teacher = AutoModel.from_pretrained(args.teacher, torch_dtype=torch.float32, device_map=None)
    teacher.eval()

    plan = read_json(args.student_config)
    student_config = AutoConfig.from_pretrained(args.teacher)
    for key in [
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
    ]:
        setattr(student_config, key, plan[key])
    if hasattr(student_config, "layer_types"):
        student_config.layer_types = ["full_attention"] * plan["num_hidden_layers"]
    if hasattr(student_config, "max_window_layers"):
        student_config.max_window_layers = plan["num_hidden_layers"]
    student_config.use_cache = False

    print("Creating student and slicing weights")
    student = AutoModel.from_config(student_config)
    layer_map = plan["layer_map"]
    if len(layer_map) != student_config.num_hidden_layers:
        raise ValueError("layer_map length must match student num_hidden_layers")

    # Copy global tensors first, then replace each student layer from the mapped teacher layer.
    copy_module_sliced(student, teacher)
    with torch.no_grad():
        for student_i, teacher_i in enumerate(layer_map):
            print(f"student layer {student_i} <- teacher layer {teacher_i}")
            copy_module_sliced(student.layers[student_i], teacher.layers[teacher_i])

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    student.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    for name in ["modules.json", "config_sentence_transformers.json"]:
        src = Path(teacher_snapshot) / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    pooling_src = Path(teacher_snapshot) / "1_Pooling"
    if pooling_src.exists():
        shutil.copytree(pooling_src, out_dir / "1_Pooling", dirs_exist_ok=True)
        pooling_config = out_dir / "1_Pooling" / "config.json"
        if pooling_config.exists():
            with pooling_config.open("r", encoding="utf-8") as f:
                pooling = json.load(f)
            pooling["word_embedding_dimension"] = student_config.hidden_size
            with pooling_config.open("w", encoding="utf-8") as f:
                json.dump(pooling, f, ensure_ascii=False, indent=2)
                f.write("\n")
    ensure_dir(out_dir / "2_Normalize")

    write_json(
        out_dir / "slice_manifest.json",
        {
            "teacher": args.teacher,
            "teacher_snapshot": str(teacher_snapshot),
            "teacher_hidden_size": teacher_config.hidden_size,
            "teacher_layers": teacher_config.num_hidden_layers,
            "student_config": plan,
            "note": "Weights were copied by exact match or leading-dimension tensor slicing.",
        },
    )
    print(f"Saved sliced student to {out_dir}")


if __name__ == "__main__":
    main()
