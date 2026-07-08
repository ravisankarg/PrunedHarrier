import importlib
import os

import torch

from common import set_cache_env


def main():
    set_cache_env()
    packages = ["torch", "transformers", "sentence_transformers", "datasets", "mteb"]
    print("Python environment check")
    for pkg in packages:
        mod = importlib.import_module(pkg)
        print(f"{pkg}: {getattr(mod, '__version__', 'installed')}")

    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"gpu_memory_gb: {total:.2f}")
        print(f"bf16_supported: {torch.cuda.is_bf16_supported()}")

    for key in ["HF_HOME", "HF_DATASETS_CACHE", "MTEB_CACHE"]:
        print(f"{key}={os.environ.get(key)}")


if __name__ == "__main__":
    main()
