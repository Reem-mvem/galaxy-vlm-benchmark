"""Central path configuration for galaxy-vlm-benchmark.

Machine-specific locations come from environment variables so nothing is hard-wired
to one cluster. Set them in your shell or a .env file; the defaults assume the repo
layout and work out of the box for the API/analysis parts.

    GVB_DATA_DIR     task catalogs + images     (default: <repo>/tasks)
    GVB_MODEL_DIR    local VLM weights          (default: <repo>/models)
    GVB_ADAPTER_DIR  LoRA adapters (off-git)    (default: <GVB_MODEL_DIR>/adapters)
    GVB_SCRATCH      fast scratch for staging   (default: $SLURM_TMPDIR or /tmp)

The trainers / evaluators also honour their own env vars — TRAIN_SAMPLE_CSV,
MODEL_LOCAL_PATH, ADAPTER_OUT, IMAGES_DIR, RINGS_CATALOG, GZEVO_TRAIN_CSV, etc.
See docs/running-on-hpc.md for the full list and example jobs.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _p(env: str, default) -> Path:
    return Path(os.environ.get(env, str(default)))


DATA_DIR    = _p("GVB_DATA_DIR",    REPO_ROOT / "tasks")
MODEL_DIR   = _p("GVB_MODEL_DIR",   REPO_ROOT / "models")
ADAPTER_DIR = _p("GVB_ADAPTER_DIR", MODEL_DIR / "adapters")
SCRATCH     = _p("GVB_SCRATCH",     os.environ.get("SLURM_TMPDIR", "/tmp"))


def task_dir(name: str) -> Path:
    """tasks/<name>/ — holds task.py, data/, results/."""
    return DATA_DIR / name
