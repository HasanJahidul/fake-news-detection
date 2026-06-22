"""Local M4 (MPS) fine-tune runner for the 03-06 checkpoint.

Colab's free tier kept evicting the GPU kernel (env-level, reproducible across accounts), so
the manual fine-tune runs here on Apple-Silicon GPU via PyTorch MPS (no CUDA). ``fp16``
auto-disables on MPS — ``transformer_train`` guards it on ``torch.cuda.is_available()`` — so
this runs fp32. Reads the local ``data/processed/`` splits and exports each backbone to
``models/transformer/<key>/`` (the exact layout 03-06's selection step expects).

Defaults to ``banglishbert`` (the code-mixed primary). Pass keys to override
(``banglishbert xlmr``). Optional env overrides for a memory-tight 16 GB box:
``TRAIN_BATCH`` (recorded default 16), ``TRAIN_EPOCHS`` (recorded default 3).
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU-fallback for any MPS-missing op
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    import torch

    from src.models import transformer_train as T
    from src.models.calibration import fit_temperature
    from src.models.transformer_data import load_splits
    from src.models.transformer_train import (
        BACKBONES,
        STAGE_GATE,
        STAGE_REALFAKE,
        export_backbone,
        train_stage,
    )

    # Optional resilience overrides (default to the recorded TRAIN_HPARAMS).
    if os.environ.get("TRAIN_BATCH"):
        T.TRAIN_HPARAMS["per_device_train_batch_size"] = int(os.environ["TRAIN_BATCH"])
    if os.environ.get("TRAIN_EPOCHS"):
        T.TRAIN_HPARAMS["num_train_epochs"] = int(os.environ["TRAIN_EPOCHS"])

    dev = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _log(
        f"torch {torch.__version__} | device {dev} | "
        f"batch {T.TRAIN_HPARAMS['per_device_train_batch_size']} | "
        f"epochs {T.TRAIN_HPARAMS['num_train_epochs']}"
    )

    keys = sys.argv[1:] or ["banglishbert"]
    unknown = [k for k in keys if k not in BACKBONES]
    if unknown:
        raise KeyError(f"unknown backbone(s) {unknown}; known: {sorted(BACKBONES)}")

    train_df, val_df, _test_df = load_splits()
    _log(f"loaded splits: {len(train_df)} train / {len(val_df)} val rows")

    out_root = REPO_ROOT / "models" / "transformer"
    out_root.mkdir(parents=True, exist_ok=True)
    for key in keys:
        model_id = BACKBONES[key]
        _log(f"=== {key} ({model_id}) — GATE stage ===")
        gate_model, gate_tok, gate_logits, gate_y = train_stage(
            model_id, STAGE_GATE, train_df, val_df
        )
        gate_T = fit_temperature(gate_logits, gate_y)
        _log(f"{key} gate done (T={gate_T:.3f})")

        _log(f"=== {key} ({model_id}) — REALFAKE stage ===")
        rf_model, rf_tok, rf_logits, rf_y = train_stage(
            model_id, STAGE_REALFAKE, train_df, val_df
        )
        rf_T = fit_temperature(rf_logits, rf_y)
        _log(f"{key} realfake done (T={rf_T:.3f})")

        out = out_root / key
        export_backbone(
            key,
            gate_model,
            gate_tok,
            rf_model,
            rf_tok,
            gate_T=gate_T,
            realfake_T=rf_T,
            gate_threshold=0.5,
            out_dir=out,
        )
        _log(f"EXPORTED {key} -> {out} (gate_T={gate_T:.3f}, rf_T={rf_T:.3f})")

    _log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
