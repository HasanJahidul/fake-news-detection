"""Detached Colab T4 fine-tune driver for the 03-06 manual checkpoint.

Runs as a plain `python` process (not a notebook kernel) so there are no stale-module
issues, and writes EVERYTHING durable to Drive so a free-tier runtime reset cannot lose
work:

  * data is staged FROM Drive (``MyDrive/fnd_data/``) into the repo's ``data/processed/``
  * each exported backbone is written TO Drive (``MyDrive/fnd_models/transformer/<key>/``)

Usage (from the repo root on Colab):

    nohup python scripts/train_colab.py banglishbert \
        > /content/drive/MyDrive/fnd_train.log 2>&1 &

Pass one or more backbone keys; defaults to ``banglishbert`` only (the code-mixed primary,
fits a free T4 comfortably). Pass ``banglishbert xlmr`` to train both. This composes the
exact 03-02/03-03 contracts (``train_stage`` → ``fit_temperature`` → ``export_backbone``);
it contains no training logic of its own. Mirrors notebooks/03_train_transformer_colab.ipynb
but as a detachable, Drive-logging script (D-05/D-06).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = "/content/fake-news-detection"
DRIVE_DATA = "/content/drive/MyDrive/fnd_data"
DRIVE_OUT = "/content/drive/MyDrive/fnd_models/transformer"

sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)


def _log(msg: str) -> None:
    print(msg, flush=True)


def stage_data() -> None:
    """Copy the parquet splits from Drive into data/processed/ (fail loudly if absent)."""
    dst = os.path.join(REPO_ROOT, "data", "processed")
    os.makedirs(dst, exist_ok=True)
    missing = []
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        src = os.path.join(DRIVE_DATA, name)
        out = os.path.join(dst, name)
        if os.path.exists(out):
            _log(f"data already present: {name}")
        elif os.path.exists(src):
            shutil.copy(src, out)
            _log(f"staged from Drive: {name}")
        else:
            missing.append(src)
    if missing:
        raise FileNotFoundError(
            "Missing parquet splits on Drive — upload train/val/test.parquet to "
            f"{DRIVE_DATA}/ . Not found: {missing}"
        )


def main() -> int:
    stage_data()

    from src.models.calibration import fit_temperature
    from src.models.transformer_data import load_splits
    from src.models.transformer_train import (
        BACKBONES,
        STAGE_GATE,
        STAGE_REALFAKE,
        export_backbone,
        train_stage,
    )

    keys = sys.argv[1:] or ["banglishbert"]
    unknown = [k for k in keys if k not in BACKBONES]
    if unknown:
        raise KeyError(f"unknown backbone(s) {unknown}; known: {sorted(BACKBONES)}")

    train_df, val_df, _test_df = load_splits()
    _log(f"loaded splits: {len(train_df)} train / {len(val_df)} val rows")

    os.makedirs(DRIVE_OUT, exist_ok=True)
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

        out = Path(DRIVE_OUT) / key
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
