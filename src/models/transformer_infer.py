"""CLS-02 / CLS-04 / D-08 — thin LOAD-ONLY transformer cascade loader.

The online half of the two-stage cascade. Mirrors the classical
:func:`src.models.train_classical.load_artifacts` discipline:

  * **Load-only (CLS-02 / SC#2).** This module imports ONLY transformers + torch +
    the shared ``src.preprocess.preprocess`` + ``src.models.calibration`` + json. It
    NEVER imports ``src.models.transformer_train`` — predicting from a saved model must
    not pull in any training code path.
  * **Two-stage cascade (CLS-04 / SC#3).** :meth:`TransformerCascade.predict` consults
    the binary gate (malicious vs not) FIRST; only when ``P_mal < gate_threshold`` does
    it consult the real/fake head. This is NOT a flat 3-way softmax.
  * **Calibrated path-product confidence (D-08).** Each head's logits are divided by its
    fitted temperature before softmax. The emitted confidence is the product of the stage
    probabilities along the taken path: ``P_mal`` for a malicious verdict, else
    ``(1 - P_mal) * P_verdict`` for a real/fake verdict — always in ``[0, 1]``.
  * **Normalizer parity (D-12).** Every text is fed through the ONE shared
    :func:`src.preprocess.preprocess` (the exact function training uses) BEFORE the
    tokenizer — no normalization fork.

Security (ASVS V14 / T-03-09..11): models are loaded via ``from_pretrained`` (safetensors
default in transformers 4.46), never ``torch.load`` of an arbitrary file; the default
load path is the fixed project-anchored ``MODELS_DIR`` constant, not a user-supplied path.

``torch`` / ``transformers`` are imported lazily inside the methods so this module imports
without the transformer optional group present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from src.data.label_map import LABELS  # noqa: F401  (locked class order; never hardcode)
from src.models.calibration import (
    apply_temperature,
    load_temperature,
    save_temperature,
)
from src.models.transformer_data import (
    GATE_CLASSES,
    MAX_LENGTH,
    REALFAKE_CLASSES,
)
from src.preprocess import preprocess

PathLike = Union[str, Path]

# Repo root = two levels up from src/models/transformer_infer.py (mirrors train_classical).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Project-anchored default export root (CLS-02 load-only, T-03-10 no user path honored here).
MODELS_DIR = REPO_ROOT / "models" / "transformer"

# Default per-head temperatures + gate threshold when a sidecar is absent (e.g. the tiny
# test fixture exports raw heads with no temperature.json). T=1.0 is the identity (no
# scaling); the 0.5 gate threshold is the 03-03 sentinel until the val sweep rewrites it.
_DEFAULT_TEMP = {"gate": 1.0, "realfake": 1.0, "gate_threshold": 0.5}


def _load_label_map(model_dir: Optional[Path]) -> dict:
    """Read ``label_map.json`` if present, else fall back to the locked stage class orders.

    The trainer writes ``label_map.json`` ({"gate": [...], "realfake": [...]}). When loading
    raw heads with no sidecar (the tiny fixture path), default to the canonical
    :data:`GATE_CLASSES` / :data:`REALFAKE_CLASSES` from ``transformer_data`` (never hardcode).
    """
    if model_dir is not None:
        p = Path(model_dir) / "label_map.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return {
                "gate": list(data.get("gate", GATE_CLASSES)),
                "realfake": list(data.get("realfake", REALFAKE_CLASSES)),
            }
    return {"gate": list(GATE_CLASSES), "realfake": list(REALFAKE_CLASSES)}


def _load_temp(model_dir: Optional[Path]) -> dict:
    """Read ``temperature.json`` if present, else the identity-temperature defaults."""
    if model_dir is not None:
        p = Path(model_dir) / "temperature.json"
        if p.exists():
            return load_temperature(p)
    return dict(_DEFAULT_TEMP)


class TransformerCascade:
    """Load-only two-stage cascade: gate (malicious vs not) → real/fake, calibrated confidence.

    Construct either from a single export directory::

        TransformerCascade(model_dir="models/transformer/banglishbert")

    (which loads ``gate/``, ``realfake/``, ``label_map.json``, ``temperature.json``), or
    from two explicit head directories (the wiring-level test path)::

        TransformerCascade(gate_dir=..., realfake_dir=...)

    No ``transformer_train`` import: this is the load-only inference surface (CLS-02).
    """

    def __init__(
        self,
        model_dir: Optional[PathLike] = None,
        *,
        gate_dir: Optional[PathLike] = None,
        realfake_dir: Optional[PathLike] = None,
    ) -> None:
        # Lazy heavy imports so module import stays torch-free.
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        # Default the export root to the project-anchored MODELS_DIR (no user path honored
        # at the pipeline boundary; explicit dirs are the test/Colab→M4 wiring path only).
        if model_dir is None and gate_dir is None and realfake_dir is None:
            model_dir = MODELS_DIR

        if gate_dir is None or realfake_dir is None:
            if model_dir is None:
                raise ValueError(
                    "TransformerCascade needs model_dir OR both gate_dir and realfake_dir"
                )
            base = Path(model_dir)
            gate_dir = base / "gate" if gate_dir is None else gate_dir
            realfake_dir = base / "realfake" if realfake_dir is None else realfake_dir
            sidecar_dir: Optional[Path] = base
        else:
            sidecar_dir = Path(model_dir) if model_dir is not None else None

        self.label_map = _load_label_map(sidecar_dir)
        self.temp = _load_temp(sidecar_dir)

        # safetensors default (transformers 4.46) — never torch.load of an arbitrary file.
        self.gate_tok = AutoTokenizer.from_pretrained(str(gate_dir))
        self.gate = AutoModelForSequenceClassification.from_pretrained(str(gate_dir)).eval()
        self.rf_tok = AutoTokenizer.from_pretrained(str(realfake_dir))
        self.rf = AutoModelForSequenceClassification.from_pretrained(str(realfake_dir)).eval()

    def _probs(self, model, tok, text: str, T: float):
        """preprocess → tokenize (head-truncate, MAX_LENGTH) → softmax(apply_temperature).

        Runs the SHARED ``preprocess()`` BEFORE tokenizing (D-12 parity), then scales the
        logits by the calibrated temperature ``T`` via the single-source
        :func:`src.models.calibration.apply_temperature` (WR-04) and softmaxes. Returns the
        1-D class probability tensor for the single input row.
        """
        import torch

        with torch.no_grad():
            enc = tok(
                preprocess(text),
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            return torch.softmax(apply_temperature(logits, T), dim=-1)[0]

    def gate_realfake_probs(self, text: str) -> tuple[float, float]:
        """Per-stage probabilities for the sweep: ``(p_mal, p_real)``.

        ``p_mal`` is the gate softmax at the ``malicious`` index; ``p_real`` is the
        real/fake head softmax at the ``real`` index — both produced by the SAME calibrated
        :meth:`_probs` path the cascade uses, so the offline sweep and the online
        :meth:`predict` share one probability source. Returns plain floats in ``[0, 1]``.
        """
        gate_probs = self._probs(self.gate, self.gate_tok, text, self.temp["gate"])
        mal_idx = self.label_map["gate"].index("malicious")
        p_mal = float(gate_probs[mal_idx])

        rf_probs = self._probs(self.rf, self.rf_tok, text, self.temp["realfake"])
        real_idx = self.label_map["realfake"].index("real")
        p_real = float(rf_probs[real_idx])
        return p_mal, p_real

    def predict(self, text: str) -> dict:
        """Run the two-stage cascade and return ``{label, confidence, path, path_probs}``.

        Gate first: ``P_mal = gate softmax at the 'malicious' index``. If
        ``P_mal >= gate_threshold`` → malicious verdict with ``confidence = P_mal`` (a
        single-stage path). Otherwise consult the real/fake head, take its argmax, and
        emit ``confidence = (1 - P_mal) * P_verdict`` (the two-stage path product, D-08).

        ``path`` lists the stages consulted; ``path_probs`` lists the per-stage
        probabilities whose product equals ``confidence`` (so the confidence is auditable).
        """
        import torch

        gate_probs = self._probs(self.gate, self.gate_tok, text, self.temp["gate"])
        mal_idx = self.label_map["gate"].index("malicious")
        p_mal = float(gate_probs[mal_idx])

        if p_mal >= self.temp["gate_threshold"]:
            return {
                "label": "malicious",
                "confidence": p_mal,
                "path": ["gate"],
                "path_probs": [p_mal],
            }

        rf_probs = self._probs(self.rf, self.rf_tok, text, self.temp["realfake"])
        idx = int(torch.argmax(rf_probs))
        verdict = self.label_map["realfake"][idx]
        p_verdict = float(rf_probs[idx])
        p_not_mal = 1.0 - p_mal
        return {
            "label": verdict,
            "confidence": p_not_mal * p_verdict,
            "path": ["gate", "realfake"],
            "path_probs": [p_not_mal, p_verdict],
        }


def write_gate_threshold(model_dir: PathLike, threshold: float) -> None:
    """Overwrite the exported ``temperature.json`` gate_threshold with the swept value.

    Loads ``<model_dir>/temperature.json`` via :func:`load_temperature`, replaces only its
    ``gate_threshold`` field with ``float(threshold)``, and re-persists via
    :func:`save_temperature` — the per-head ``gate``/``realfake`` temperatures are preserved
    untouched. Lets the selector overwrite the 0.5 sentinel after the val sweep WITHOUT
    re-exporting weights (T-03-16: selector-internal, fixed project-anchored path, no user
    input crosses this boundary).
    """
    p = Path(model_dir) / "temperature.json"
    temp = load_temperature(p)
    save_temperature(
        p,
        gate=temp["gate"],
        realfake=temp["realfake"],
        gate_threshold=float(threshold),
    )


def load(model_dir: Optional[PathLike] = None) -> TransformerCascade:
    """Convenience load-only entrypoint mirroring ``train_classical.load_artifacts``.

    Returns a ready :class:`TransformerCascade` for ``model_dir`` (defaults to the
    project-anchored :data:`MODELS_DIR`). No training code path is imported.
    """
    return TransformerCascade(model_dir=model_dir)
