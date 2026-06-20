"""D-08 / CLS-04 — temperature-scaling calibration, fit on VAL only.

One calibrated confidence per prediction (D-08). Pure functions (the only stateful object
is a local one-scalar LBFGS parameter) implementing the canonical gpleiss temperature
scaler (Pattern 5):

  * :func:`fit_temperature` — fit a single scalar ``T`` by minimizing NLL of ``logits / T``
    against the labels, using LBFGS. Its signature accepts ONLY validation tensors — there
    is structurally no ``test`` parameter, so it cannot fit on the test split (Pitfall 3 /
    T-03-04).
  * :func:`apply_temperature` — divide logits by ``T`` (the caller softmaxes).
  * :func:`save_temperature` / :func:`load_temperature` — persist/round-trip the
    ``{"gate", "realfake", "gate_threshold"}`` ``temperature.json`` schema.

``torch`` is imported lazily inside the fit/apply functions so this module imports without
torch present (the fast suite installs without the transformer optional group).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def fit_temperature(val_logits, val_labels) -> float:
    """Fit a single temperature ``T`` on VALIDATION logits/labels only (gpleiss, Pattern 5).

    Minimizes ``CrossEntropyLoss(val_logits / T, val_labels)`` over a one-scalar LBFGS
    parameter and returns ``float(T)`` (guaranteed > 0). The signature accepts ONLY the
    validation tensors — there is NO ``test`` parameter, so the fit can never touch the
    test split (Pitfall 3 / T-03-04). Reduces or ties NLL vs ``T = 1`` on a miscalibrated
    set.

    ``torch`` is imported lazily here so the module imports without torch present.
    """
    import torch
    from torch import nn

    logits = val_logits.detach() if hasattr(val_logits, "detach") else torch.as_tensor(val_logits)
    labels = val_labels.detach() if hasattr(val_labels, "detach") else torch.as_tensor(val_labels)
    logits = logits.float()
    labels = labels.long()

    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def _closure():
        optimizer.zero_grad()
        # Clamp to a tiny positive floor inside the closure so division stays well-defined.
        loss = criterion(logits / temperature.clamp_min(1e-6), labels)
        loss.backward()
        return loss

    optimizer.step(_closure)

    t = float(temperature.detach())
    # Guard: a degenerate/negative step must never produce a non-positive temperature.
    if not (t > 0.0):
        t = 1.0
    return t


def apply_temperature(logits, T: float):
    """Scale logits by the temperature: ``logits / T`` (the caller applies softmax)."""
    return logits / T


def save_temperature(
    path: PathLike,
    gate: float,
    realfake: float,
    gate_threshold: float,
) -> None:
    """Persist the calibration artifact as ``temperature.json`` ({gate, realfake, gate_threshold}).

    Written to a caller-supplied path that the trainer/selector anchors to
    ``models/transformer/<backbone>/`` (T-03-05) — not derived from untrusted input.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": float(gate),
        "realfake": float(realfake),
        "gate_threshold": float(gate_threshold),
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_temperature(path: PathLike) -> dict:
    """Round-trip :func:`save_temperature`: read ``temperature.json`` back into a dict.

    Returns ``{"gate", "realfake", "gate_threshold"}`` as floats.
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        "gate": float(data["gate"]),
        "realfake": float(data["realfake"]),
        "gate_threshold": float(data["gate_threshold"]),
    }
