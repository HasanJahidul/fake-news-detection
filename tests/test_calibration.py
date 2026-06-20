"""D-08 — temperature-scaling calibration on VAL only (Wave 0 RED).

Proves (greened by plan 03-03 `src/models/calibration.py`):
  fitting a temperature on the validation logits reduces NLL vs ``T=1``, and the fit
  touches the val split ONLY (never test) — Pitfall 3.

Guarded with ``importorskip("torch")``; the calibration module is imported via
``importorskip`` so this SKIPS until 03-03 lands ``src.models.calibration``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _toy_val_logits_labels():
    import torch

    # Mildly over-confident logits vs true labels — temperature scaling should help.
    torch.manual_seed(0)
    logits = torch.tensor(
        [
            [3.0, -3.0],
            [2.5, -2.5],
            [-3.0, 3.0],
            [-2.0, 2.0],
            [2.0, -2.0],
            [-2.5, 2.5],
        ]
    )
    labels = torch.tensor([0, 1, 1, 1, 0, 1])  # one deliberate mismatch (row 1)
    return logits, labels


def test_temperature_reduces_nll_vs_T1():
    """Fitted temperature lowers val NLL relative to T=1.

    RED until 03-03. ``fit_temperature(val_logits, val_labels)`` returns a scalar T>0;
    the NLL at the fitted T must be <= the NLL at T=1.
    """
    import torch

    cal = pytest.importorskip("src.models.calibration")
    logits, labels = _toy_val_logits_labels()

    T = cal.fit_temperature(logits, labels)
    assert float(T) > 0.0

    nll = torch.nn.functional.cross_entropy
    nll_T1 = nll(logits, labels).item()
    nll_fit = nll(logits / float(T), labels).item()
    assert nll_fit <= nll_T1 + 1e-6


def test_fit_uses_val_only(monkeypatch):
    """Calibration fit must not read the test split (Pitfall 3).

    RED until 03-03. The fitter signature accepts val logits/labels only — assert it
    does not expose a ``test`` parameter that could leak the test split into the fit.
    """
    import inspect

    cal = pytest.importorskip("src.models.calibration")
    sig = inspect.signature(cal.fit_temperature)
    assert "test" not in sig.parameters
