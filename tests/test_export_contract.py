"""CLS-02 / SC#2 — transformer export-directory layout contract (Wave 0 RED).

Proves (greened by plan 03-02 `src/models/transformer_train.py` export):
  an exported ``models/transformer/<backbone>/`` directory contains the two-stage
  cascade subdirs + sidecar files: ``gate/``, ``realfake/``, ``label_map.json``,
  ``temperature.json`` (RESEARCH Pattern 4).

Guarded with ``importorskip("torch")``; the export helper is imported via
``importorskip`` so this SKIPS until 03-02 lands the export contract.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


_REQUIRED_SUBDIRS = ("gate", "realfake")
_REQUIRED_FILES = ("label_map.json", "temperature.json")


def test_export_dir_has_cascade_layout(tiny_seqcls_model, tmp_path):
    """Exported backbone dir has gate/ realfake/ label_map.json temperature.json.

    RED: ``src.models.transformer_train`` does not exist yet (plan 03-02). The export
    helper must persist BOTH cascade stages plus the label map + temperature sidecar.
    """
    train = pytest.importorskip("src.models.transformer_train")

    out = tmp_path / "transformer" / "tiny"
    # The export function name is part of the 03-02 contract; accept either spelling.
    export = getattr(train, "export_cascade", None) or getattr(train, "save_pretrained_cascade", None)
    assert export is not None, "transformer_train must expose a cascade export helper"

    export(out, gate_dir=tiny_seqcls_model, realfake_dir=tiny_seqcls_model)

    for sub in _REQUIRED_SUBDIRS:
        assert (out / sub).is_dir(), f"export missing {sub}/ subdir"
    for fname in _REQUIRED_FILES:
        assert (out / fname).is_file(), f"export missing {fname}"
