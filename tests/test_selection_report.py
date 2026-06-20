"""D-11 / SC#4 — transformer selection report contract (Wave 0 RED).

Proves (greened by plan 03-04 `src/models/select_transformer.py` report writer):
  the written ``reports/transformer_selection_report.md`` records the chosen model +
  per-language and code-mixed metrics, and carries NO latency line (D-11), mirroring
  the classical report's counts/metrics-only discipline (SYS-02).

Guarded with ``importorskip("torch")``; the report writer is imported via
``importorskip`` so this SKIPS until 03-04 lands ``src.models.select_transformer``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def test_report_records_chosen_model_and_per_language(tmp_path):
    """Report names the chosen model + per-language/code-mixed metrics, no latency.

    RED until 03-04. ``write_selection_report(...)`` writes a Markdown report whose
    text names the chosen backbone and contains per-language rows (bn/en/code-mixed)
    and MUST NOT contain a latency metric line (D-11).
    """
    sel = pytest.importorskip("src.models.select_transformer")
    writer = getattr(sel, "write_selection_report", None)
    assert writer is not None, "select_transformer must expose write_selection_report"

    out = tmp_path / "transformer_selection_report.md"
    # Minimal results payload shape the writer renders (full schema fixed in 03-04).
    writer(out, results={}, chosen="banglishbert")

    text = out.read_text(encoding="utf-8")
    assert "banglishbert" in text
    assert "language" in text.lower()
    # D-11: no latency metric in the selection report.
    assert "latency" not in text.lower()


def test_report_is_counts_metrics_only(tmp_path):
    """Report writer never emits raw input text (SYS-02).

    RED until 03-04. Contract assertion: the writer accepts metrics/counts, not raw
    corpus text — assert its signature has no ``texts``/``raw`` parameter.
    """
    import inspect

    sel = pytest.importorskip("src.models.select_transformer")
    writer = getattr(sel, "write_selection_report", None)
    assert writer is not None
    params = inspect.signature(writer).parameters
    assert "texts" not in params and "raw" not in params
