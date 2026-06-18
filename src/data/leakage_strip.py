"""DATA-03 — offline per-source boilerplate / source-leakage stripping (D-09).

THE make-or-break task of Phase 01: ISOT's "real" class is 100% Reuters-datelined
(every true article begins ``'CITY (Reuters) -'``; fake never does). A model trained
without removing this hits ~100% accuracy detecting the substring "Reuters" and learns
nothing about truth (RESEARCH Pitfall 1; arXiv 2305.19194 / 2312.00292; UVic ISOT readme).

This module strips that dateline (and analogous outlet / byline / URL-as-source /
image-credit artifacts) **offline, BEFORE splitting** — and it must NEVER be imported by
``src.preprocess.preprocess`` (D-09): live user input has no datelines, so doing this at
inference time is dead cost and risks stripping legitimately-quoted user text (Pitfall 4).
Plan 01-07's source-stripped leakage probe proves the leak is actually gone.

Stripping is conservative and per-source (dispatch on ``source_dataset``) — it removes only
known source artifacts, never legitimate content. Regexes are anchored / bounded to avoid
catastrophic backtracking on the offline corpus (T-04-DoS); dataset text is handled strictly
as DATA (no eval / render — the phishing corpus is literally malicious text, T-04-TM).
"""

import re

# --- ISOT Reuters dateline (Pitfall 1, THE critical leak) -------------------
# Leading 'CITY (Reuters) -' prefix on ISOT real bodies. City names are short
# Title-Case tokens (e.g. WASHINGTON, NEW YORK, SAN FRANCISCO); bound the city
# run so the anchored regex cannot backtrack catastrophically (T-04-DoS).
_ISOT_LEADING_DATELINE = re.compile(
    r"^[A-Z][A-Za-z .,'/()-]{0,60}?\(Reuters\)\s*[-–—]\s*"
)
# Residual '(Reuters)' mentions elsewhere in the body (also a leak signal). We drop
# the token and any immediately-adjacent dash/spacing so no '(Reuters) -' survives.
_ISOT_RESIDUAL_REUTERS = re.compile(r"\s*\(Reuters\)\s*[-–—]?\s*")
# Collapse any double spaces the removals leave behind.
_WS = re.compile(r"\s{2,}")


def strip_isot_dateline(text: str) -> str:
    """Remove the ISOT leading Reuters dateline + residual '(Reuters)' mentions.

    Idempotent: ``strip_isot_dateline(strip_isot_dateline(x)) == strip_isot_dateline(x)``.
    Returns "" for None / empty input.
    """
    if not text:
        return ""
    # 1. Strip the leading 'CITY (Reuters) -' prefix.
    text = _ISOT_LEADING_DATELINE.sub("", text, count=1)
    # 2. Strip any residual '(Reuters)' mentions left in the body.
    text = _ISOT_RESIDUAL_REUTERS.sub(" ", text)
    # 3. Tidy spacing introduced by the removals.
    text = _WS.sub(" ", text).strip()
    return text


# --- Per-source dispatch ----------------------------------------------------
# Only ISOT carries the Reuters dateline leak. BanFakeNews/phishing per-source
# boilerplate (outlet names, bylines, URL-as-source, image-credit lines) is sparse
# and embedded as provenance columns rather than inside the article body in the
# acquired corpora (01-02 schemas: banfakenews has a separate ``domain``/``source``
# column; phishing keeps ``sender``/``subject`` separate from ``body``), so no
# in-body strip rule is needed for them today. Hooks are kept here so 01-07 probe
# findings can add conservative rules without touching the dispatch surface.


def strip_boilerplate(text: str, source_dataset: str) -> str:
    """Apply the per-source boilerplate strip for ``source_dataset`` to ``text``.

    Conservative per-source dispatch (NOT a global strip): only the source it is told
    to is touched, so a non-ISOT row containing the literal '(Reuters)' is left intact.
    Returns "" for None / empty input.
    """
    if not text:
        return ""
    if source_dataset == "isot":
        return strip_isot_dateline(text)
    # No in-body boilerplate rule for the other sources (see module note above).
    return text


# ISOT carries a second leak: publication years differ systematically between the
# real and fake classes, so a ``date`` column would let a model read the label off
# the year. Drop it from the corpus entirely (Pitfall 1).
_ISOT_LEAKING_COLUMNS = ("date",)


def strip_dataframe(df):
    """Apply the right strip rule per ``source_dataset`` row, offline, before splitting.

    - Each row's ``text`` is stripped according to its ``source_dataset`` (per-source
      dispatch; non-ISOT rows are unaffected by the Reuters rule).
    - The ISOT ``date`` column is DROPPED from the corpus (year leak, Pitfall 1).

    Returns a new DataFrame; the input is not mutated.
    """
    out = df.copy()
    if "text" in out.columns and "source_dataset" in out.columns:
        out["text"] = [
            strip_boilerplate(t, s)
            for t, s in zip(out["text"], out["source_dataset"])
        ]
    # Drop the ISOT year-leak column(s) from the whole corpus.
    drop = [c for c in _ISOT_LEAKING_COLUMNS if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    return out
