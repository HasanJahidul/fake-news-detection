"""Streamlit UI — real-time fake-news / malicious-content detector.

Run from repo root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make `src` importable when launched via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import analyze  # noqa: E402

st.set_page_config(page_title="Fake News & Malicious Content Detector",
                   page_icon="🛡️", layout="wide")

_LABEL_STYLE = {
    "real": ("✅ REAL", "#1a7f37"),
    "fake": ("⚠️ FAKE", "#b35900"),
    "malicious": ("⛔ MALICIOUS", "#b3001b"),
}
_CAT_COLOR = {
    "scam_phishing": "#ffd6d6", "clickbait": "#fff2cc",
    "sensational": "#ffe0cc", "clickbait_bn": "#fff2cc",
}


def highlight(text: str, matches: list) -> str:
    """Wrap matched style spans in colored <mark> tags (non-overlapping)."""
    if not matches:
        return text.replace("\n", "<br>")
    spans = sorted(matches, key=lambda m: m["start"])
    out, cursor = [], 0
    for m in spans:
        if m["start"] < cursor:
            continue
        out.append(text[cursor:m["start"]])
        color = _CAT_COLOR.get(m["category"], "#eee")
        seg = text[m["start"]:m["end"]]
        out.append(f"<mark style='background:{color}' title='{m['category']}'>{seg}</mark>")
        cursor = m["end"]
    out.append(text[cursor:])
    return "".join(out).replace("\n", "<br>")


st.title("🛡️ AI-Driven Fake News & Malicious Content Detection")
st.caption("Bilingual (English + বাংলা) · classifier + source credibility + writing-style "
           "+ external verification · decision fusion")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Analyze", ["Text", "URL"], horizontal=True)
    text_in = url_in = None
    if mode == "Text":
        text_in = st.text_area("Paste content", height=220,
                               placeholder="Paste a news snippet or message ...")
    else:
        url_in = st.text_input("Article URL", placeholder="https://...")
    run = st.button("Analyze", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Set `NEWSAPI_KEY` in the environment to enable external verification.")

if run:
    if (mode == "Text" and not text_in) or (mode == "URL" and not url_in):
        st.warning("Provide input first.")
        st.stop()
    with st.spinner("Analyzing ..."):
        res = analyze(text=text_in, url=url_in)

    if "error" in res:
        st.error(res["error"])
        st.stop()

    label_txt, color = _LABEL_STYLE.get(res["label"], (res["label"].upper(), "#333"))
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<h2 style='color:{color}'>{label_txt}</h2>", unsafe_allow_html=True)
    c2.metric("Confidence", f"{res['confidence']:.0%}")
    c3.metric("Fused risk", f"{res['fused_risk']:.2f}")

    st.progress(min(1.0, float(res["fused_risk"])), text="Overall risk")

    st.subheader("Classifier probabilities")
    st.bar_chart(res["classifier"]["proba"])
    st.caption(f"Model: {res['classifier']['name']}")

    st.subheader("Module contributions (decision fusion)")
    rows = [{"module": k, "risk": v["risk"], "weight": v["weight"],
             "weighted": v["weighted"]} for k, v in res["contributions"].items()]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Why — explainability")
    for mod, reasons in res["explanations"].items():
        for r in reasons:
            st.markdown(f"- **{mod}**: {r}")

    st.subheader("Writing-style highlights")
    raw_preview = (text_in or res["input"].get("preview", ""))
    st.markdown(highlight(raw_preview, res["style"]["matches"]), unsafe_allow_html=True)

    with st.expander("Raw JSON"):
        st.json(res)
else:
    st.info("Enter text or a URL in the sidebar, then click **Analyze**.")
