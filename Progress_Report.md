# Project Progress Report

**Project:** AI-Driven Real-Time Detection of Fake News and Malicious Content
**Course:** PMICS Capstone, Department of CSE, University of Dhaka
**Team:** A.S.M Rafiuzzaman Sazin (H-404), Mahi Naz Islam (H-426)
**Supervisor:** Mr. Jargis Ahmed
**Co-supervisor:** Mr. Md. Faisal Hossain
**Date:** 19 June 2026

---

## Where we are right now

We have finished the first part of the system. The full plan has 7 stages and we are done with the first 2. In simple words, we have built the data and the basic models. The actual app that a user will use comes in the later stages.

Out of the 29 things we planned to build, 7 are done so far.

## What we have done

### 1. Collected and cleaned the data

We collected news and message data from 5 public datasets. Some are Bangla, some are English, and some are spam/phishing data. After cleaning and removing duplicate items we ended up with about **137,000 articles**, and each one is labelled as **real, fake, or malicious**.

One thing we were careful about is that a model can sometimes "cheat". For example, if every Reuters news item has the word "Reuters" in it, the model can just learn that word instead of actually reading the news. We wrote a small test to catch this kind of problem. It actually found one, so we fixed it and checked again to make sure it was clean. We also split the data carefully so that the test data is completely separate from the training data.

We also made one cleaning function that works for Bangla, English and mixed text, and we use the same function both for training and later for the live app, so they stay consistent.

### 2. Built the first models

We trained 3 simple models (Logistic Regression, Naive Bayes, and Random Forest) on this data. The best one was **Logistic Regression**, which got a macro-F1 score of about **0.91** on the test set. This is a good starting point.

We did not only look at accuracy, because accuracy can be misleading when the classes are not balanced. So we also checked the score for each class (real, fake, malicious) separately, and made a confusion matrix.

After training, we saved the model so the next parts of the project can just load it and use it.

So far this is around 2,600 lines of code and 1,600 lines of tests, all kept in git.

## What is still left

We still have 5 stages left. These are the bigger parts that turn it into a real product:

- **Stage 3 — Stronger model:** train a transformer model (BanglaBERT / BanglishBERT) that understands Bangla and mixed text better than the simple models.
- **Stage 4 — Extra checks:** check the reputation of the source, look at the writing style (clickbait words, ALL CAPS, too many !!!), and detect phishing, scam, and suspicious links.
- **Stage 5 — Fact checking:** look up the claim on free sources like Google Fact Check and Wikipedia.
- **Stage 6 — Combine and explain:** mix all these signals into one final answer, and show which words or reasons led to that answer.
- **Stage 7 — The app:** the Streamlit web page where you paste text or a link and get the result instantly.

The main feature we are aiming for — paste some text or a link and get a real / fake / malicious answer with an explanation — will be ready in the last two stages.

## A few things to keep in mind

- The 0.91 score is on all the data mixed together. How well it works on only Bangla or mixed text is something the transformer model in the next stage should improve.
- We still need to confirm the license of one dataset. This is low risk because we are not sharing the raw data anywhere.
- We need to check how fast the transformer runs on a normal laptop, since the app is supposed to feel instant.

## Quick summary

| | |
|---|---|
| Stages done | 2 out of 7 |
| Requirements done | 7 out of 29 |
| Best model so far | Logistic Regression (macro-F1 ≈ 0.91) |
| Dataset size | ~137,000 labelled items (Bangla + English + spam/phishing) |
| Next stage | Transformer model (BanglaBERT / BanglishBERT) |
