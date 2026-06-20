# How to run this project on another PC

This is the v1.0 stage of the project: the **data pipeline + classical models**.
There is no web app yet (that comes in a later stage), so "running it" means
setting up Python, building the dataset, and training the models.

You have two ways to do this. **Option A is the easiest** if you just want to get
it working on another machine quickly.

---

## What you need first (both options)

- **Python 3.11 or newer** — https://www.python.org/downloads/
- **Git** — https://git-scm.com/downloads
- About **3 GB free disk space** (data + libraries).

---

## Option A — Copy the whole folder (easiest, no internet needed)

Best for a demo or moving to a friend's PC. You already have the data and the
trained models on this machine, so you just copy them along.

1. Copy the entire project folder (`Fake News Detection`) to the other PC — USB
   drive, Google Drive, etc.
   - **Important:** delete the `.venv` folder before copying (it's ~440 MB and
     will not work on another PC — you will make a fresh one). Keep `data` and
     `models`.

2. On the other PC, open a terminal **inside the project folder** and make a
   fresh virtual environment:

   ```bash
   # Mac / Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Check it works (runs the fast tests):

   ```bash
   python -m pytest -m "not slow"
   ```

   If you see something like `100 passed`, you're done. The dataset and trained
   models are already there, so nothing else to build.

---

## Option B — Set up fresh from Git (clean, re-downloads everything)

Use this if you cloned the repo and the `data/` and `models/` folders are empty
(they are not stored in git on purpose — the datasets are large/licensed).

1. Get the code and enter the folder:

   ```bash
   git clone <your-repo-url>
   cd "Fake News Detection"
   ```

2. Make the virtual environment and install libraries (same as Option A steps 2–3):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Set up a Kaggle token** (needed to download the datasets — it's free):
   - Make a Kaggle account → https://www.kaggle.com
   - Go to your account settings → **Create New API Token**. This downloads a
     file called `kaggle.json`.
   - Put it here:
     - Mac / Linux: `~/.kaggle/kaggle.json`
     - Windows: `C:\Users\<you>\.kaggle\kaggle.json`

4. Run the pipeline, in order:

   ```bash
   python -m src.data.acquire          # downloads the datasets (~1 GB) → data/raw/
   python -m src.data.build_corpus     # cleans + builds the corpus → data/processed/
   python -m src.models.train_classical # trains + saves the models → models/
   ```

   Each step prints what it's doing. After the last one you'll have
   `models/best_model.joblib` and `models/vectorizer.joblib`.

5. Check everything:

   ```bash
   python -m pytest -m "not slow"
   ```

---

## Show a quick live demo

There is no web app yet (that is a later stage), but the trained model works,
so you can classify text from the command line. Run this from the project
folder with the virtual environment active:

```bash
# classify a few built-in example messages
python demo_predict.py

# classify your own text
python demo_predict.py --text "Your account is locked. Verify now at http://secure-login.ru"
```

It prints, for each input: the verdict (REAL / FAKE / MALICIOUS), a confidence
percentage, the score for all three classes, and the words that pushed the
decision. This needs the trained model in `models/` (already there if you used
Option A, or after running `train_classical` in Option B).

---

## Common problems

- **`python: command not found`** → try `python3` instead (and `pip3`).
- **`No module named src` when running `pytest`** → run it as
  `python -m pytest -m "not slow"` (the `python -m` part puts the project folder
  on the path). Always run from the **top project folder** with the virtual
  environment active.
- **`No module named normalizer` / preprocessing errors** → the csebuetnlp
  normalizer didn't install. Run it manually:
  `pip install git+https://github.com/csebuetnlp/normalizer`
- **Kaggle "401 / forbidden" or "could not authenticate"** → the `kaggle.json`
  file is missing or in the wrong place (see Option B, step 3).

---

## What's in each folder

| Folder | What it is | In git? |
|--------|-----------|---------|
| `src/` | the code (data pipeline + models) | yes |
| `tests/` | automated tests | yes |
| `data/raw/` | downloaded datasets (~927 MB) | no — re-download or copy |
| `data/processed/` | the built corpus (~170 MB) | no — rebuild or copy |
| `models/` | the trained model files (~6 MB) | no — retrain or copy |
| `.venv/` | the Python libraries | no — always make fresh |
