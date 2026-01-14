# Data download (Avazu CTR Prediction)

Competition page: https://www.kaggle.com/competitions/avazu-ctr-prediction

## Option A: Kaggle API (recommended)

1. Put your Kaggle API token at `~/.kaggle/kaggle.json` (permissions `600`).
2. From repo root:
   - `kaggle competitions download -c avazu-ctr-prediction -p data/raw`
   - `unzip -n data/raw/avazu-ctr-prediction.zip -d data/raw`

Expected path for the baseline script: `data/raw/train.csv`

## Option B: Manual download

1. Download the competition data via browser.
2. Extract `train.csv` into `data/raw/train.csv`.

