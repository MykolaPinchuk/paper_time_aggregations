# Time Aggregation Features for CTR (Reproducibility Repo)

### arXiv link: https://arxiv.org/abs/2601.10019

This repository contains the code and scripts needed to reproduce the results and figures used in the accompanying paper (Avazu CTR prediction; strict out-of-time evaluation; online-safe history features).

## Quickstart

1) Install deps:
- `python -m venv .venv`
- `. .venv/bin/activate && pip install -r requirements.txt`

2) Download data (not included) and place `train.csv` at `data/raw/train.csv`.

3) See `docs/REPRODUCE.md` for the full end-to-end workflow (runs → tables/figures → paper bundle).

## Paper bundle

The current arXiv-ready bundle is under `paper_bundle/arxiv_v1/`.

