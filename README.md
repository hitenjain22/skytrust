# SkyTrust

*An astronomy cloud forecast that tells you how often it's been wrong.*

> Work in progress: backtest, blend, live forecast, and app are built; deployment is next. See [SPEC.md](SPEC.md) for the plan
> and [docs/DATA_NOTES.md](docs/DATA_NOTES.md) for what the data sources actually provide.

## Headline results

<!-- RESULTS:START -->

| Night-before forecast (lead 1), test year | Brier Skill Score ↑ | False-clear rate ↓ | AUC ↑ |
|---|---|---|---|
| Blend | 0.627 [0.569, 0.676] | 9.4% [7.0%, 12.2%] | 0.948 |
| Equal-weight average (B5) | 0.596 [0.540, 0.645] | 13.2% [10.4%, 16.3%] | 0.937 |
| ECMWF calibrated (B4) | 0.515 [0.444, 0.581] | 12.5% [9.5%, 15.7%] | 0.920 |
| Climatology (B1) | 0.000 [0.000, 0.000] | 36.6% [30.0%, 43.6%] | 0.634 |

_Auto-generated from `artifacts/metrics.json` by `python -m skytrust report` (commit `0e6a89e-dirty`). 95% CIs from a week-block bootstrap. Full results: [docs/RESULTS.md](docs/RESULTS.md)._

<!-- RESULTS:END -->

## Run it locally

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv (Python + venv manager)
uv sync                                           # Python 3.12 + pinned dependencies into .venv
make app                                          # the Streamlit app -> http://localhost:8501
make tonight SITE=BIH                             # the same forecast as text in the terminal
make test                                         # all offline tests (incl. app smoke tests)
```

Rebuild everything from the raw data (downloads ~1,000 files on first run, then cached):

```bash
make all      # fetch -> build-dataset -> train -> evaluate -> report
```

## Data attribution

Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).
ASOS observations courtesy of the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/), Iowa State University.
