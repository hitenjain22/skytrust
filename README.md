# SkyTrust

*An astronomy cloud forecast that tells you how often it's been wrong.*

> Work in progress (backtest baselines done; blend, live forecast, and app in progress). See [SPEC.md](SPEC.md) for the plan
> and [docs/DATA_NOTES.md](docs/DATA_NOTES.md) for what the data sources actually provide.

## Headline results

<!-- RESULTS:START -->
<!-- RESULTS:END -->

## Run it locally

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv (Python + venv manager)
uv sync                                           # installs Python 3.12 deps into .venv
uv run pytest                                     # offline test suite
uv run python -m skytrust validate-sites          # checks the ASOS stations (hits the network)
```

## Data attribution

Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).
ASOS observations courtesy of the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/), Iowa State University.
