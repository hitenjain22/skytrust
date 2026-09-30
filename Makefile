.PHONY: setup lint format test test-fast test-network tonight validate-sites data climatology dataset train hourly sensitivity walkforward spatial report app all

# All commands run inside the uv-managed virtual environment.
RUN = uv run

setup:
	uv sync

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) mypy

format:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

test:  ## everything offline, incl. slow app smoke tests (what CI runs)
	$(RUN) pytest -m "not network" --cov --cov-report=term-missing

test-fast:  ## quick loop: skips slow app smoke tests
	$(RUN) pytest

test-network:
	$(RUN) pytest -m network

validate-sites:
	$(RUN) python -m skytrust validate-sites

# Targets below are implemented in later phases (see SPEC.md Section 13).
data:
	$(RUN) python -m skytrust fetch --source asos
	$(RUN) python -m skytrust fetch --source era5
	$(RUN) python -m skytrust fetch --source prevruns

climatology:  ## 20-year reference climatology (history cached after the first run)
	$(RUN) python -m skytrust fetch --source asos --start 2004-01-01 --end 2023-12-31
	$(RUN) python -m skytrust fetch --source era5 --start 2004-01-01 --end 2023-12-31
	$(RUN) python -m skytrust build-climatology

dataset:
	$(RUN) python -m skytrust build-dataset

train:
	$(RUN) python -m skytrust train

sensitivity:  ## robustness grid over cloud definitions (~15 min)
	$(RUN) python -m skytrust sensitivity

walkforward:  ## monthly refit-and-forecast evaluation (~10 min)
	$(RUN) python -m skytrust walkforward

hourly:  ## hourly P(clear) model (build, train, evaluate)
	$(RUN) python -m skytrust hourly

spatial:  ## leave-one-site-out test (needs trained artifacts)
	$(RUN) python -m skytrust spatial

report:
	$(RUN) python -m skytrust evaluate
	$(RUN) python -m skytrust report

app:
	$(RUN) streamlit run app/streamlit_app.py

tonight:
	$(RUN) python -m skytrust tonight --site $(or $(SITE),SAC)

all: data climatology dataset train hourly sensitivity walkforward spatial report
