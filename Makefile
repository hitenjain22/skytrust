.PHONY: setup lint format test test-network validate-sites data dataset train report app all

# All commands run inside the uv-managed virtual environment.
RUN = uv run

setup:
	uv sync

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

test:
	$(RUN) pytest --cov --cov-report=term-missing

test-network:
	$(RUN) pytest -m network

validate-sites:
	$(RUN) python -m skytrust validate-sites

# Targets below are implemented in later phases (see SPEC.md Section 13).
data:
	$(RUN) python -m skytrust fetch --source asos
	$(RUN) python -m skytrust fetch --source era5
	$(RUN) python -m skytrust fetch --source prevruns

dataset:
	$(RUN) python -m skytrust build-dataset

train:
	$(RUN) python -m skytrust train

report:
	$(RUN) python -m skytrust evaluate
	$(RUN) python -m skytrust report

app:
	$(RUN) streamlit run app/streamlit_app.py

all: data dataset train report
