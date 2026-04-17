.PHONY: all data features train reports clean install test

## Run the full pipeline (data → features → train → evaluate → save artifacts)
all:
	uv run python main.py

## Download raw data and save interim parquet files to data/interim/
data:
	uv run python main.py --stage data

## Build features from interim data and save to data/processed/
features:
	uv run python main.py --stage features

## Run walk-forward backtest, evaluate, and save model outputs to models/
train:
	uv run python main.py --stage train

## Install / sync dependencies
install:
	uv sync

## Run the test suite
test:
	uv run pytest tests/ -v

## Export report notebooks to HTML in reports/
reports:
	uv run jupyter nbconvert --to html --output-dir=reports/ \
		notebooks/reports/3.0-wvs-backtest-results.ipynb \
		notebooks/reports/4.0-wvs-strategy-evaluation.ipynb

## Build documentation site
docs:
	uv run mkdocs build

## Serve documentation locally
docs-serve:
	uv run mkdocs serve

## Delete compiled Python files and cached data
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete all generated artifacts (data, models, reports)
clean-all: clean
	rm -rf data/interim/* data/processed/* models/* reports/figures/* reports/*.html
