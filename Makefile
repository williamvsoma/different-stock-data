.PHONY: all data features train reports clean install

## Run the full pipeline (data → features → train → evaluate → save artifacts)
all:
	uv run python main.py

## Install / sync dependencies
install:
	uv sync

## Export report notebooks to HTML in reports/
reports:
	uv run jupyter nbconvert --to html --output-dir=reports/ \
		notebooks/reports/3.0-wvs-backtest-results.ipynb \
		notebooks/reports/4.0-wvs-strategy-evaluation.ipynb

## Delete compiled Python files and cached data
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete all generated artifacts (data, models, reports)
clean-all: clean
	rm -rf data/interim/* data/processed/* models/* reports/figures/* reports/*.html
