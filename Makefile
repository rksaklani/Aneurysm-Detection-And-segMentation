.PHONY: help install install-dev test lint format clean docs build publish

help: ## Show this help message
	@echo "Medical Image Segmentation Benchmarking Framework"
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=data --cov=models --cov=training --cov=evaluation --cov=utils --cov-report=html --cov-report=term

test-fast: ## Run fast tests only
	pytest tests/ -v -m "not slow"

test-gpu: ## Run GPU tests
	pytest tests/ -v -m "gpu"

lint: ## Run linting
	flake8 data/ models/ training/ evaluation/ utils/ main.py run_benchmark.py
	mypy data/ models/ training/ evaluation/ utils/ main.py run_benchmark.py
	bandit -r data/ models/ training/ evaluation/ utils/ main.py run_benchmark.py

format: ## Format code
	black data/ models/ training/ evaluation/ utils/ main.py run_benchmark.py
	isort data/ models/ training/ evaluation/ utils/ main.py run_benchmark.py

format-check: ## Check code formatting
	black --check data/ models/ training/ evaluation/ utils/ main.py run_benchmark.py
	isort --check-only data/ models/ training/ evaluation/ utils/ main.py run_benchmark.py

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

build: ## Build package
	python -m build

publish: ## Publish to PyPI
	python -m twine upload dist/*

security: ## Run security checks
	bandit -r data/ models/ training/ evaluation/ utils/ main.py run_benchmark.py
	safety check
	semgrep --config=auto data/ models/ training/ evaluation/ utils/ main.py run_benchmark.py

benchmark: ## Run quick benchmark
	python run_benchmark.py --data_path /path/to/ADAM_release_subjs --output_dir ./experiments/quick_benchmark

benchmark-full: ## Run full benchmark
	python main.py --config configs/benchmark_config.yaml --data_path /path/to/ADAM_release_subjs --output_dir ./experiments/full_benchmark

validate-dataset: ## Validate dataset structure
	python -c "from data.utils import validate_dataset; validate_dataset('/path/to/ADAM_release_subjs')"

setup-pre-commit: ## Setup pre-commit hooks
	pre-commit install
	pre-commit run --all-files

update-deps: ## Update dependencies
	pip-compile requirements.in
	pip-compile requirements-dev.in

docker-build: ## Build Docker image
	docker build -t medical-segmentation-benchmark .

docker-run: ## Run Docker container
	docker run -it --gpus all -v $(PWD)/data:/app/data medical-segmentation-benchmark

ci: ## Run CI pipeline locally
	make format-check
	make lint
	make test
	make security

all: ## Run all checks
	make format
	make lint
	make test
	make security
	make docs
