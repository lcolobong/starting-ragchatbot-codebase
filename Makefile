.PHONY: format check-format test quality fix

format:
	uv run black .

check-format:
	uv run black --check .

test:
	uv run pytest

quality: check-format test

fix: format test
