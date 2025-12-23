.PHONY: install test

install: download-stockfish
	uv sync
	uv run pre-commit install

download-stockfish:
	@chmod +x scripts/download_stockfish.sh
	@./scripts/download_stockfish.sh

test:
	uv run pytest
