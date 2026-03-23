# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TPS (Tokens Per Second) Calculator for Text Models. Measures inference performance by calculating TPS from streaming API responses, tracking time-to-first-token and end-to-end throughput.

## Commands

```bash
# Install dependencies
uv sync

# Run the calculator
uv run main.py --model qwen3.5 --base-url http://localhost:11434/v1

# Run with custom prompt
uv run main.py --model qwen3.5 --prompt "Your prompt here"

# Multiple runs for statistics
uv run main.py --model qwen3.5 --runs 5 --warmup

# Export to CSV
uv run main.py --model qwen3.5 --csv results.csv

# JSON output (for scripting)
uv run main.py --model qwen3.5 --json

# Type checking
mypy main.py
```

## Architecture

Single-file CLI application (`main.py`) organized into sections:

- **Configuration**: TOML config loading, defaults for base_url (Ollama localhost:11434), api_key (sk-no-key)
- **CSV Export**: Append-only CSV logging with sanitized output previews
- **Token Counting**: tiktoken integration with fallback to cl100k_base
- **UI Components**: Rich library for live panel updates during streaming
- **Statistics**: Mean, median, min, max, stddev for TPS and TTFT across runs
- **Core TPS Calculation**: Streaming response processing with precise timing
- **CLI Entry Point**: Argument parsing with config file precedence

## Key Metrics

- **TPS**: Token generation speed (tokens between first and last token)
- **End-to-End TPS**: Total throughput including time-to-first-token
- **TTFT**: Time to First Token (network + model startup latency)

## Configuration

Copy `llm-speedtest.example.toml` to `llm-speedtest.toml` for persistent configuration. Config values are overridden by CLI arguments.

## Dependencies

- **openai**: OpenAI-compatible API client
- **rich**: Terminal UI with live updates
- **tiktoken**: Token counting
