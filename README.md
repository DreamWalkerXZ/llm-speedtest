# LLM Speedtest

A CLI tool for measuring LLM API response speed. Calculate Tokens Per Second (TPS), Time to First Token (TTFT), and end-to-end throughput for any OpenAI-compatible API.

![Demo](./demo.gif)

## Features

- **Real-time streaming metrics** - Live display of TTFT and TPS during generation
- **Multiple runs with statistics** - Run multiple tests to get mean, median, min, max, and standard deviation
- **Flexible configuration** - TOML config file or CLI arguments
- **CSV export** - Log all results to CSV for analysis
- **JSON output** - Machine-readable format for scripting
- **Token counting** - Accurate token measurement using tiktoken

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-speedtest.git
cd llm-speedtest

# Install dependencies with uv
uv sync
```

## Quick Start

Test your local Ollama instance:

```bash
uv run main.py --model qwen3.5
```

Test a remote API:

```bash
uv run main.py --model gpt-5 --base-url https://api.openai.com/v1 --api-key your-api-key
```

## Usage Examples

### Single Run

```bash
uv run main.py --model qwen3.5
```

### Multiple Runs with Statistics

```bash
uv run main.py --model qwen3.5 --runs 5 --warmup
```

The `--warmup` flag discards the first run to cold-start effects.

### Custom Prompt

```bash
uv run main.py --model qwen3.5 --prompt "Explain quantum computing in simple terms."
```

### Export to CSV

```bash
uv run main.py --model qwen3.5 --csv results.csv
```

### JSON Output (for scripting)

```bash
uv run main.py --model qwen3.5 --json
```

### With Configuration File

Copy the example config and customize:

```bash
cp llm-speedtest.example.toml llm-speedtest.toml
```

Then run without arguments:

```bash
uv run main.py --model qwen3.5
```

## Command Line Options

| Option          | Description                   | Default                     |
| --------------- | ----------------------------- | --------------------------- |
| `--model`       | Model name (required)         | -                           |
| `--base-url`    | API base URL                  | `http://localhost:11434/v1` |
| `--api-key`     | API key                       | `sk-no-key`                 |
| `--prompt`      | Input prompt                  | A short story prompt        |
| `--max-tokens`  | Maximum tokens to generate    | Unlimited                   |
| `--temperature` | Sampling temperature          | `1.0`                       |
| `--runs`        | Number of runs                | `1`                         |
| `--warmup`      | Discard first run as warmup   | `false`                     |
| `--csv`         | Save results to CSV file      | -                           |
| `--json`        | Output results in JSON format | `false`                     |
| `--config`      | Path to config file           | `llm-speedtest.toml`        |

## Metrics Explained

| Metric                         | Description                                                      |
| ------------------------------ | ---------------------------------------------------------------- |
| **TPS** (Tokens Per Second)    | Token generation speed, measured from first to last token output |
| **End-to-End TPS**             | Total throughput including TTFT (network + model startup time)   |
| **TTFT** (Time to First Token) | Latency from request start to first token received               |

## Configuration File

Create `llm-speedtest.toml` in the current directory:

```toml
base_url = "http://localhost:11434/v1"
model = "qwen3.5"
api_key = "sk-no-key"
prompt = "Write a short story about a robot learning to love."
temperature = 1.0
runs = 5
warmup = true
csv = "results.csv"
```

## Requirements

- Python 3.12+
- OpenAI-compatible API endpoint

## License

MIT
