#!/usr/bin/env python3
"""LLM API Speed Testing Tool.

Calculates TTFT (Time to First Token), TPS (Tokens Per Second), and End-to-End TPS.
Supports OpenAI Compatible APIs with custom base_url and model name.
"""

# Standard Library
import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, TypedDict

# Third Party
import tiktoken
import tomllib
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich import box

# Constants
DEFAULT_CONFIG_FILE = "llm-speedtest.toml"
DEFAULT_API_KEY = "sk-no-key"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_PROMPT = "Write a short story about a robot learning to love."
CSV_PREVIEW_LENGTH = 8
OUTPUT_VISIBLE_LINES = 8
PANEL_WIDTH = 80
PANEL_HEIGHT = 10


# Type Definitions
class TPSResult(TypedDict):
    """Result of a successful TPS calculation."""

    tps: float
    end_to_end_tps: float
    total_tokens: int
    time_to_first_token: float
    total_time: float
    generation_time: float
    output_text: str


class TPSResultError(TypedDict):
    """Result when no tokens are generated."""

    error: str
    tps: float
    end_to_end_tps: float
    total_tokens: int
    time_to_first_token: float
    total_time: float
    generation_time: float
    output_text: str


class RunStatistics(TypedDict):
    """Statistics aggregated from multiple runs."""

    mean_tps: float
    median_tps: float
    min_tps: float
    max_tps: float
    std_tps: float
    mean_end_to_end_tps: float
    median_end_to_end_tps: float
    min_end_to_end_tps: float
    max_end_to_end_tps: float
    std_end_to_end_tps: float
    mean_ttft: float
    median_ttft: float
    min_ttft: float
    max_ttft: float
    std_ttft: float
    runs: int
    results: list[TPSResult]


AnyResult = TPSResult | TPSResultError


# =============================================================================
# Configuration
# =============================================================================


def load_config(config_path: Optional[str] = None) -> dict:
    """Load config from TOML file.

    Returns empty dict if no config found.
    Raises FileNotFoundError if explicit config path doesn't exist.
    """
    path = Path(config_path or DEFAULT_CONFIG_FILE)
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    if config_path:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return {}


# =============================================================================
# CSV Export
# =============================================================================


def save_to_csv(
    result: AnyResult,
    csv_path: str,
    model: str,
    prompt: str,
    temperature: float,
    is_warmup: bool = False,
) -> None:
    """Save results to CSV file.

    Creates file with headers if it doesn't exist, otherwise appends.
    """
    path = Path(csv_path)
    file_exists = path.exists()

    fieldnames = [
        "timestamp",
        "model",
        "prompt",
        "temperature",
        "warmup",
        "tps",
        "end_to_end_tps",
        "total_tokens",
        "time_to_first_token",
        "generation_time",
        "total_time",
        "output_text",
    ]

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        output_preview = _sanitize_output_preview(result["output_text"])

        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "warmup": is_warmup,
                "tps": result["tps"],
                "end_to_end_tps": result["end_to_end_tps"],
                "total_tokens": result["total_tokens"],
                "time_to_first_token": result["time_to_first_token"],
                "generation_time": result["generation_time"],
                "total_time": result["total_time"],
                "output_text": output_preview,
            }
        )


def _sanitize_output_preview(text: str) -> str:
    """Sanitize output text for CSV preview."""
    preview = text[:CSV_PREVIEW_LENGTH]
    preview = "".join(
        c if c.isprintable() and c not in ',"\n\r\t' else " " for c in preview
    )
    return preview.strip()


# =============================================================================
# Token Counting
# =============================================================================


@lru_cache(maxsize=16)
def _get_encoding(model: str) -> tiktoken.Encoding:
    """Get tiktoken encoding for a model (cached)."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def get_token_count(text: str, model: str) -> int:
    """Count tokens using tiktoken."""
    encoding = _get_encoding(model)
    return len(encoding.encode(text))


# =============================================================================
# UI Components
# =============================================================================


def _display_prompt_panel(console: Console, prompt: str, model: str) -> None:
    """Display prompt and model info in a panel."""
    info_panel = Panel(
        f"[bold]Prompt:[/bold] {prompt}\n[bold]Model:[/bold] {model}",
        title="[bold bright_blue]LLM Speedtest[/bold bright_blue]",
        border_style="bright_blue",
        width=PANEL_WIDTH,
    )
    console.print(info_panel)


def _build_run_title(run_number: int, total_runs: int, is_warmup: bool) -> str:
    """Build the run title string."""
    warmup_suffix = " (warmup)" if is_warmup else ""
    return f"[bold]Run [cyan]{run_number}[/cyan]/{total_runs}{warmup_suffix}[/bold]"


def _build_panel_title(
    run_number: int,
    total_runs: int,
    time_to_first_token: float,
    tps: Optional[float],
    is_warmup: bool,
) -> str:
    """Build the panel title string."""
    warmup_suffix = " (warmup)" if is_warmup else ""
    run_prefix = (
        f"[bold]Run [cyan]{run_number}[/cyan]/{total_runs}{warmup_suffix} | [/bold]"
    )
    ttft_str = (
        f"[bold]TTFT: [spring_green4]{time_to_first_token:.2f}[/spring_green4] s[/bold]"
    )

    if tps is not None:
        tps_str = f"[bold] | TPS: [dark_orange]{tps:.2f}[/dark_orange] tokens/s[/bold]"
        return run_prefix + ttft_str + tps_str
    return run_prefix + ttft_str


def _create_live_panel(
    run_number: int, total_runs: int, is_warmup: bool = False
) -> Panel:
    """Create the initial live panel with loading spinner."""
    loading_spinner = Spinner("dots", text="Waiting for first token...")
    title = _build_run_title(run_number, total_runs, is_warmup)
    return Panel(
        loading_spinner,
        title=title,
        border_style="bright_blue",
        height=PANEL_HEIGHT,
        width=PANEL_WIDTH,
    )


def _update_live_panel(
    live: Live,
    full_output: list[str],
    time_to_first_token: float,
    run_number: int,
    total_runs: int,
    tps: Optional[float] = None,
    is_warmup: bool = False,
) -> None:
    """Update the live panel with current output."""
    full_text = "".join(full_output)
    lines = full_text.split("\n")
    visible_lines = (
        lines[-OUTPUT_VISIBLE_LINES:] if len(lines) > OUTPUT_VISIBLE_LINES else lines
    )
    visible_text = Text("\n".join(visible_lines))

    panel_title = _build_panel_title(
        run_number, total_runs, time_to_first_token, tps, is_warmup
    )
    live.update(
        Panel(
            visible_text,
            title=panel_title,
            border_style="bright_blue",
            height=PANEL_HEIGHT,
            width=PANEL_WIDTH,
        )
    )


# =============================================================================
# Statistics
# =============================================================================


def _calculate_statistics(results: list[AnyResult]) -> RunStatistics:
    """Calculate statistics from multiple runs."""
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        return _empty_statistics()

    tps_values = [r["tps"] for r in valid_results]
    end_to_end_tps_values = [r["end_to_end_tps"] for r in valid_results]
    ttft_values = [r["time_to_first_token"] for r in valid_results]

    return {
        "mean_tps": statistics.mean(tps_values),
        "median_tps": statistics.median(tps_values),
        "min_tps": min(tps_values),
        "max_tps": max(tps_values),
        "std_tps": statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0,
        "mean_end_to_end_tps": statistics.mean(end_to_end_tps_values),
        "median_end_to_end_tps": statistics.median(end_to_end_tps_values),
        "min_end_to_end_tps": min(end_to_end_tps_values),
        "max_end_to_end_tps": max(end_to_end_tps_values),
        "std_end_to_end_tps": statistics.stdev(end_to_end_tps_values)
        if len(end_to_end_tps_values) > 1
        else 0.0,
        "mean_ttft": statistics.mean(ttft_values),
        "median_ttft": statistics.median(ttft_values),
        "min_ttft": min(ttft_values),
        "max_ttft": max(ttft_values),
        "std_ttft": statistics.stdev(ttft_values) if len(ttft_values) > 1 else 0.0,
        "runs": len(valid_results),
        "results": valid_results,
    }


def _empty_statistics() -> RunStatistics:
    """Return empty statistics (all zeros)."""
    return {
        "mean_tps": 0.0,
        "median_tps": 0.0,
        "min_tps": 0.0,
        "max_tps": 0.0,
        "std_tps": 0.0,
        "mean_end_to_end_tps": 0.0,
        "median_end_to_end_tps": 0.0,
        "min_end_to_end_tps": 0.0,
        "max_end_to_end_tps": 0.0,
        "std_end_to_end_tps": 0.0,
        "mean_ttft": 0.0,
        "median_ttft": 0.0,
        "min_ttft": 0.0,
        "max_ttft": 0.0,
        "std_ttft": 0.0,
        "runs": 0,
        "results": [],
    }


def _display_statistics_table(console: Console, stats: RunStatistics) -> None:
    """Display statistics from multiple runs."""
    table = Table(box=box.ROUNDED, border_style="red1", width=PANEL_WIDTH)
    table.add_column("Metric", style="bold", width=25)
    table.add_column("Value")

    # Runs
    table.add_row("Runs", f"{stats['runs']}")
    table.add_row("", "")

    # Time to First Token section
    table.add_row(
        "Mean TTFT", f"[bold bright_blue]{stats['mean_ttft']:.2f}[/bold bright_blue] s"
    )
    table.add_row("Median TTFT", f"{stats['median_ttft']:.2f} s")
    table.add_row("Min TTFT", f"{stats['min_ttft']:.2f} s")
    table.add_row("Max TTFT", f"{stats['max_ttft']:.2f} s")
    table.add_row("Std Dev", f"{stats['std_ttft']:.2f} s")
    table.add_row("", "")

    # TPS section (generation)
    table.add_row(
        "Mean TPS",
        f"[bold bright_blue]{stats['mean_tps']:.2f}[/bold bright_blue] tokens/s",
    )
    table.add_row("Median TPS", f"{stats['median_tps']:.2f} tokens/s")
    table.add_row("Min TPS", f"{stats['min_tps']:.2f} tokens/s")
    table.add_row("Max TPS", f"{stats['max_tps']:.2f} tokens/s")
    table.add_row("Std Dev", f"{stats['std_tps']:.2f} tokens/s")
    table.add_row("", "")

    # End-to-End TPS section
    table.add_row(
        "Mean End-to-End TPS",
        f"[bold bright_blue]{stats['mean_end_to_end_tps']:.2f}[/bold bright_blue] tokens/s",
    )
    table.add_row(
        "Median End-to-End TPS", f"{stats['median_end_to_end_tps']:.2f} tokens/s"
    )
    table.add_row("Min End-to-End TPS", f"{stats['min_end_to_end_tps']:.2f} tokens/s")
    table.add_row("Max End-to-End TPS", f"{stats['max_end_to_end_tps']:.2f} tokens/s")
    table.add_row("Std Dev", f"{stats['std_end_to_end_tps']:.2f} tokens/s")

    console.print(table)


# =============================================================================
# Core TPS Calculation
# =============================================================================


def calculate_tps(
    base_url: str,
    model: str,
    prompt: str,
    api_key: Optional[str] = DEFAULT_API_KEY,
    max_tokens: Optional[int] = None,
    temperature: float = 1.0,
    quiet: bool = False,
    show_prompt_panel: bool = True,
    run_number: int = 1,
    total_runs: int = 1,
    is_warmup: bool = False,
) -> AnyResult:
    """Calculate TPS (Tokens Per Second) for a text model.

    Args:
        base_url: API base URL
        model: Model name
        prompt: Input prompt
        api_key: API key
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        quiet: Suppress all progress output
        show_prompt_panel: Show prompt panel at start
        run_number: Current run number (for display)
        total_runs: Total number of runs (for display)
        is_warmup: This is a warmup run (for display)

    Returns:
        Dictionary with TPS metrics or error info.
    """
    console = Console()
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Initialize timing variables
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    output_chunks: list[str] = []

    # Display prompt panel if needed
    if show_prompt_panel and not quiet:
        _display_prompt_panel(console, prompt, model)

    # Create live panel
    full_output: list[str] = []
    live_panel = (
        None if quiet else _create_live_panel(run_number, total_runs, is_warmup)
    )

    with Live(live_panel, console=console, refresh_per_second=1) as live:
        start_time = time.time()

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            if quiet:
                raise
            console.print(f"[red]Error: API request failed - {e}[/red]")
            raise

        # Process streaming response
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()

                last_token_time = time.time()
                content = chunk.choices[0].delta.content
                output_chunks.append(content)

                if not quiet:
                    full_output.append(content)
                    # first_token_time is guaranteed non-None here
                    assert first_token_time is not None
                    _update_live_panel(
                        live,
                        full_output,
                        first_token_time - start_time,
                        run_number,
                        total_runs,
                        is_warmup=is_warmup,
                    )

        # Calculate and display final TPS
        output_text = "".join(output_chunks)
        token_count = get_token_count(output_text, model)
        tps = _compute_tps_metrics(token_count, first_token_time, last_token_time)

        if tps is not None and not quiet and first_token_time is not None:
            _update_live_panel(
                live,
                full_output,
                first_token_time - start_time,
                run_number,
                total_runs,
                tps,
                is_warmup,
            )

    # Return result
    return _build_result(
        output_text, token_count, first_token_time, last_token_time, start_time
    )


def _compute_tps_metrics(
    token_count: int,
    first_token_time: Optional[float],
    last_token_time: Optional[float],
) -> Optional[float]:
    """Compute TPS from token count and timing data."""
    if first_token_time is None or last_token_time is None:
        return None

    generation_time = last_token_time - first_token_time
    return token_count / generation_time if generation_time > 0 else 0


def _build_result(
    output_text: str,
    token_count: int,
    first_token_time: Optional[float],
    last_token_time: Optional[float],
    start_time: float,
) -> AnyResult:
    """Build the result dictionary from output and timing data."""
    if first_token_time is None or last_token_time is None:
        return {
            "error": "No tokens generated",
            "tps": 0.0,
            "end_to_end_tps": 0.0,
            "total_tokens": 0,
            "time_to_first_token": 0.0,
            "total_time": 0.0,
            "generation_time": 0.0,
            "output_text": output_text,
        }

    generation_time = last_token_time - first_token_time
    total_time = last_token_time - start_time
    tps = token_count / generation_time if generation_time > 0 else 0
    end_to_end_tps = token_count / total_time if total_time > 0 else 0

    return {
        "tps": tps,
        "end_to_end_tps": end_to_end_tps,
        "total_tokens": token_count,
        "time_to_first_token": first_token_time - start_time,
        "total_time": total_time,
        "generation_time": generation_time,
        "output_text": output_text,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point."""
    parser = _build_arg_parser()
    args = _parse_args(parser)
    console = Console()

    # Run TPS calculations
    all_results = _run_all_tests(args, console)

    # Display results
    _display_results(console, all_results, args)

    # Show CSV save confirmation (only in non-JSON mode)
    if args.csv and not args.json:
        console.print(
            f"Results saved to [bold bright_blue]{args.csv}[/bold bright_blue]"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Calculate TTFT, TPS, and End-to-End TPS for text models"
    )
    parser.add_argument("--config", type=str, help="Path to config file (TOML format)")
    parser.add_argument("--base-url", type=str, help="API base URL (overrides config)")
    parser.add_argument("--model", type=str, help="Model name (e.g., qwen3.5, gpt-5)")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )
    parser.add_argument("--csv", type=str, help="Save results to CSV file")
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of times to run (default: 1)"
    )
    parser.add_argument(
        "--warmup", action="store_true", help="Discard first run as warmup"
    )
    return parser


def _parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse arguments with config file support."""
    args, remaining = parser.parse_known_args()
    config = load_config(args.config)

    parser.set_defaults(
        base_url=config.get("base_url", DEFAULT_BASE_URL),
        model=config.get("model"),
        prompt=config.get("prompt", DEFAULT_PROMPT),
        api_key=config.get("api_key", DEFAULT_API_KEY),
        max_tokens=config.get("max_tokens"),
        temperature=config.get("temperature", 1.0),
        json=config.get("json", args.json),
        csv=config.get("csv"),
        runs=config.get("runs", args.runs),
        warmup=config.get("warmup", args.warmup),
    )
    args = parser.parse_args(remaining)

    if not args.model:
        parser.error("--model is required (provide via CLI or config file)")

    return args


def _run_all_tests(args: argparse.Namespace, console: Console) -> list[AnyResult]:
    """Run all TPS tests and collect results."""
    all_results: list[AnyResult] = []
    total_runs = args.runs + (1 if args.warmup else 0)

    for run_idx in range(total_runs):
        is_warmup = args.warmup and run_idx == 0
        is_multi_run = total_runs > 1
        show_prompt_panel = not is_multi_run

        if not args.json and is_multi_run and run_idx == 0:
            _display_prompt_panel(console, args.prompt, args.model)

        result = calculate_tps(
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            quiet=args.json,
            show_prompt_panel=show_prompt_panel,
            run_number=run_idx + 1,
            total_runs=total_runs,
            is_warmup=is_warmup,
        )

        if not is_warmup:
            all_results.append(result)

        if args.csv:
            save_to_csv(
                result, args.csv, args.model, args.prompt, args.temperature, is_warmup
            )

    return all_results


def _display_results(
    console: Console, all_results: list[AnyResult], args: argparse.Namespace
) -> None:
    """Display statistics or JSON results."""
    stats = _calculate_statistics(all_results)

    if args.json:
        output = {
            "statistics": {k: v for k, v in stats.items() if k != "results"},
            "runs": all_results,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        _display_statistics_table(console, stats)


if __name__ == "__main__":
    main()
