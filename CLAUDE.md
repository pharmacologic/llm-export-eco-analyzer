# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A single-file Python CLI tool (`claude_chatgpt_eco_analysis.py`) that parses `conversations.json` exports from Claude.ai or ChatGPT and estimates environmental impact (energy, CO₂, water) using the EcoLogits library. There is also an HTML version (`llm-impact-analyzer.html`) that runs in the browser.

Feature parity between the two implementations is tracked in [`FEATURE_PARITY.md`](./FEATURE_PARITY.md). Update it whenever adding a feature to either version.

## Installation & Running

```bash
pip install ecologits tiktoken rich

# Basic usage
python claude_chatgpt_eco_analysis.py conversations.json

# With options
python claude_chatgpt_eco_analysis.py conversations.json --model claude-sonnet-4-20250514
python claude_chatgpt_eco_analysis.py conversations.json --mix "claude-sonnet-4-20250514:75,claude-haiku-4-5-20251001:25"
python claude_chatgpt_eco_analysis.py conversations.json --zone FRA --output report.json --top 50
```

## Architecture

The script is organized into distinct sections (marked with `═` comment headers):

1. **CONFIGURATION** — `CLAUDE_MODEL_MAP` and `CHATGPT_MODEL_MAP` dicts map export slugs to EcoLogits model names. `DEFAULT_CLAUDE_MODEL`, `DEFAULT_OPENAI_MODEL`, `DEFAULT_LATENCY` set fallbacks.

2. **DATA STRUCTURES** — `RequestImpact` dataclass holds per-message results. `AggImpact` dataclass accumulates totals with `add()`.

3. **EXPORT FORMAT DETECTION** — `detect_export_type()` distinguishes Claude (has `chat_messages`/`uuid`) from ChatGPT (has `mapping`/`conversation_id`).

4. **MODEL DETECTION** — `detect_model_and_provider()` dispatches to `_detect_claude_model()` or `_detect_chatgpt_model()`. Claude reads top-level `model` key or message metadata; ChatGPT traverses the mapping tree to find `model_slug`.

5. **PARSING** — `extract_messages()` normalizes both formats. For Claude, counts output tokens including text blocks, `tool_use` inputs (artifacts), and `thinking` blocks. For ChatGPT, reads `content.parts`. `parse_conversations()` orchestrates everything, handles `--model`/`--mix` priority, calls EcoLogits, and applies fallback models when a model isn't registered.

6. **AGGREGATION** — `aggregate()` groups `RequestImpact` results into cumulative, by-conversation, by-week, by-month, and by-model `AggImpact` dicts.

7. **OUTPUT** — `print_cumulative()`, `print_agg_table()`, and `equivalents()` handle terminal display (Rich tables when available, plain text fallback). `save_json()` serializes results.

8. **CLI** — `main()` handles `argparse`, loads JSON, calls `parse_conversations()` → `aggregate()` → print functions.

## Key Design Decisions

- **Token counting**: Uses `tiktoken` with `cl100k_base` encoding (works for both Claude and GPT). Falls back to word-count × 1.35 if not installed.
- **EcoLogits call signature**: `llm_impacts(provider, model_name, output_token_count, request_latency, electricity_mix_zone?)`. Only output tokens are counted (EcoLogits limitation).
- **Model fallback chain**: If EcoLogits returns `None` for a model, tries provider-specific fallbacks before skipping the message.
- **Latency**: For Claude exports, extracted from `thinking` block timestamps; otherwise uses `DEFAULT_LATENCY = 30.0` seconds.
- **Model priority**: data-detected model > `--model` override > `--mix` weighted average.
