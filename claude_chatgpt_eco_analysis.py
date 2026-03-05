#!/usr/bin/env python3
"""
Claude.ai & ChatGPT Data Export — Environmental Impact Analyzer
==============================================================
Parses claude.ai and ChatGPT conversations.json exports and uses EcoLogits to estimate
the energy, GHG, and water consumption of your LLM usage.

Requirements:
    pip install ecologits tiktoken rich

Usage:
    # Single fallback model when export lacks model metadata
    python claude_eco_analysis.py conversations.json --model claude-sonnet-4-20250514

    # Weighted model mix (ratios are normalised automatically)
    python claude_eco_analysis.py conversations.json \\
        --mix "claude-sonnet-4-20250514:75,claude-haiku-3-5-20241022:10,claude-haiku-4-5-20251001:15"

    # Save full results to JSON and use French electricity mix
    python claude_eco_analysis.py conversations.json --output report.json --zone FRA
"""

import argparse
import json
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── optional pretty output ──────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    class _Console:
        def print(self, *a, **kw): print(*a)
        def rule(self, t=""): print(f"\n{'─'*60} {t} {'─'*60}\n" if t else "─"*120)
    console = _Console()

# ── EcoLogits ───────────────────────────────────────────────────────────────
try:
    from ecologits.tracers.utils import llm_impacts
    from ecologits.utils.range_value import RangeValue
    ECOLOGITS_OK = True
except ImportError:
    ECOLOGITS_OK = False
    print("ERROR: ecologits not installed. Run: pip install ecologits", file=sys.stderr)
    sys.exit(1)

# ── tiktoken (token counting) ────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")   # works for Claude and GPT
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text or ""))
except ImportError:
    warnings.warn("tiktoken not installed — using word-count approximation (less accurate).")
    def count_tokens(text: str) -> int:
        return max(1, int(len((text or "").split()) * 1.35))


# ════════════════════════════════════════════════════════════════════════════
# UNIT FORMATTING
# ════════════════════════════════════════════════════════════════════════════

def fmt_energy(kwh: float) -> tuple[float, str]:
    """
    Convert kWh to the most readable unit.
    Returns (value, unit_label).
      <1 Wh      → mWh
      1–999 Wh   → Wh
      ≥1000 Wh   → kWh
    """
    wh = kwh * 1000
    if wh < 1.0:
        return wh * 1000, "mWh"
    elif wh < 1000.0:
        return wh, "Wh"
    else:
        return kwh, "kWh"

def fmt_energy_str(kwh: float, decimals: int = 3) -> str:
    v, u = fmt_energy(kwh)
    return f"{v:.{decimals}f} {u}"

def fmt_energy_range_str(mid_kwh: float, lo_kwh: float, hi_kwh: float) -> str:
    """Format energy with range, using consistent units based on mid value."""
    mid_v, u = fmt_energy(mid_kwh)
    # Convert lo/hi to same unit
    if u == "mWh":
        lo_v = lo_kwh * 1e6
        hi_v = hi_kwh * 1e6
    elif u == "Wh":
        lo_v = lo_kwh * 1000
        hi_v = hi_kwh * 1000
    else:
        lo_v = lo_kwh
        hi_v = hi_kwh
    return f"{mid_v:.3f} {u}  [{lo_v:.3f}–{hi_v:.3f} {u}]"


def fmt_ghg(kgco2: float) -> tuple[float, str]:
    """
    Convert kgCO₂eq to the most readable unit.
      <1 g    → mg CO₂eq
      1–999 g → g CO₂eq
      ≥1 kg   → kg CO₂eq
    """
    g = kgco2 * 1000
    if g < 1.0:
        return g * 1000, "mg CO₂eq"
    elif g < 1000.0:
        return g, "g CO₂eq"
    else:
        return kgco2, "kg CO₂eq"

def fmt_ghg_str(kgco2: float, decimals: int = 2) -> str:
    v, u = fmt_ghg(kgco2)
    return f"{v:.{decimals}f} {u}"

def fmt_ghg_range_str(mid_kgco2: float, lo_kgco2: float, hi_kgco2: float) -> str:
    mid_v, u = fmt_ghg(mid_kgco2)
    if u == "mg CO₂eq":
        lo_v = lo_kgco2 * 1e6
        hi_v = hi_kgco2 * 1e6
    elif u == "g CO₂eq":
        lo_v = lo_kgco2 * 1000
        hi_v = hi_kgco2 * 1000
    else:
        lo_v = lo_kgco2
        hi_v = hi_kgco2
    return f"{mid_v:.2f} {u}  [{lo_v:.2f}–{hi_v:.2f} {u}]"


def fmt_water(liters: float) -> tuple[float, str]:
    """
    Convert liters to the most readable unit.
      <1 mL   → µL  (sub-mL, rare)
      1–9999 mL → mL
      ≥10 L   → L
    """
    ml = liters * 1000
    if ml < 1.0:
        return ml * 1000, "µL"
    elif ml < 10000.0:
        return ml, "mL"
    else:
        return liters, "L"

def fmt_water_str(liters: float, decimals: int = 1) -> str:
    v, u = fmt_water(liters)
    return f"{v:.{decimals}f} {u}"

def fmt_water_range_str(mid_l: float, lo_l: float, hi_l: float) -> str:
    mid_v, u = fmt_water(mid_l)
    if u == "µL":
        lo_v = lo_l * 1e6
        hi_v = hi_l * 1e6
    elif u == "mL":
        lo_v = lo_l * 1000
        hi_v = hi_l * 1000
    else:
        lo_v = lo_l
        hi_v = hi_l
    return f"{mid_v:.1f} {u}  [{lo_v:.1f}–{hi_v:.1f} {u}]"


def _col_unit_label(kwh_values: list[float], col: str) -> tuple[str, str, float]:
    """
    For a table column, pick the best unit based on the median value,
    and return (header_label, unit_suffix, scale_factor_from_kwh).
    col is one of 'energy', 'ghg', 'water'.
    """
    if not kwh_values:
        if col == "energy":   return "Energy (Wh)", "Wh", 1000.0
        if col == "ghg":      return "GHG (g CO₂eq)", "g CO₂eq", 1000.0
        if col == "water":    return "Water (mL)", "mL", 1000.0

    vals = sorted(kwh_values)
    median = vals[len(vals) // 2]

    if col == "energy":
        _, u = fmt_energy(median)
        scale = {"mWh": 1e6, "Wh": 1e3, "kWh": 1.0}[u]
        return f"Energy ({u})", u, scale
    elif col == "ghg":
        _, u = fmt_ghg(median)
        scale = {"mg CO₂eq": 1e6, "g CO₂eq": 1e3, "kg CO₂eq": 1.0}[u]
        return f"GHG ({u})", u, scale
    elif col == "water":
        _, u = fmt_water(median)
        scale = {"µL": 1e6, "mL": 1e3, "L": 1.0}[u]
        return f"Water ({u})", u, scale
    return col, "", 1.0


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# Map Claude model slugs → ecologits model names.
# ecologits uses the exact Anthropic model name strings.
# If a slug doesn't match, we fall back to a sensible default.
def parse_model_mix(mix_str: str) -> list[tuple[str, float]]:
    """
    Parse a model mix string like "modelA:75,modelB:25" into a normalised
    list of (model_name, weight) tuples that sum to 1.0.
    """
    parts = [p.strip() for p in mix_str.split(",") if p.strip()]
    pairs = []
    for part in parts:
        if ":" in part:
            name, _, weight = part.rpartition(":")
            pairs.append((name.strip(), float(weight.strip())))
        else:
            pairs.append((part.strip(), 1.0))
    total = sum(w for _, w in pairs)
    return [(name, w / total) for name, w in pairs]


CLAUDE_MODEL_MAP = {
    # claude.ai may surface any of these in the export
    "claude-opus-4-5":          "claude-opus-4-5-20250514",
    "claude-opus-4-1":          "claude-opus-4-1-20250805",
    "claude-sonnet-4-5":        "claude-sonnet-4-5-20251001",
    "claude-sonnet-4":          "claude-sonnet-4-20250514",
    "claude-haiku-4-5":         "claude-haiku-4-5-20251001",
    "claude-3-7-sonnet":        "claude-sonnet-3-7-20250219",
    "claude-3-5-sonnet":        "claude-sonnet-3-5-20241022",
    "claude-3-5-haiku":         "claude-haiku-3-5-20241022",
    "claude-3-opus":            "claude-opus-3-20240229",
    "claude-3-sonnet":          "claude-sonnet-3-20240229",
    "claude-3-haiku":           "claude-haiku-3-20240307",
    "claude-2.1":               "claude-2-1",
    "claude-2.0":               "claude-2-0",
    "claude-instant-1.2":       "claude-instant-1-2",
    # Claude Code short-form IDs not directly registered in EcoLogits
    "claude-haiku-4-6":         "claude-haiku-4-5-20251001",
}

# Map ChatGPT model slugs → ecologits-compatible model names
# Includes both historical and current OpenAI models
# Note: Some newer models may not be in ecologits yet; fallbacks to latest available
CHATGPT_MODEL_MAP = {
    # GPT-5 (if available, otherwise falls back)
    "gpt-5":                                    "gpt-5",  # Fallback to latest
    
    # GPT-4o (current, all variants supported)
    "gpt-4o":                                   "gpt-4o",
    "gpt-4o-2024-05-13":                        "gpt-4o",
    "gpt-4o-2024-08-06":                        "gpt-4o",
    "gpt-4o-2024-11-20":                        "gpt-4o",
    
    # GPT-4 Turbo
    "gpt-4-turbo":                              "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09":                   "gpt-4-turbo",
    "gpt-4-turbo-preview":                      "gpt-4-turbo",
    
    # GPT-4
    "gpt-4":                                    "gpt-4",
    "gpt-4-32k":                                "gpt-4-32k",
    "gpt-4-0613":                               "gpt-4",
    "gpt-4-32k-0613":                           "gpt-4-32k",
    
    # GPT-3.5 Turbo (text-davinci-002-render-sha variants)
    # These are typically from older ChatGPT exports
    # Note: Use gpt-3.5-turbo (with hyphen, not underscore)
    "text-davinci-002-render-sha":              "gpt-3.5-turbo",
    "text-davinci-002-render-sha-mobile":       "gpt-3.5-turbo",
    "text-davinci-002-render-sha-code-interpreter": "gpt-3.5-turbo",
    "text-davinci-002-render-sha-plugin":       "gpt-3.5-turbo",
    
    # GPT-3.5 Turbo (with correct hyphen format)
    "gpt-3.5-turbo":                            "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613":                       "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-0125":                       "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106":                       "gpt-3.5-turbo-1106",
}

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"  # Fallback for Claude exports
DEFAULT_OPENAI_MODEL = "gpt-4o"                    # Fallback for ChatGPT exports
DEFAULT_LATENCY      = 30.0                        # seconds; used when not measurable

CLAUDE_PROVIDER = "anthropic"
OPENAI_PROVIDER = "openai"


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

def _mid(v):
    """Return the midpoint of a RangeValue, or the value itself."""
    if isinstance(v, RangeValue):
        return (v.min + v.max) / 2
    return v if v is not None else 0.0

def _lo(v):
    return v.min if isinstance(v, RangeValue) else (v or 0.0)

def _hi(v):
    return v.max if isinstance(v, RangeValue) else (v or 0.0)


@dataclass
class RequestImpact:
    conversation_id:   str
    conversation_name: str
    timestamp:         datetime
    model:             str
    provider:          str
    output_tokens:     int
    energy_kwh:        float   # midpoint
    energy_kwh_lo:     float
    energy_kwh_hi:     float
    gwp_kgco2:         float
    gwp_kgco2_lo:      float
    gwp_kgco2_hi:      float
    water_l:           float
    water_l_lo:        float
    water_l_hi:        float
    warnings_list:     list = field(default_factory=list)


@dataclass
class AggImpact:
    requests:      int   = 0
    output_tokens: int   = 0
    energy_kwh:    float = 0.0
    energy_lo:     float = 0.0
    energy_hi:     float = 0.0
    gwp_kgco2:     float = 0.0
    gwp_lo:        float = 0.0
    gwp_hi:        float = 0.0
    water_l:       float = 0.0
    water_lo:      float = 0.0
    water_hi:      float = 0.0

    def add(self, r: RequestImpact):
        self.requests      += 1
        self.output_tokens += r.output_tokens
        self.energy_kwh    += r.energy_kwh
        self.energy_lo     += r.energy_kwh_lo
        self.energy_hi     += r.energy_kwh_hi
        self.gwp_kgco2     += r.gwp_kgco2
        self.gwp_lo        += r.gwp_kgco2_lo
        self.gwp_hi        += r.gwp_kgco2_hi
        self.water_l       += r.water_l
        self.water_lo      += r.water_l_lo
        self.water_hi      += r.water_l_hi


# ════════════════════════════════════════════════════════════════════════════
# EXPORT FORMAT DETECTION
# ════════════════════════════════════════════════════════════════════════════

def detect_export_type(data: list) -> str:
    """
    Detect whether this is a Claude or ChatGPT export by examining the structure
    of the first conversation.
    
    Claude: has 'chat_messages' or 'uuid' at top level
    ChatGPT: has 'mapping' at top level and conversation_id
    """
    if not data:
        return "unknown"
    
    first = data[0]
    
    # ChatGPT specific keys
    if "mapping" in first and "conversation_id" in first:
        return "chatgpt"
    
    # Claude specific keys
    if "chat_messages" in first or "uuid" in first:
        return "claude"
    
    # Fallback: check for common patterns
    if "mapping" in first:
        return "chatgpt"
    if "chat_messages" in first:
        return "claude"
    
    return "unknown"


# ════════════════════════════════════════════════════════════════════════════
# CLAUDE CODE FORMAT DETECTION & PARSING
# ════════════════════════════════════════════════════════════════════════════

def is_claudecode_input(path: Path) -> bool:
    """Return True if the path looks like Claude Code data (JSONL file or directory)."""
    if path.is_dir():
        return True
    return path.suffix.lower() == ".jsonl"


def _collect_jsonl_paths(path: Path) -> list[Path]:
    """Return a sorted list of .jsonl files from a file or directory."""
    if path.is_file():
        return [path]
    # Directory: find all .jsonl files (non-recursive — each project dir is flat)
    files = sorted(path.glob("*.jsonl"))
    if not files:
        # Fall back to recursive search
        files = sorted(path.rglob("*.jsonl"))
    return files


def _session_name_from_entries(entries: list[dict]) -> str:
    """Derive a human-readable session name from JSONL entries."""
    # Try the first user message content
    for e in entries:
        if e.get("type") == "user":
            msg = e.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                text = content.strip().replace("\n", " ")
                return text[:70] + ("…" if len(text) > 70 else "")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip().replace("\n", " ")
                        if text:
                            return text[:70] + ("…" if len(text) > 70 else "")
    # Fall back to project directory name from cwd
    for e in entries:
        cwd = e.get("cwd", "")
        if cwd:
            return Path(cwd).name
    return "unnamed session"


def parse_claudecode_sessions(
    paths: list[Path],
    zone: Optional[str],
    override_model: Optional[str] = None,
    model_mix: Optional[list[tuple[str, float]]] = None,
) -> list[RequestImpact]:
    """
    Parse Claude Code JSONL session files and estimate environmental impacts.

    Each JSONL file is one session (conversation). Only assistant entries with a
    non-null stop_reason ('end_turn' or 'tool_use') are counted — this avoids
    double-counting streaming intermediates and thinking-block flushes.

    Token counts come directly from message.usage.output_tokens (exact API value),
    so tiktoken is not needed here.
    """
    results: list[RequestImpact] = []
    skipped = 0
    total_files = len(paths)

    console.print(f"[dim]Detected export format: claude-code ({total_files} session file(s))[/dim]\n"
                  if RICH else f"Detected export format: claude-code ({total_files} session file(s))\n")

    for path in paths:
        # Load all entries from this JSONL file
        try:
            with open(path, encoding="utf-8") as f:
                entries = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            console.print(f"[yellow]⚠ Could not read {path.name}: {e}[/yellow]"
                          if RICH else f"⚠ Could not read {path.name}: {e}")
            continue

        session_id = path.stem  # filename without .jsonl
        session_name = _session_name_from_entries(entries)

        # Only process complete assistant responses (not streaming intermediates)
        for entry in entries:
            if entry.get("type") != "assistant":
                continue

            msg = entry.get("message", {})
            stop_reason = msg.get("stop_reason")
            if stop_reason not in ("end_turn", "tool_use"):
                continue

            usage = msg.get("usage", {})
            output_tokens = usage.get("output_tokens", 0)
            if output_tokens == 0:
                skipped += 1
                continue

            # Timestamp
            ts_str = entry.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)

            # Model resolution
            raw_model = msg.get("model", "")
            if override_model:
                resolved_model = override_model
            elif raw_model and raw_model != "<synthetic>":
                resolved_model = CLAUDE_MODEL_MAP.get(raw_model, raw_model)
            else:
                resolved_model = DEFAULT_CLAUDE_MODEL

            # Decide which model(s) to call
            if model_mix and not override_model and (not raw_model or raw_model == "<synthetic>"):
                use_mix = model_mix
            else:
                use_mix = [(resolved_model, 1.0)]

            fallback_models = [
                ("claude-sonnet-4-20250514", 1.0),
                ("claude-sonnet-4-5-20251001", 1.0),
                ("claude-opus-4-5-20250514", 1.0),
                ("claude-haiku-4-5-20251001", 1.0),
            ]

            # EcoLogits call(s)
            agg_energy = agg_energy_lo = agg_energy_hi = 0.0
            agg_gwp    = agg_gwp_lo    = agg_gwp_hi    = 0.0
            agg_water  = agg_water_lo  = agg_water_hi  = 0.0
            warn_msgs: list[str] = []
            mix_label = use_mix[0][0] if len(use_mix) == 1 else (
                "+".join(f"{m}({w*100:.0f}%)" for m, w in use_mix))

            ok = False
            for m_name, m_weight in list(use_mix):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        kwargs = dict(
                            provider=CLAUDE_PROVIDER,
                            model_name=m_name,
                            output_token_count=output_tokens,
                            request_latency=DEFAULT_LATENCY,
                        )
                        if zone:
                            kwargs["electricity_mix_zone"] = zone
                        impacts = llm_impacts(**kwargs)

                        if impacts is None or impacts.energy is None:
                            found_fallback = False
                            for fb_name, fb_weight in fallback_models:
                                try:
                                    kwargs["model_name"] = fb_name
                                    impacts = llm_impacts(**kwargs)
                                    if impacts is not None:
                                        warn_msgs.append(f"[{m_name}] not in EcoLogits, using {fb_name}")
                                        m_name = fb_name
                                        found_fallback = True
                                        break
                                except Exception:
                                    pass
                            if not found_fallback:
                                warn_msgs.append(f"[{m_name}] no fallback found, skipping")
                                continue

                        ok = True
                    except Exception as e:
                        warn_msgs.append(f"[{m_name}] {e}")
                        continue

                warn_msgs += [str(w.message) for w in caught]
                if impacts is None or impacts.energy is None:
                    continue

                energy = impacts.energy.value
                gwp    = impacts.gwp.value
                water  = impacts.wcf.value

                agg_energy    += _mid(energy) * m_weight
                agg_energy_lo += _lo(energy)  * m_weight
                agg_energy_hi += _hi(energy)  * m_weight
                agg_gwp       += _mid(gwp)    * m_weight
                agg_gwp_lo    += _lo(gwp)     * m_weight
                agg_gwp_hi    += _hi(gwp)     * m_weight
                agg_water     += _mid(water)  * m_weight
                agg_water_lo  += _lo(water)   * m_weight
                agg_water_hi  += _hi(water)   * m_weight

            if not ok:
                skipped += 1
                for w in warn_msgs:
                    console.print(f"[yellow]⚠ {session_name}: {w}[/yellow]"
                                  if RICH else f"⚠ {session_name}: {w}")
                continue

            results.append(RequestImpact(
                conversation_id=session_id,
                conversation_name=session_name,
                timestamp=ts,
                model=mix_label,
                provider=CLAUDE_PROVIDER,
                output_tokens=output_tokens,
                energy_kwh=agg_energy,
                energy_kwh_lo=agg_energy_lo,
                energy_kwh_hi=agg_energy_hi,
                gwp_kgco2=agg_gwp,
                gwp_kgco2_lo=agg_gwp_lo,
                gwp_kgco2_hi=agg_gwp_hi,
                water_l=agg_water,
                water_l_lo=agg_water_lo,
                water_l_hi=agg_water_hi,
                warnings_list=warn_msgs,
            ))

    if skipped > 0:
        console.print(f"[yellow]Skipped {skipped} entries with no output or unresolvable model.[/yellow]\n"
                      if RICH else f"Skipped {skipped} entries with no output or unresolvable model.\n")

    return results


# ════════════════════════════════════════════════════════════════════════════
# MODEL DETECTION & MAPPING
# ════════════════════════════════════════════════════════════════════════════

def detect_model_and_provider(
    conversation: dict,
    export_type: str,
    override_model: Optional[str] = None
) -> tuple[str, str]:
    """
    Detect both the model name and provider from a conversation.
    Returns (model_name, provider) tuple.
    
    For Claude exports: searches for model in top-level or message metadata
    For ChatGPT exports: searches mapping for model_slug in assistant message metadata
    """
    if override_model:
        # Try to infer provider from model name if not explicitly provided
        if any(k in override_model for k in ["gpt-", "text-davinci"]):
            return override_model, OPENAI_PROVIDER
        else:
            return override_model, CLAUDE_PROVIDER
    
    if export_type == "chatgpt":
        model, provider = _detect_chatgpt_model(conversation)
    elif export_type == "claude":
        model, provider = _detect_claude_model(conversation)
    else:
        # Unknown type - default to Claude (more conservative)
        model, provider = DEFAULT_CLAUDE_MODEL, CLAUDE_PROVIDER
    
    return model, provider


def _detect_claude_model(conversation: dict) -> tuple[str, str]:
    """Detect model from Claude export structure."""
    
    # Check top-level model key
    if "model" in conversation:
        slug = conversation["model"]
        return CLAUDE_MODEL_MAP.get(slug, slug), CLAUDE_PROVIDER
    
    # Scan assistant messages for metadata
    messages = conversation.get("chat_messages", [])
    for msg in messages:
        if msg.get("sender") == "assistant":
            meta = msg.get("metadata", {}) or {}
            if "model" in meta:
                slug = meta["model"]
                return CLAUDE_MODEL_MAP.get(slug, slug), CLAUDE_PROVIDER
    
    # Fallback
    return DEFAULT_CLAUDE_MODEL, CLAUDE_PROVIDER


def _detect_chatgpt_model(conversation: dict) -> tuple[str, str]:
    """Detect model from ChatGPT export structure (mapping-based)."""
    
    mapping = conversation.get("mapping", {})
    
    # Walk the message tree looking for assistant messages with model_slug
    def traverse_for_model(node_id: str) -> Optional[str]:
        if node_id not in mapping:
            return None
        
        node = mapping[node_id]
        if not node:
            return None
        
        msg = node.get("message")
        if msg:
            author = msg.get("author", {})
            
            # Look for assistant messages
            if author.get("role") == "assistant":
                meta = msg.get("metadata", {}) or {}
                
                # model_slug contains the model information
                if "model_slug" in meta:
                    return meta["model_slug"]
        
        # Traverse children
        for child_id in node.get("children", []):
            result = traverse_for_model(child_id)
            if result:
                return result
        
        return None
    
    # Start from root
    slug = None
    for node_id in mapping:
        node = mapping[node_id]
        if node and node.get("parent") is None:
            slug = traverse_for_model(node_id)
            break
    
    if slug:
        # Map ChatGPT slug to ecologits model name
        model_name = CHATGPT_MODEL_MAP.get(slug, slug)
        return model_name, OPENAI_PROVIDER
    
    # Fallback
    return DEFAULT_OPENAI_MODEL, OPENAI_PROVIDER


# ════════════════════════════════════════════════════════════════════════════
# PARSING
# ════════════════════════════════════════════════════════════════════════════

def extract_messages(conversation: dict, export_type: str) -> list[dict]:
    """
    Extract messages from either Claude or ChatGPT export format.
    Returns a normalized list of message dicts with keys:
      - sender: "user" or "assistant"
      - created_at: ISO timestamp
      - text: message content
      - model: detected model (for display/logging)
    """
    messages = []
    
    if export_type == "chatgpt":
        messages = _extract_chatgpt_messages(conversation)
    elif export_type == "claude":
        messages = _extract_claude_messages(conversation)
    
    return messages


def _extract_claude_messages(conversation: dict) -> list[dict]:
    """Extract messages from Claude export format."""
    messages = []
    conv_create = conversation.get("created_at")
    
    for msg in conversation.get("chat_messages", []):
        sender = msg.get("sender")
        if sender != "assistant":
            continue
        
        # Extract timestamp
        ts_str = msg.get("created_at") or conv_create
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)
        
        # Extract text content
        text = ""
        latency = DEFAULT_LATENCY
        
        content_blocks = msg.get("content", [])
        for block in content_blocks:
            btype = block.get("type")
            if btype == "text":
                text += block.get("text", "")
            elif btype == "tool_use":
                # Artifact/tool content lives in block["input"]
                inp = block.get("input", {})
                # For artifacts specifically, the code is in inp["content"]
                text += inp.get("content", "")
                # Also capture title, language, etc. if you want to be thorough
            # elif btype == "tool_result":
            #     for sub in block.get("content", []):
            #         if sub.get("type") == "text":
            #             text += sub.get("text", "")
            elif btype == "thinking":
                text += block.get("thinking", "")
                # Try to extract latency from timestamps
                try:
                    t0 = datetime.fromisoformat(
                        block["start_timestamp"].replace("Z", "+00:00"))
                    t1 = datetime.fromisoformat(
                        block["stop_timestamp"].replace("Z", "+00:00"))
                    dur = (t1 - t0).total_seconds()
                    if dur > 0:
                        latency = dur
                except Exception:
                    pass
        
        # Fallback text extraction
        if not text:
            text = msg.get("text", "")
        
        messages.append({
            "sender": sender,
            "created_at": ts,
            "text": text,
            "latency": latency,
        })
    
    return messages


def _extract_chatgpt_messages(conversation: dict) -> list[dict]:
    """Extract messages from ChatGPT export format (mapping tree)."""
    messages = []
    mapping = conversation.get("mapping", {})
    
    # Build a tree to traverse
    def traverse_node(node_id: str):
        if node_id not in mapping:
            return
        
        node = mapping[node_id]
        if not node:
            return
        
        msg = node.get("message")
        if msg:
            author = msg.get("author", {})
            
            if author.get("role") == "assistant":
                # Extract timestamp
                ts_float = msg.get("create_time")
                if ts_float:
                    ts = datetime.fromtimestamp(ts_float, tz=timezone.utc)
                else:
                    ts = datetime.now(timezone.utc)
                
                # Extract text from content.parts
                text = ""
                content = msg.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if isinstance(part, str):
                        text += part
                
                messages.append({
                    "sender": "assistant",
                    "created_at": ts,
                    "text": text,
                    "latency": DEFAULT_LATENCY,
                })
        
        # Traverse children (continue recursion even if no message)
        for child_id in node.get("children", []):
            traverse_node(child_id)
    
    # Start traversal from root (node with parent=None)
    for node_id in mapping:
        node = mapping[node_id]
        if node and node.get("parent") is None:
            traverse_node(node_id)
            break
    
    return messages


def parse_conversations(
    data: list,
    zone: Optional[str],
    override_model: Optional[str] = None,
    model_mix: Optional[list[tuple[str, float]]] = None,
) -> list[RequestImpact]:
    """
    Walk every conversation → every assistant message → estimate impacts.
    Each assistant reply is treated as one LLM request.

    If model_mix is provided, each request's impact is computed as the
    weighted average across all models in the mix (simulating a realistic
    multi-model usage pattern when the export lacks model metadata).
    """
    results: list[RequestImpact] = []
    skipped = 0
    
    # Detect export type once
    export_type = detect_export_type(data)
    console.print(f"[dim]Detected export format: {export_type}[/dim]\n" if RICH else f"Detected export format: {export_type}\n")

    for conv in data:
        conv_id = conv.get("uuid") or conv.get("conversation_id") or "unknown"
        conv_name = conv.get("name") or "Untitled"

        # Determine whether we detected a model from the data or must use override/mix
        detected_model, detected_provider = detect_model_and_provider(conv, export_type, override_model=None)
        # Check if model was explicitly in the conversation data (not just a default)
        model_in_data = any(k in conv for k in ["model", "mapping", "chat_messages"])
        resolved_model, resolved_provider = detect_model_and_provider(conv, export_type, override_model=override_model)

        messages = extract_messages(conv, export_type)

        for msg in messages:
            text = msg.get("text", "")
            ts = msg.get("created_at")
            latency_secs = msg.get("latency", DEFAULT_LATENCY)

            # ── token counting ─────────────────────────────────────────────
            output_tokens = count_tokens(text)

            if output_tokens == 0:
                skipped += 1
                continue

            # ── decide which model(s) to use ───────────────────────────────
            # Priority: data-detected model > --model override > --mix
            if model_in_data and not override_model:
                use_mix = [(resolved_model, 1.0)]
                use_provider = resolved_provider
            elif model_mix and not override_model:
                use_mix = model_mix
                use_provider = resolved_provider
            else:
                use_mix = [(resolved_model, 1.0)]
                use_provider = resolved_provider
            
            # Prepare fallback models in case primary fails
            fallback_models = []
            if use_provider == OPENAI_PROVIDER:
                fallback_models = [
                    ("gpt-4o", 1.0),
                    ("gpt-4-turbo", 1.0),
                    ("gpt-4", 1.0),
                    ("gpt-3.5-turbo", 1.0),
                ]
            else:  # Anthropic
                fallback_models = [
                    ("claude-sonnet-4-20250514", 1.0),
                    ("claude-opus-4-5-20250514", 1.0),
                    ("claude-haiku-4-5-20251001", 1.0),
                ]

            # ── ecologits call(s) ──────────────────────────────────────────
            # For a mix, compute weighted sum of all impact components.
            agg_energy = agg_energy_lo = agg_energy_hi = 0.0
            agg_gwp    = agg_gwp_lo    = agg_gwp_hi    = 0.0
            agg_water  = agg_water_lo  = agg_water_hi  = 0.0
            warn_msgs: list[str] = []
            mix_model_label = use_mix[0][0] if len(use_mix) == 1 else (
                "+".join(f"{m}({w*100:.0f}%)" for m, w in use_mix))

            ok = False
            models_to_try = list(use_mix)  # Try primary models first
            
            for m_name, m_weight in models_to_try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    try:
                        kwargs = dict(
                            provider=use_provider,
                            model_name=m_name,
                            output_token_count=output_tokens,
                            request_latency=latency_secs,
                        )
                        if zone:
                            kwargs["electricity_mix_zone"] = zone
                        impacts = llm_impacts(**kwargs)
                        
                        # Check if impacts is None (model not registered)
                        if impacts is None or impacts.energy is None:
                            # Try fallback models
                            found_fallback = False
                            for fallback_name, fallback_weight in fallback_models:
                                try:
                                    kwargs["model_name"] = fallback_name
                                    impacts = llm_impacts(**kwargs)
                                    if impacts is not None:
                                        warn_msgs.append(f"[{m_name}] not available, using {fallback_name}")
                                        m_name = fallback_name
                                        found_fallback = True
                                        break
                                except:
                                    pass
                            
                            if not found_fallback:
                                warn_msgs.append(f"[{m_name}] Model not registered, no fallback found")
                                continue
                        
                        ok = True
                    except Exception as e:
                        warn_msgs.append(f"[{m_name}] {e}")
                        continue

                warn_msgs += [str(w.message) for w in caught]
                
                # Only process if we successfully got impacts
                if impacts is None or impacts.energy is None:
                    continue
                    
                energy = impacts.energy.value
                gwp    = impacts.gwp.value
                water  = impacts.wcf.value

                agg_energy    += _mid(energy) * m_weight
                agg_energy_lo += _lo(energy)  * m_weight
                agg_energy_hi += _hi(energy)  * m_weight
                agg_gwp       += _mid(gwp)    * m_weight
                agg_gwp_lo    += _lo(gwp)     * m_weight
                agg_gwp_hi    += _hi(gwp)     * m_weight
                agg_water     += _mid(water)  * m_weight
                agg_water_lo  += _lo(water)   * m_weight
                agg_water_hi  += _hi(water)   * m_weight

            if not ok:
                skipped += 1
                for w in warn_msgs:
                    console.print(f"[yellow]⚠ {conv_name} (msg): {w}[/yellow]"
                                if RICH else f"⚠ {conv_name} (msg): {w}")
                continue

            results.append(RequestImpact(
                conversation_id=conv_id,
                conversation_name=conv_name,
                timestamp=ts,
                model=mix_model_label,
                provider=use_provider,
                output_tokens=output_tokens,
                energy_kwh=agg_energy,
                energy_kwh_lo=agg_energy_lo,
                energy_kwh_hi=agg_energy_hi,
                gwp_kgco2=agg_gwp,
                gwp_kgco2_lo=agg_gwp_lo,
                gwp_kgco2_hi=agg_gwp_hi,
                water_l=agg_water,
                water_l_lo=agg_water_lo,
                water_l_hi=agg_water_hi,
                warnings_list=warn_msgs,
            ))

    if skipped > 0:
        console.print(f"[yellow]Skipped {skipped} messages with no output.[/yellow]\n"
                    if RICH else f"Skipped {skipped} messages with no output.\n")

    return results


# ════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ════════════════════════════════════════════════════════════════════════════

def aggregate(results: list[RequestImpact]) -> tuple[AggImpact, dict, dict, dict, dict]:
    """
    Aggregate RequestImpacts into cumulative, by-conversation, by-week, by-month,
    and by-model summaries.
    """
    cumulative = AggImpact()
    by_conv    = {}
    by_week    = {}
    by_month   = {}
    by_model   = {}

    for r in results:
        cumulative.add(r)
        
        # By conversation
        conv_label = f"{r.conversation_id}|{r.conversation_name}"
        if conv_label not in by_conv:
            by_conv[conv_label] = AggImpact()
        by_conv[conv_label].add(r)
        
        # By week (ISO week)
        week_label = r.timestamp.strftime("%Y-W%V")
        if week_label not in by_week:
            by_week[week_label] = AggImpact()
        by_week[week_label].add(r)
        
        # By month
        month_label = r.timestamp.strftime("%Y-%m")
        if month_label not in by_month:
            by_month[month_label] = AggImpact()
        by_month[month_label].add(r)
        
        # By model
        if r.model not in by_model:
            by_model[r.model] = AggImpact()
        by_model[r.model].add(r)

    return cumulative, by_conv, by_week, by_month, by_model


# ════════════════════════════════════════════════════════════════════════════
# REAL-WORLD EQUIVALENTS  (metric + optional US customary)
# ════════════════════════════════════════════════════════════════════════════

def equivalents(agg: AggImpact, use_miles: bool = False, use_us_volume: bool = False) -> list[str]:
    """
    Generate real-world equivalents for energy and emissions.
    Uses midpoint estimates.
    use_miles:     report distances in miles instead of km
    use_us_volume: report water in fl oz / gallons instead of mL / L
    """
    items = []

    # ── Energy ──────────────────────────────────────────────────────────────
    wh = agg.energy_kwh * 1000
    if wh >= 1.0:
        energy_str = fmt_energy_str(agg.energy_kwh, decimals=1)
        items.append(f"{energy_str} — charging a phone ~{wh/5:.1f}×")

    # ── GHG ─────────────────────────────────────────────────────────────────
    gco2 = agg.gwp_kgco2 * 1000
    if gco2 >= 0.1:
        ghg_str = fmt_ghg_str(agg.gwp_kgco2, decimals=1)
        if use_miles:
            dist = gco2 / 322           # ~322 gCO₂eq/mile (avg US car ≈ 200 gCO₂/km × 1.609)
            dist_label = f"{dist:.1f} mi"
        else:
            dist = gco2 / 200           # ~200 gCO₂eq/km
            dist_label = f"{dist:.1f} km"
        tree_months = gco2 / 20  # ~20 gCO₂eq per month per mature tree
        items.append(f"{ghg_str} — driving {dist_label} or {tree_months:.1f} tree-months offset")

    # ── Water ────────────────────────────────────────────────────────────────
    ml = agg.water_l * 1000
    if ml >= 1.0:
        if use_us_volume:
            fl_oz = ml / 29.5735
            if fl_oz >= 128:           # ≥ 1 gallon
                gal = fl_oz / 128
                vol_str = f"{gal:.1f} gal"
                equiv   = f"{fl_oz/128:.1f}× a gallon jug"
            elif fl_oz >= 8:
                vol_str = f"{fl_oz:.1f} fl oz"
                equiv   = f"{fl_oz/8:.1f}× a cup of water"
            else:
                vol_str = f"{fl_oz:.2f} fl oz"
                equiv   = f"{fl_oz/8:.2f}× a cup of water"
        else:
            vol_str = fmt_water_str(agg.water_l, decimals=1)
            equiv   = f"{ml/250:.1f}× a glass of water"
        items.append(f"{vol_str} — {equiv}")

    return items or ["(negligible impact)"]


# ════════════════════════════════════════════════════════════════════════════
# PRINTING
# ════════════════════════════════════════════════════════════════════════════

def _pick_decimal(v: float) -> int:
    """Return sensible decimal places for a display value."""
    if v == 0:      return 1
    if v >= 100:    return 0
    if v >= 10:     return 1
    if v >= 1:      return 2
    return 3


def print_agg_table(title: str, aggs: dict, label_header: str = "Label", top_n: Optional[int] = None):
    """Print an aggregated impact table with auto-scaled units chosen per column."""
    sorted_items = sorted(aggs.items(), key=lambda x: -x[1].gwp_kgco2)
    if top_n:
        sorted_items = sorted_items[:top_n]

    if not sorted_items:
        console.print(f"[yellow]{title}: (no data)[/yellow]" if RICH else f"{title}: (no data)")
        return

    # Determine best unit for each column based on all values in this table
    energy_vals = [a.energy_kwh for _, a in sorted_items]
    ghg_vals    = [a.gwp_kgco2  for _, a in sorted_items]
    water_vals  = [a.water_l    for _, a in sorted_items]

    e_hdr, _, e_scale = _col_unit_label(energy_vals, "energy")
    g_hdr, _, g_scale = _col_unit_label(ghg_vals,    "ghg")
    w_hdr, _, w_scale = _col_unit_label(water_vals,   "water")

    if RICH:
        t = Table(box=box.SIMPLE, padding=(0, 2))
        t.add_column(label_header, style="bold cyan")
        t.add_column("Requests",     justify="right")
        t.add_column("Output Tokens", justify="right")
        t.add_column(e_hdr,          justify="right")
        t.add_column(g_hdr,          justify="right")
        t.add_column(w_hdr,          justify="right")
        for label, agg in sorted_items:
            disp_label = label.split("|", 1)[-1] if "|" in label else label
            ev = agg.energy_kwh * e_scale
            gv = agg.gwp_kgco2  * g_scale
            wv = agg.water_l    * w_scale
            t.add_row(
                disp_label,
                f"{agg.requests}",
                f"{agg.output_tokens:,}",
                f"{ev:.{_pick_decimal(ev)}f}",
                f"{gv:.{_pick_decimal(gv)}f}",
                f"{wv:.{_pick_decimal(wv)}f}",
            )
        console.print(t)
    else:
        print(f"\n{title}:")
        header = (f"  {'Label':<40} | {'Req':>5} | {'Tokens':>10} | "
                  f"{e_hdr:>14} | {g_hdr:>16} | {w_hdr:>12}")
        print(header)
        print("  " + "─" * (len(header) - 2))
        for label, agg in sorted_items:
            disp_label = label.split("|", 1)[-1] if "|" in label else label
            ev = agg.energy_kwh * e_scale
            gv = agg.gwp_kgco2  * g_scale
            wv = agg.water_l    * w_scale
            print(f"  {disp_label:<40} | {agg.requests:>5} | "
                  f"{agg.output_tokens:>10,} | "
                  f"{ev:>14.{_pick_decimal(ev)}f} | "
                  f"{gv:>16.{_pick_decimal(gv)}f} | "
                  f"{wv:>12.{_pick_decimal(wv)}f}")


def print_cumulative(agg: AggImpact, use_miles: bool = False, use_us_volume: bool = False):
    console.rule("CUMULATIVE TOTALS")
    rows = [
        ("Conversations / Requests",  f"{agg.requests}"),
        ("Output tokens generated",   f"{agg.output_tokens:,}"),
        ("Energy consumed",           fmt_energy_range_str(agg.energy_kwh, agg.energy_lo, agg.energy_hi)),
        ("GHG emissions",             fmt_ghg_range_str(agg.gwp_kgco2, agg.gwp_lo, agg.gwp_hi)),
        ("Water consumption",         fmt_water_range_str(agg.water_l, agg.water_lo, agg.water_hi)),
    ]
    if RICH:
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        t.add_column("Metric", style="bold green")
        t.add_column("Value")
        for k, v in rows:
            t.add_row(k, v)
        console.print(t)
        console.print("\n[bold]Real-world equivalents (midpoint estimates):[/bold]")
        for e in equivalents(agg, use_miles=use_miles, use_us_volume=use_us_volume):
            console.print(f"  • {e}")
    else:
        for k, v in rows:
            print(f"  {k:<35} {v}")
        print("\nReal-world equivalents (midpoint estimates):")
        for e in equivalents(agg, use_miles=use_miles, use_us_volume=use_us_volume):
            print(f"  • {e}")


# ════════════════════════════════════════════════════════════════════════════
# JSON EXPORT
# ════════════════════════════════════════════════════════════════════════════

def to_serializable(agg: AggImpact) -> dict:
    return {
        "requests":      agg.requests,
        "output_tokens": agg.output_tokens,
        "energy_kwh":    {"mid": agg.energy_kwh, "lo": agg.energy_lo, "hi": agg.energy_hi},
        "gwp_kgco2eq":   {"mid": agg.gwp_kgco2,  "lo": agg.gwp_lo,   "hi": agg.gwp_hi},
        "water_l":       {"mid": agg.water_l,     "lo": agg.water_lo, "hi": agg.water_hi},
    }


def build_json_report(results, cumulative, by_conv, by_week, by_month, by_model):
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cumulative": to_serializable(cumulative),
        "by_model": {k: to_serializable(v) for k, v in sorted(by_model.items())},
        "by_month": {k: to_serializable(v) for k, v in sorted(by_month.items())},
        "by_week":  {k: to_serializable(v) for k, v in sorted(by_week.items())},
        "by_conversation": {
            label.split("|", 1)[-1]: to_serializable(agg)
            for label, agg in sorted(by_conv.items(), key=lambda x: -x[1].gwp_kgco2)
        },
        "per_request": [
            {
                "conversation": r.conversation_name,
                "timestamp":    r.timestamp.isoformat(),
                "model":        r.model,
                "provider":     r.provider,
                "output_tokens": r.output_tokens,
                "energy_kwh":   {"mid": r.energy_kwh, "lo": r.energy_kwh_lo, "hi": r.energy_kwh_hi},
                "gwp_kgco2eq":  {"mid": r.gwp_kgco2,  "lo": r.gwp_kgco2_lo,  "hi": r.gwp_kgco2_hi},
                "water_l":      {"mid": r.water_l,     "lo": r.water_l_lo,     "hi": r.water_l_hi},
            }
            for r in sorted(results, key=lambda x: x.timestamp)
        ],
    }


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="claude_chatgpt_eco_analysis.py",
        description="Measure the environmental impact of your Claude.ai & ChatGPT conversations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START
───────────────────────────────────────────────────────────────────────────────
  python claude_chatgpt_eco_analysis.py conversations.json
      Analyzes your conversation export. Auto-detects Claude vs ChatGPT.

  python claude_chatgpt_eco_analysis.py conversations.json --output report.json
      Save detailed results to JSON for further analysis.


WHAT THIS TOOL DOES
───────────────────────────────────────────────────────────────────────────────
  • Parses Claude.ai or ChatGPT conversation exports (JSON)
  • Also analyzes Claude Code session files (.jsonl) or project directories
  • Estimates energy (kWh), CO₂ emissions (kg), and water usage (L)
  • Shows impact by model, time period, and conversation
  • Provides real-world equivalents (km driven, meals, showers, etc.)
  • Uses EcoLogits for peer-reviewed environmental calculations


HOW TO EXPORT YOUR CONVERSATIONS
───────────────────────────────────────────────────────────────────────────────
  Claude.ai:
    1. Visit https://claude.ai
    2. Profile → Data export
    3. Download conversations.json

  ChatGPT:
    1. Visit https://openai.com/account/data-export
    2. Request your data export
    3. Check email for download link (usually within hours)
    4. Extract conversations.json

  Claude Code:
    Sessions are stored automatically at:
      ~/.claude/projects/<project-name>/*.jsonl
    Pass a single .jsonl file (one session) or an entire project directory:
      python claude_chatgpt_eco_analysis.py ~/.claude/projects/my-project/
      python claude_chatgpt_eco_analysis.py ~/.claude/projects/my-project/abc123.jsonl
    Token counts come directly from the API usage data — no estimation needed.


MODEL SELECTION (if export lacks model metadata)
───────────────────────────────────────────────────────────────────────────────
  Use a single fallback model:
    --model claude-sonnet-4-20250514

  Or use a weighted model mix (weights auto-normalized):
    --mix "claude-sonnet-4:70,claude-haiku-4-5:30"
    --mix "gpt-4o:60,gpt-3.5-turbo:40"

  NOTE: --model and --mix are mutually exclusive.


REGIONAL ELECTRICITY MIX (for accuracy)
───────────────────────────────────────────────────────────────────────────────
  Defaults to world-average grid. For your region, use ISO 3166-1 alpha-3:
    --zone FRA    # France (cleaner grid → lower emissions)
    --zone DEU    # Germany
    --zone USA    # United States
    --zone IND    # India (coal-heavy → higher emissions)
    --zone GBR    # United Kingdom
    etc.


EXAMPLES
───────────────────────────────────────────────────────────────────────────────
  1. Basic analysis with auto-detected model:
     $ python claude_chatgpt_eco_analysis.py conversations.json

  2. Specify a fallback model:
     $ python claude_chatgpt_eco_analysis.py conversations.json \\
         --model claude-sonnet-4-20250514

  3. Use a model mix (e.g., used different models at different times):
     $ python claude_chatgpt_eco_analysis.py conversations.json \\
         --mix "claude-sonnet-4-20250514:75,claude-haiku-4-5:25"

  4. Save detailed JSON report:
     $ python claude_chatgpt_eco_analysis.py conversations.json \\
         --output my_report.json

  5. Adjust for France's cleaner grid:
     $ python claude_chatgpt_eco_analysis.py conversations.json --zone FRA

  6. Top 50 conversations instead of default 20:
     $ python claude_chatgpt_eco_analysis.py conversations.json --top 50

  7. Combine options:
     $ python claude_chatgpt_eco_analysis.py conversations.json \\
         --mix "gpt-4o:50,gpt-3.5-turbo:50" \\
         --output report.json \\
         --zone DEU \\
         --top 30


AVAILABLE CLAUDE MODELS
───────────────────────────────────────────────────────────────────────────────
  Latest:
    claude-opus-4-5-20250514
    claude-sonnet-4-5-20251001
    claude-haiku-4-5-20251001

  Previous versions:
    claude-opus-4-1-20250805
    claude-sonnet-4-20250514
    claude-sonnet-3-7-20250219
    claude-sonnet-3-5-20241022
    claude-haiku-3-5-20241022
    And earlier (3.0 and 2.x versions)


AVAILABLE OPENAI MODELS
───────────────────────────────────────────────────────────────────────────────
  Latest:
    gpt-5

  Current (GPT-4o):
    gpt-4o
    gpt-4o-2024-05-13
    gpt-4o-2024-08-06
    gpt-4o-2024-11-20

  Previous:
    gpt-4-turbo, gpt-4-turbo-2024-04-09
    gpt-4, gpt-4-32k
    gpt-3.5-turbo, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106


NOTES & CAVEATS
───────────────────────────────────────────────────────────────────────────────
  • Ranges (min–max) reflect inherent uncertainty in model efficiency.
  • Token counts use cl100k_base (OpenAI's standard encoding).
  • Input tokens: EcoLogits currently doesn't include these in energy estimates.
  • Latency: Estimated from message timestamps (server-side may differ).
  • Electricity mix: Defaults to world average; use --zone for regional accuracy.
  • Privacy: All computation happens locally on your machine. No data is uploaded.
  • Accuracy: Ideal for awareness & comparison, not forensic auditing.


DEPENDENCIES
───────────────────────────────────────────────────────────────────────────────
  Required:
    pip install ecologits

  Recommended (for accuracy):
    pip install tiktoken

  Optional (for pretty output):
    pip install rich

  Install all:
    pip install ecologits tiktoken rich


LEARN MORE
───────────────────────────────────────────────────────────────────────────────
  EcoLogits methodology: https://ecologits.ai/
  Anthropic sustainability: https://anthropic.com/research/ai-and-environmental-sustainability
        """)
    parser.add_argument("input",
        help="Path to conversations.json (Claude.ai or ChatGPT), a Claude Code .jsonl session "
             "file, or a Claude Code project directory containing .jsonl files")
    parser.add_argument("--output", "-o", help="Save full results to this JSON file")
    parser.add_argument("--zone", "-z",
        help="ISO 3166-1 alpha-3 electricity mix zone (e.g. FRA, DEU, USA). "
             "Defaults to EcoLogits provider default (world average).")
    parser.add_argument("--top", "-n", type=int, default=20,
        help="Show top N conversations (default 20)")
    parser.add_argument("--model", "-m",
        help="Override model for all requests (ecologits model name). "
             "Cannot be combined with --mix.")
    parser.add_argument("--mix",
        help='Weighted model mix, e.g. "gpt-4o-2024-05-13:75,gpt-3-5-turbo:25". '
             "Cannot be combined with --model.")

    # ── unit flags ────────────────────────────────────────────────────────────
    unit_group = parser.add_argument_group(
        "units (affect real-world equivalents only; scientific values always use auto-scaled SI)")
    unit_group.add_argument("--miles",     action="store_true",
        help="Use miles instead of kilometres for driving distance equivalents")
    unit_group.add_argument("--us-volume", action="store_true", dest="us_volume",
        help="Use fl oz / gallons instead of mL / L for water equivalents")
    unit_group.add_argument("--us",        action="store_true",
        help="Shorthand for --miles and --us-volume combined")

    args = parser.parse_args()

    if args.model and args.mix:
        parser.error("--model and --mix are mutually exclusive.")

    use_miles     = args.miles or args.us
    use_us_volume = args.us_volume or args.us

    model_mix = None
    if args.mix:
        try:
            model_mix = parse_model_mix(args.mix)
        except Exception as e:
            parser.error(f"Could not parse --mix value: {e}")
        label_width = max(len(m) for m, _ in model_mix)
        console.print("\n  Model mix:")
        for m, w in model_mix:
            console.print(f"    {m:<{label_width}}  {w*100:.1f}%")
        console.print()

    override_model = args.model or None

    # ── load ─────────────────────────────────────────────────────────────────
    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    console.rule("Claude.ai, ChatGPT & Claude Code Environmental Impact Analyzer")
    console.print(f"  Loading: [bold]{path}[/bold]\n" if RICH else f"  Loading: {path}\n")

    # ── parse & calculate ────────────────────────────────────────────────────
    if is_claudecode_input(path):
        jsonl_paths = _collect_jsonl_paths(path)
        if not jsonl_paths:
            print(f"ERROR: No .jsonl files found in: {path}", file=sys.stderr)
            sys.exit(1)
        console.print(f"  Found {len(jsonl_paths)} session file(s). Calculating impacts…\n")
        results = parse_claudecode_sessions(
            jsonl_paths,
            zone=args.zone,
            override_model=override_model,
            model_mix=model_mix,
        )
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("ERROR: Expected a JSON array at top level.", file=sys.stderr)
            sys.exit(1)

        console.print(f"  Found {len(data)} conversation(s). Calculating impacts…\n")

        results = parse_conversations(
            data,
            zone=args.zone,
            override_model=override_model,
            model_mix=model_mix,
        )

    if not results:
        console.print("[red]No results — no assistant messages found.[/red]"
                      if RICH else "No results — no assistant messages found.")
        sys.exit(0)

    cumulative, by_conv, by_week, by_month, by_model = aggregate(results)

    # ── print ────────────────────────────────────────────────────────────────
    print_cumulative(cumulative, use_miles=use_miles, use_us_volume=use_us_volume)

    console.rule("BY MODEL")
    print_agg_table("Model breakdown", by_model, label_header="Model")

    console.rule("BY MONTH")
    print_agg_table("Monthly breakdown", by_month, label_header="Month")

    console.rule("BY WEEK")
    print_agg_table("Weekly breakdown", by_week, label_header="ISO Week")

    console.rule("BY CONVERSATION (top {})".format(args.top))
    print_agg_table(
        f"Top {args.top} conversations by GHG impact",
        by_conv,
        label_header="Conversation",
        top_n=args.top,
    )

    # ── notes ─────────────────────────────────────────────────────────────────
    console.rule("NOTES")
    notes = [
        "Export type automatically detected (Claude vs ChatGPT).",
        "Model detection uses message metadata when available.",
        "Ranges reflect uncertainty in model architectures.",
        "Input tokens are NOT currently included in EcoLogits energy calculations.",
        "Latency is estimated from message timestamps; actual server latency may differ.",
        "Electricity mix zone defaults to world average; use --zone for regional accuracy.",
        "See https://ecologits.ai/methodology for full methodology details.",
    ]
    for n in notes:
        console.print(f"  {'[dim]' if RICH else ''}• {n}{'[/dim]' if RICH else ''}")

    # ── save JSON ─────────────────────────────────────────────────────────────
    if args.output:
        report = build_json_report(results, cumulative, by_conv, by_week, by_month, by_model)
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        console.print(f"\n  ✓ Full JSON report saved to: [bold]{out_path}[/bold]\n"
                      if RICH else f"\n  ✓ Full JSON report saved to: {out_path}\n")


if __name__ == "__main__":
    main()
