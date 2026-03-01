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
            if block.get("type") == "text":
                text += block.get("text", "")
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
                        if impacts is None:
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
                if impacts is None:
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
# REAL-WORLD EQUIVALENTS
# ════════════════════════════════════════════════════════════════════════════

def equivalents(agg: AggImpact) -> list[str]:
    """
    Generate real-world equivalents for energy and emissions.
    Uses midpoint estimates.
    """
    items = []

    # Energy: Wh
    wh = agg.energy_kwh * 1000
    if wh >= 1.0:
        items.append(f"{wh:.1f} Wh — charging a phone ~{wh/5:.1f}×")

    # GHG: gCO₂eq
    gco2 = agg.gwp_kgco2 * 1000
    if gco2 >= 0.1:
        km_driven = gco2 / 0.2  # ~200 gCO₂eq per km (avg car)
        tree_months = gco2 / 20  # ~20 gCO₂eq per month per mature tree
        items.append(f"{gco2:.1f} gCO₂eq — driving {km_driven:.1f} km or {tree_months:.1f} tree-months offset")

    # Water: mL
    ml = agg.water_l * 1000
    if ml >= 1.0:
        items.append(f"{ml:.0f} mL — {ml/250:.1f}× a glass of water")

    return items or ["(negligible impact)"]


# ════════════════════════════════════════════════════════════════════════════
# PRINTING
# ════════════════════════════════════════════════════════════════════════════

def print_agg_table(title: str, aggs: dict, label_header: str = "Label", top_n: Optional[int] = None):
    """Print an aggregated impact table (by model, month, etc.)."""
    sorted_items = sorted(aggs.items(), key=lambda x: -x[1].gwp_kgco2)
    if top_n:
        sorted_items = sorted_items[:top_n]

    if not sorted_items:
        console.print(f"[yellow]{title}: (no data)[/yellow]" if RICH else f"{title}: (no data)")
        return

    if RICH:
        t = Table(box=box.SIMPLE, padding=(0, 2))
        t.add_column(label_header, style="bold cyan")
        t.add_column("Requests",  justify="right")
        t.add_column("Output Tokens", justify="right")
        t.add_column("Energy (mWh)", justify="right")
        t.add_column("GHG (mgCO₂eq)", justify="right")
        t.add_column("Water (mL)", justify="right")
        for label, agg in sorted_items:
            disp_label = label.split("|", 1)[-1] if "|" in label else label
            t.add_row(
                disp_label,
                f"{agg.requests}",
                f"{agg.output_tokens:,}",
                f"{agg.energy_kwh*1e6:.1f}",
                f"{agg.gwp_kgco2*1e6:.0f}",
                f"{agg.water_l*1000:.1f}",
            )
        console.print(t)
    else:
        print(f"\n{title}:")
        for label, agg in sorted_items:
            disp_label = label.split("|", 1)[-1] if "|" in label else label
            print(f"  {disp_label:<40} | {agg.requests:>5} req | "
                  f"{agg.output_tokens:>10,} tkn | {agg.energy_kwh*1e6:>10.1f} mWh | "
                  f"{agg.gwp_kgco2*1e6:>12.0f} mgCO₂")


def print_cumulative(agg: AggImpact):
    console.rule("CUMULATIVE TOTALS")
    rows = [
        ("Conversations / Requests",  f"{agg.requests}"),
        ("Output tokens generated",   f"{agg.output_tokens:,}"),
        ("Energy consumed",           f"{agg.energy_kwh*1000:.3f} Wh  "
                                      f"[{agg.energy_lo*1000:.3f}–{agg.energy_hi*1000:.3f} Wh]"),
        ("GHG emissions",             f"{agg.gwp_kgco2*1000:.2f} gCO₂eq  "
                                      f"[{agg.gwp_lo*1000:.2f}–{agg.gwp_hi*1000:.2f}]"),
        ("Water consumption",         f"{agg.water_l*1000:.1f} mL  "
                                      f"[{agg.water_lo*1000:.1f}–{agg.water_hi*1000:.1f}]"),
    ]
    if RICH:
        t = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
        t.add_column("Metric", style="bold green")
        t.add_column("Value")
        for k, v in rows:
            t.add_row(k, v)
        console.print(t)
        console.print("\n[bold]Real-world equivalents (midpoint estimates):[/bold]")
        for e in equivalents(agg):
            console.print(f"  • {e}")
    else:
        for k, v in rows:
            print(f"  {k:<35} {v}")
        print("\nReal-world equivalents (midpoint estimates):")
        for e in equivalents(agg):
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
    parser.add_argument("input",  help="Path to conversations.json (Claude or ChatGPT)")
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
    args = parser.parse_args()

    if args.model and args.mix:
        parser.error("--model and --mix are mutually exclusive.")

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
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    console.rule("Claude.ai & ChatGPT Environmental Impact Analyzer")
    console.print(f"  Loading: [bold]{path}[/bold]\n" if RICH else f"  Loading: {path}\n")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("ERROR: Expected a JSON array at top level.", file=sys.stderr)
        sys.exit(1)

    console.print(f"  Found {len(data)} conversation(s). Calculating impacts…\n")

    # ── parse & calculate ────────────────────────────────────────────────────
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
    print_cumulative(cumulative)

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
