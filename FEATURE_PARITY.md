# Feature Parity

Tracks which features exist in the Python CLI (`claude_chatgpt_eco_analysis.py`) and the browser-based HTML analyzer (`llm-impact-analyzer.html`). Update this whenever adding a feature to either version.

| Feature | Python CLI | HTML | Notes |
|---|---|---|---|
| Export format detection (Claude / ChatGPT) | ✅ | ✅ | |
| Claude Code session file (.jsonl) support | ✅ | ✅ | |
| Model detection + fallback chain | ✅ | ✅ | |
| Token counting (tiktoken / word-count fallback) | ✅ | ✅ | HTML uses word-count only (no tiktoken in browser) |
| EcoLogits impact calculation | ✅ | ✅ | HTML bundles EcoLogits lookup tables inline |
| Cumulative totals | ✅ | ✅ | |
| By-model breakdown | ✅ | ✅ | |
| By-month breakdown | ✅ | ✅ | |
| By-week (ISO) breakdown | ✅ | ✅ | |
| By-day breakdown | ✅ (--by-day flag) | ✅ | Python: optional terminal table + always in JSON; HTML: accordion |
| By-conversation breakdown | ✅ | ✅ | |
| Accordion drill-down (month/week → days) | ❌ | ✅ | UI concept; not applicable to CLI |
| JSON output | ✅ (--output flag) | ✅ (Download JSON) | |
| Smart unit auto-scaling (mWh/Wh/kWh/MWh) | ✅ | ✅ | |
| Real-world equivalents display | ✅ | ✅ | |
| Electricity zone selection | ✅ (--zone flag) | ✅ (dropdown) | |
| --model / --mix model override | ✅ | ❌ | CLI-only; HTML always uses detected model |
| US customary units (miles, fl oz, gallons) | ✅ (--miles / --us-volume / --us) | ✅ (toggle) | |
| Pagination for conversations table | ❌ | ✅ | CLI uses --top N instead |
| Sortable columns | ❌ | ✅ | |

## When adding a feature

1. Implement it in both versions where applicable.
2. Update this table.
3. If a feature only makes sense in one version (e.g. interactive UI elements), mark the other column ❌ with a note.
