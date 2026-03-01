# Claude & ChatGPT Environmental Impact Analyzer

Measure the environmental footprint of your AI conversations. This tool parses your `conversations.json` exports from Claude.ai or ChatGPT and estimates the energy consumption, CO₂ emissions (GHG), and water usage of every request using [EcoLogits](https://ecologits.ai/).

## Features

- **Automatic format detection**: Works with both Claude.ai and ChatGPT exports
- **Multi-model support**: Automatically detects model metadata from exports, or override with `--model` or `--mix`
- **Detailed breakdowns**: See environmental impact by model, month, week, and conversation
- **Uncertainty ranges**: All metrics include low/high estimates reflecting model architecture uncertainty
- **Real-world equivalents**: Compare your impact to tangible activities (e.g., km driven, meals, showers)
- **JSON export**: Save full results for further analysis
- **Regional electricity mix**: Adjust calculations for your region's energy grid (e.g., France, Germany, USA)
- **Pretty output**: Rich terminal formatting with color and tables (graceful fallback to plain text)

## Installation

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   pip install ecologits tiktoken rich
   ```
   - **ecologits**: Environmental impact calculations
   - **tiktoken**: Accurate token counting (optional but recommended for accuracy)
   - **rich**: Pretty terminal output (optional; plain text fallback if not installed)

## Usage

### Basic Usage

```bash
# Claude export with default model (Claude Sonnet 4)
python claude_chatgpt_eco_analysis.py conversations.json

# ChatGPT export with default model (GPT-4o)
python claude_chatgpt_eco_analysis.py conversations.json
```

### Override Model (Single Model)

If your export lacks model metadata, specify a fallback model:

```bash
python claude_chatgpt_eco_analysis.py conversations.json --model claude-sonnet-4-20250514
```

### Use a Weighted Model Mix

If you've used multiple models, specify a weighted mix:

```bash
python claude_chatgpt_eco_analysis.py conversations.json \
    --mix "claude-sonnet-4-20250514:75,claude-haiku-4-5-20251001:25"
```

Weights are automatically normalized, so `10:20:30` and `1:2:3` are equivalent.

### Save Full Results to JSON

```bash
python claude_chatgpt_eco_analysis.py conversations.json --output report.json
```

The JSON includes per-request details, aggregates by model/month/week/conversation, and timestamps.

### Adjust Electricity Mix (Regional)

By default, calculations use a world-average electricity mix. For accuracy, specify your region using ISO 3166-1 alpha-3 codes:

```bash
python claude_chatgpt_eco_analysis.py conversations.json --zone FRA  # France
python claude_chatgpt_eco_analysis.py conversations.json --zone DEU  # Germany
python claude_chatgpt_eco_analysis.py conversations.json --zone USA  # United States
```

### Show Top N Conversations

```bash
python claude_chatgpt_eco_analysis.py conversations.json --top 50
```

(Default is 20)

### View All Options

```bash
python claude_chatgpt_eco_analysis.py --help
```

## Example Output

```
━━━━━━━━━━━━━━━━━━━━━━ Claude.ai & ChatGPT Environmental Impact Analyzer ━━━━━━━━━━━━━━━━━━━━━━
  Loading: /home/user/conversations.json

  Found 127 conversation(s). Calculating impacts…

Metric                             Value
────────────────────────────────────────────────────────────────
Total requests                     387
Output tokens                      1,234,567
Energy consumption                 0.42 kWh  [0.18–0.89]
GHG emissions                      0.078 kg CO₂eq  [0.033–0.167]
Water consumption                  850 mL  [410–1,900]

Real-world equivalents (midpoint estimates):
  • 5.2 km driven in an average car
  • 0.98 meals (production only)
  • 3.4 showers (10 min, mixed flow)
```

## How It Works

1. **Export parsing**: Detects whether the JSON is from Claude.ai or ChatGPT
2. **Model detection**: Extracts model info from message metadata; falls back to `--model` or `--mix` if needed
3. **Token counting**: Uses `tiktoken` for accurate counts (word approximation if not installed)
4. **Impact calculation**: Uses EcoLogits' LLM environmental impact library to estimate:
   - Energy per token (based on model architecture, latency, inference efficiency)
   - GHG emissions (from energy × electricity mix)
   - Water consumption (from electricity generation)
5. **Aggregation**: Groups results by model, time period, and conversation
6. **Output**: Displays summary, breakdowns, and optionally saves full JSON report

## Model References

### Claude Models

- `claude-opus-4-5-20250514`
- `claude-opus-4-1-20250805`
- `claude-sonnet-4-5-20251001`
- `claude-sonnet-4-20250514`
- `claude-sonnet-3-7-20250219`
- `claude-sonnet-3-5-20241022`
- `claude-haiku-4-5-20251001`
- `claude-haiku-3-5-20241022`
- And others (see `--help` or source code for full list)

### OpenAI Models

- `gpt-5`
- `gpt-4o`, `gpt-4o-2024-05-13`, `gpt-4o-2024-08-06`, `gpt-4o-2024-11-20`
- `gpt-4-turbo`, `gpt-4-turbo-2024-04-09`
- `gpt-4`, `gpt-4-32k`
- `gpt-3.5-turbo`, `gpt-3.5-turbo-0125`, `gpt-3.5-turbo-1106`
- And others (see `--help` or source code for full list)

## How to Export Your Conversations

### Claude.ai

1. Visit [claude.ai](https://claude.ai)
2. Click your profile → **Data export**
3. Download the JSON file
4. Run the analyzer on it

### ChatGPT

1. Visit [openai.com/account/data-export](https://openai.com/account/data-export)
2. Request your data export
3. Wait for the email (usually within hours)
4. Download and extract `conversations.json`
5. Run the analyzer on it

## Notes on Accuracy

- **Ranges**: All metrics are reported as min–mid–max ranges. The midpoint is most commonly used; ranges reflect inherent uncertainty in model efficiency estimates from EcoLogits.
- **Input tokens**: EcoLogits currently does not include input tokens in energy calculations (only output tokens).
- **Latency**: Estimated from message timestamps; actual server-side latency may differ slightly.
- **Electricity mix**: Defaults to world-average grid emissions. Use `--zone` to adjust for your region's energy sources.
- **Model metadata**: Claude exports may lack explicit model info for older conversations. Use `--model` or `--mix` to override.

## Advanced Usage

### Combine with Filters (Bash)

Analyze only a subset of conversations:
```bash
# Create a filtered export with jq (first 100 conversations)
jq '.[0:100]' conversations.json > subset.json
python claude_chatgpt_eco_analysis.py subset.json
```

### Batch Processing

Analyze multiple exports:
```bash
for file in claude_*.json; do
    echo "=== Analyzing $file ==="
    python claude_chatgpt_eco_analysis.py "$file" --output "report_${file%.json}.json"
done
```

### Track Impact Over Time

Save reports regularly and compare:
```bash
python claude_chatgpt_eco_analysis.py conversations.json --output "report_$(date +%Y-%m-%d).json"
```

## Limitations & Caveats

- Token counts use `cl100k_base` (works for both Claude and GPT-3.5/4). Some models may have slight variations.
- Latency is estimated; ideal for research and awareness, not for precise energy auditing.
- EcoLogits methodology is based on published research; improvements and updates may change results slightly.
- Very new models may not yet be in EcoLogits; the tool will attempt fallback detection.

## Methodology & Sources

Environmental impact calculations are powered by **[EcoLogits](https://ecologits.ai/)**, which uses peer-reviewed research on:

- Model architecture and inference efficiency
- Energy consumption per inference
- Regional electricity grid emissions data
- Water usage in data centers and electricity generation

See the [EcoLogits methodology](https://ecologits.ai/methodology) for technical details.

## License

This script is provided as-is. See the source file for any license information.

## Contributing

Found a bug? Have a suggestion? Feel free to open an issue or submit a pull request.

## See Also

- [EcoLogits](https://ecologits.ai/) – Environmental impact library
- [Anthropic (Claude)](https://anthropic.com)
- [OpenAI (ChatGPT)](https://openai.com)
- [Sustainable AI research](https://www.anthropic.com/research/ai-and-environmental-sustainability)
