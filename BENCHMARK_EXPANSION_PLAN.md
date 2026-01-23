# Multi-Benchmark Expansion Plan

## Overview
Extend the ECI Accessibility Gap app to support multiple benchmarks from Epoch AI's database, allowing users to select which benchmark they want to see the gap analysis for.

## Environment Setup

For local development, set these environment variables:
```bash
export AIRTABLE_BASE_ID="appKfIQ7h4TtoGmQw"
export AIRTABLE_PERSONAL_ACCESS_TOKEN="your_token_here"
```

For GitHub Actions, add these as repository secrets:
- `AIRTABLE_BASE_ID`
- `AIRTABLE_PERSONAL_ACCESS_TOKEN`

## Implementation Steps

### Step 1: Install Epoch AI Package & Explore Available Benchmarks
- Install `epochai` package
- Write exploration script to list all available benchmarks/tasks
- Identify benchmarks with sufficient data for gap analysis
- **Commit after completing**

### Step 2: Create Benchmark Data Fetcher Module
- Create `scripts/benchmark_fetcher.py` - modular data fetching
- Abstract the data fetching to handle both ECI CSV and Airtable benchmarks
- Include model accessibility classification (open vs closed)
- **Commit after completing**

### Step 3: Extend Data Processing Pipeline
- Modify `scripts/update_data.py` to process multiple benchmarks
- Create benchmark-agnostic gap calculation functions
- Update JSON output structure to namespace by benchmark
- **Commit after completing**

### Step 4: Update Frontend - Benchmark Selector
- Add benchmark dropdown to `templates/index.html`
- Update `static/script.js` to load and switch between benchmarks
- Ensure ECI remains the default selection
- **Commit after completing**

### Step 5: Update Chart Rendering
- Parameterize chart labels/titles by benchmark name
- Update axis labels for different benchmark metrics
- Handle different score scales across benchmarks
- **Commit after completing**

### Step 6: Testing & Polish
- Add tests for new benchmark functionality
- Verify all existing ECI functionality still works
- Update GitHub Actions workflow if needed
- **Commit after completing**

## Data Structure Changes

### Current (single benchmark):
```json
{
  "models": [...],
  "gaps": [...],
  "statistics": {...},
  "trends": {...},
  "historical_gaps": [...],
  "china_framing": {...}
}
```

### New (multi-benchmark):
```json
{
  "benchmarks": {
    "eci": {
      "name": "Epoch Capabilities Index (ECI)",
      "unit": "ECI Score",
      "models": [...],
      "gaps": [...],
      "statistics": {...},
      "trends": {...},
      "historical_gaps": [...],
      "china_framing": {...}
    },
    "gpqa_diamond": {
      "name": "GPQA Diamond",
      "unit": "Accuracy %",
      ...
    }
  },
  "default_benchmark": "eci",
  "last_updated": "..."
}
```

## Key Considerations
1. Different benchmarks have different scales (ECI ~0-200, accuracy 0-100%, etc.)
2. Gap threshold may need to be benchmark-specific
3. Some benchmarks may have sparse data - need graceful handling
4. Maintain backward compatibility for existing ECI functionality
