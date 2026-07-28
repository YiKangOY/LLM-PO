# ECO Database

Simple storage system for ECO (Engineering Change Order) optimization iterations.

## Structure

### Core Classes

**`IterationData`** - Complete iteration data structure
- `iteration`: Iteration number
- `timestamp`: ISO timestamp
- `qor_report`: Quality of Results metrics
- `power_report`: Power consumption metrics
- `last_eco_option`: ECO type ("drc", "timing", "power", or None)
- `fixing_report`: ECO execution results (optional)
- `unfixable_reasons`: Issues that couldn't be resolved (optional)
- `metrics`: Additional custom metrics (optional)

**`ReportParser`** - Parses QoR and Power reports
- `parse_qor_report()`: Extracts timing, area, and DRC metrics
- `parse_power_report()`: Extracts power breakdown and totals
- `parse_drc_log()`: Extracts the DRC fixing logs from last iteration.
- `parse_drc_unfixing()`: Extracts the unfixable DRC violations from last iteration.
- `parse_timing_log()`: Extracts the Timing fixing logs from last iteration.
- `parse_timing_unfixing()`: Extracts the unfixable Timing violations from last iteration.
- `parse_power_log()`: Extracts the Power fixing logs from last iteration.

**`ECODatabase`** - JSON-based storage manager
- Stores iterations with extensible structure
- Handles baseline (iteration 0) correctly
- Provides optimization trend analysis

## Functions

### Database Operations
- `store_iteration()`: Save iteration data
- `get_iteration()`: Retrieve specific iteration
- `get_all_iterations()`: Get all stored iterations
- `parse_and_store_reports()`: Parse report files and store

### Analysis
- `get_eco_history()`: ECO command history (excluding baseline)
- `get_optimization_trend()`: Trends across iterations
- `export_summary()`: Database summary statistics

### Utilities
- `create_baseline_iteration()`: Generate baseline data
- `create_eco_iteration()`: Generate ECO iteration data
- `demo_database()`: Demonstration with real report parsing
- `test_report_parsing()`: Direct parser testing

## Storage Format

JSON structure with metadata and iterations:
```json
{
  "metadata": {
    "created": "timestamp",
    "version": "1.0"
  },
  "iterations": {
    "0": { /* baseline data */ },
    "1": { /* ECO iteration 1 */ }
  }
}
```
