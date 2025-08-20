# ash_experiments/reddit Cleanup Summary

## Files Being Kept

### Core Data Files (Essential)
- `reddit_binary_5pct.pkl` - Main dataset for experiments
- `reddit_binary_20pct.pkl` - Larger dataset option  
- `reddit_networkx.pkl` - Full Reddit graph
- `reddit_networkx_5pct.pkl` - 5% sample
- `reddit_networkx_20pct.pkl` - 20% sample
- `data/reddit.npz` - Raw Reddit features
- `data/reddit_adj.npz` - Raw adjacency matrix
- `dataset.json` - Dataset documentation

### Main Experiment Script (Current Working Version)
- `run_binary_experiment_rw_with_features.py` - **MAIN SCRIPT** with random walk + Reddit features

### Essential Data Creation Scripts
- `scripts/load_reddit_to_networkx.py` - Converts raw data to NetworkX format
- `scripts/create_sampled_reddit.py` - Creates stratified samples
- `create_binary_reddit.py` - Creates binary classification datasets

## Files Being Removed

### Superseded Experiment Scripts
- `run_binary_experiment.py` - Old k-hop version (superseded by random walk)
- `run_binary_experiment_rw.py` - Random walk without Reddit features (superseded)
- `run_20pct_experiment.py` - Old experiment script
- `run_full_neext_analysis.py` - Old full analysis script

### Temporary Analysis Scripts  
- `analyze_class_distribution.py` - One-time analysis
- `analyze_binary_grouping.py` - One-time analysis
- `check_class_distribution.py` - One-time check

### Test/Demo Scripts
- `experiments/reddit_neext_experiment.py` - Early experiment
- `experiments/reddit_quick_demo.py` - Demo script
- `tests/quick_test.py` - Test script
- `tests/test_neext_integration.py` - Integration test

### Old Logs
- `reddit_experiment_20250818_224946.log` - Old log
- `reddit_full_analysis_20250818_232718.log` - Old log

### Miscellaneous
- `scripts/create_20pct_sample.py` - Redundant (functionality in create_sampled_reddit.py)

## Directory Structure After Cleanup

```
ash_experiments/reddit/
├── data/
│   ├── reddit.npz          # Raw features
│   └── reddit_adj.npz      # Raw adjacency
├── scripts/
│   ├── load_reddit_to_networkx.py   # Data conversion
│   └── create_sampled_reddit.py     # Sampling utility
├── create_binary_reddit.py          # Binary dataset creator
├── run_binary_experiment_rw_with_features.py  # MAIN EXPERIMENT
├── dataset.json                     # Documentation
├── reddit_binary_5pct.pkl          # Main test dataset
├── reddit_binary_20pct.pkl         # Larger dataset
├── reddit_networkx.pkl             # Full graph
├── reddit_networkx_5pct.pkl        # 5% sample
├── reddit_networkx_20pct.pkl       # 20% sample
├── README.md                        # Usage instructions
└── CLEANUP_SUMMARY.md              # This file
```

## Usage After Cleanup

Main workflow:
```bash
# Run the main experiment with Reddit features
python run_binary_experiment_rw_with_features.py

# Create new binary datasets if needed
python create_binary_reddit.py

# Convert raw data to NetworkX format
python scripts/load_reddit_to_networkx.py
```