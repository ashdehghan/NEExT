# Embedding Dimension Experiment

## Overview

This experiment tests the effect of Wasserstein embedding dimension size on classification performance using the Reddit 5% dataset.

## Experiment Design

1. **Fixed Components** (computed once):
   - 200 egonets sampled via random walk
   - Structural features (4 features: degree, clustering, PageRank, betweenness)
   - Reddit node features (602 features aggregated by mean)

2. **Variable Tested**: 
   - Embedding dimensions: 2, 5, 10, 15, 25, 50, max (full feature size)

3. **Models Compared**:
   - Combined (structural + node features)
   - Node features only
   - Structural features only
   - Random baseline (0.5 for binary classification)

## Files

- `experiment_embedding_dimensions.py` - Main experiment script
- `experiment_results/` - CSV results and plots
- `experiment_cache/` - Cached egonets and features for reuse

## Usage

```bash
python experiment_embedding_dimensions.py
```

## Expected Outputs

1. **CSV Results**: `experiment_results/embedding_dimensions_TIMESTAMP.csv`
2. **Scientific Plot**: `experiment_results/embedding_dimensions_plot_TIMESTAMP.html`
3. **Cached Data**: 
   - `experiment_cache/egonets_200.pkl`
   - `experiment_cache/features_200.pkl`

## Performance

- First run: ~10-15 minutes (computes egonets and features)
- Subsequent runs: ~5 minutes (uses cached data)

## Dependencies

- NEExT framework
- tqdm (progress bars)
- plotly (optional, for plots)
- Standard scientific Python stack (numpy, pandas, etc.)

## Expected Results

The experiment should show:
- Optimal embedding dimension around 10-25 for most cases
- Combined features outperforming individual feature types
- Diminishing returns at very high dimensions
- Node features contributing significantly to performance