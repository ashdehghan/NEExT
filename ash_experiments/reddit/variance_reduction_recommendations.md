# Variance Reduction Recommendations for Reddit Experiments

## Current Situation
- **Standard deviations**: 0.048-0.060 (5-6% of accuracy)
- **Root cause**: Small test sets (40 samples per fold)
- **Impact**: ±10% uncertainty makes it harder to compare methods

## Recommended Approaches (in order of effectiveness)

### 1. **Increase Number of Egonets** (Most Effective)
```python
# Current: 200 egonets → 40 per fold
# Recommended: 1000 egonets → 200 per fold
NUM_EGONETS = 1000  # 5x increase
```
**Benefits:**
- Reduces std by √5 ≈ 2.2x (from 0.05 to ~0.022)
- Allows testing up to 1000-dimensional embeddings
- Better statistical power for comparisons
**Cost:** 5x longer runtime (~3 hours)

### 2. **Use Stratified K-Fold with More Folds**
```python
# Instead of 5-fold CV, use 10-fold
sample_size = 10  # was 5
# Each fold: 180 train, 20 test (still small but more stable averaging)
```
**Benefits:**
- More stable average across folds
- Better use of available data
**Cost:** 2x longer runtime

### 3. **Repeated Cross-Validation**
```python
# Run 5-fold CV multiple times with different seeds
n_repeats = 3
for repeat in range(n_repeats):
    results = nxt.train_ml_model(
        sample_size=5,
        random_state=42 + repeat
    )
```
**Benefits:**
- 3x more measurements → std/√3 reduction
- Captures different fold partitions
**Cost:** 3x longer runtime

### 4. **Use Larger Reddit Sample** (Moderate Effect)
Current: 5% sample (7,315 nodes)
Options:
- **10% sample**: ~15,000 nodes → 2x more potential egonets
- **20% sample**: ~30,000 nodes → 4x more potential egonets

**Note:** This increases the pool but you still need to sample more egonets

### 5. **Bootstrap Confidence Intervals**
```python
# Instead of relying on CV std, use bootstrap
from sklearn.utils import resample
n_bootstrap = 100
bootstrap_scores = []
for i in range(n_bootstrap):
    X_boot, y_boot = resample(X_test, y_test)
    score = model.score(X_boot, y_boot)
    bootstrap_scores.append(score)
```
**Benefits:**
- More robust confidence intervals
- Better for small samples

## Recommended Experiment Configuration

For publication-quality results with low variance:

```python
# Configuration for low-variance experiments
NUM_EGONETS = 1000           # Increased from 200
SAMPLE_SIZE = 10             # 10-fold CV instead of 5
N_REPEATS = 3                # Repeated CV
DATASET = "reddit_20pct.pkl" # Larger pool

# Expected improvements:
# - Std reduction: ~3x (from 0.05 to ~0.015-0.020)
# - Test set per fold: 100 samples (vs 40)
# - Total measurements: 30 (vs 5)
# - Runtime: ~3-4 hours (vs 35 minutes)
```

## Quick Fix (If Time is Limited)

Minimum changes for meaningful improvement:
```python
NUM_EGONETS = 500  # 2.5x increase
SAMPLE_SIZE = 10   # 10-fold CV
# Expected: std ~0.03 (vs 0.05), runtime ~90 minutes
```

## Statistical Significance Testing

Add proper statistical tests:
```python
from scipy.stats import wilcoxon, friedmanchisquare

# Compare two models
stat, p_value = wilcoxon(model1_scores, model2_scores)
if p_value < 0.05:
    print("Significant difference detected")

# Compare multiple models
stat, p_value = friedmanchisquare(model1_scores, model2_scores, model3_scores)
```

## Conclusion

The high variance is primarily due to **small test sets** (40 samples per fold). The most effective solution is to **increase the number of egonets to 1000**, which will:
1. Reduce standard deviation by ~2-3x
2. Enable testing true high-dimensional embeddings (up to 1000)
3. Provide more reliable model comparisons

The tradeoff is increased computation time, but for publication-quality results, this investment is worthwhile.