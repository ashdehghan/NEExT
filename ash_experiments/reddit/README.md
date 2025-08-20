# Reddit Binary Classification with NEExT

Cleaned and optimized experiments for Reddit node classification using NEExT framework with random walk sampling and Reddit features integration.

## 📁 Directory Structure (After Cleanup)

```
reddit/
├── data/                     # Raw Reddit dataset files
│   ├── reddit.npz           # Node features and labels (602 features, 41 classes)
│   └── reddit_adj.npz       # Sparse adjacency matrix (232K nodes, 11.6M edges)
├── scripts/                  # Core data processing scripts
│   ├── load_reddit_to_networkx.py      # Convert Reddit data to NetworkX format
│   └── create_sampled_reddit.py        # Create stratified samples
├── run_binary_experiment_rw_with_features.py  # MAIN EXPERIMENT SCRIPT
├── create_binary_reddit.py   # Create binary classification datasets
├── dataset.json             # Dataset documentation
├── reddit_binary_5pct.pkl   # Binary dataset - 7,315 nodes
├── reddit_binary_20pct.pkl  # Binary dataset - 29,258 nodes
├── reddit_networkx.pkl      # Full Reddit graph (3.3GB)
├── reddit_networkx_5pct.pkl # 5% sampled graph (158MB)
├── reddit_networkx_20pct.pkl # 20% sampled graph
└── CLEANUP_SUMMARY.md       # Cleanup documentation
```

## 🚀 Quick Start

### Run Main Experiment (Recommended)
```bash
python run_binary_experiment_rw_with_features.py
```
- Uses 100 egonets for fast testing (~10-20 seconds)
- Combines 602 Reddit features + 4 structural features
- Expected accuracy: 70-85% on binary classification

### Data Creation (If Needed)

#### Create Binary Dataset
```bash
python create_binary_reddit.py
# Creates binary classification datasets (serious vs entertainment)
```

#### Convert Raw Data to NetworkX
```bash
python scripts/load_reddit_to_networkx.py
# Creates: reddit_networkx.pkl (3.3GB)
```

#### Create Stratified Samples
```bash
python scripts/create_sampled_reddit.py
# Creates 5% and 20% stratified samples
```

## 📊 Dataset Overview

- **Nodes**: 232,965 Reddit posts from September 2014
- **Edges**: 11,606,919 connections (posts with shared commenters)
- **Features**: 602-dimensional (GloVe embeddings + metadata)
- **Task**: Predict subreddit (41 classes) from local graph structure
- **Splits**: Train (152K) / Val (24K) / Test (55K)

## 🔬 NEExT Integration

The integration uses NEExT's **EgonetCollection** with advanced sampling and feature combination:

1. **Graph Loading**: NetworkX graph preserving 602 Reddit features per node
2. **Random Walk Sampling**: Efficient bounded neighborhoods (vs exponential k-hop)
3. **Feature Extraction**: 
   - Original Reddit features (602 content/behavior features)
   - Structural features (PageRank, centrality, clustering)
4. **Feature Combination**: Using NEExT's `Features.__add__()` operator
5. **Graph Embeddings**: Wasserstein distance-based embeddings
6. **Classification**: XGBoost for binary classification

## 🎯 Key Features

- **Stratified Sampling**: Preserves class distribution and train/val/test splits
- **Memory Efficient**: 5% sample reduces from 3.3GB to 158MB
- **NEExT Compatible**: Proper node attributes for features and labels
- **Flexible Configuration**: Adjustable k-hop, sampling rate, feature sets

## 📈 Results

### Binary Classification Performance
- **Baseline (structural features only)**: ~43% accuracy (random)
- **With Reddit features**: ~70-85% accuracy 
- **Improvement**: +63% relative performance gain

### Efficiency Metrics (100 egonets)
- **Egonet generation**: 2 seconds with random walk sampling
- **Feature extraction**: <1 second for 606 features
- **Total runtime**: 10-20 seconds end-to-end
- **Memory efficient**: Max 30 nodes per egonet (bounded)

## 🔧 Configuration Options

### Main Experiment Configuration
```python
# In run_binary_experiment_rw_with_features.py
SAMPLE_FRACTION = 100 / 7315  # ~100 egonets for testing
WALK_LENGTH = 8                # Random walk length
NUM_WALKS = 4                  # Number of walks per node
MAX_NODES_PER_EGONET = 30     # Bounded neighborhood size
EMBEDDING_DIM = 20             # Embedding dimensions
```

### Binary Dataset Classes
- **Class 0 (Serious)**: news, science, tech, politics subreddits
- **Class 1 (Entertainment)**: funny, videos, gaming, art subreddits

## 📝 Notes

- Large-scale experiments should use the full dataset or larger samples
- Class imbalance (41 classes) requires sufficient samples per class
- For best results, use at least 10-20% of nodes for training
- The pipeline demonstrates NEExT's capability for real-world node classification

## 🔗 References

- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)
- [NEExT Framework](https://github.com/ashdehghan/NEExT)
- Reddit dataset: Posts from September 2014, labeled by subreddit