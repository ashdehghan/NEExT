# Reddit Dataset Integration with NEExT

This directory contains scripts and experiments for integrating the Reddit GraphSAGE dataset with the NEExT framework for node-level classification tasks.

## 📁 Directory Structure

```
reddit/
├── data/                     # Raw Reddit dataset files
│   ├── reddit.npz           # Node features and labels (602 features, 41 classes)
│   └── reddit_adj.npz       # Sparse adjacency matrix (232K nodes, 11.6M edges)
├── scripts/                  # Core data processing scripts
│   ├── load_reddit_to_networkx.py      # Convert Reddit data to NetworkX format
│   └── create_sampled_reddit.py        # Create stratified 5% sample
├── experiments/              # NEExT experiment scripts  
│   ├── reddit_neext_experiment.py      # Full node classification pipeline
│   └── reddit_quick_demo.py            # Quick demo with minimal parameters
├── tests/                    # Test and validation scripts
│   ├── test_neext_integration.py       # Integration testing
│   └── quick_test.py                    # Quick validation
├── reddit_networkx.pkl       # Full Reddit graph (3.3GB)
└── reddit_networkx_5pct.pkl  # 5% sampled graph (158MB)
```

## 🚀 Quick Start

### 1. Download Reddit Dataset
```bash
# Download reddit.npz and reddit_adj.npz to data/ folder
# Files available from GraphSAGE repository
```

### 2. Convert to NetworkX Format
```bash
cd scripts/
python load_reddit_to_networkx.py
# Creates: reddit_networkx.pkl (3.3GB)
```

### 3. Create Sampled Version (Optional)
```bash
python create_sampled_reddit.py
# Creates: reddit_networkx_5pct.pkl (158MB)
```

### 4. Run NEExT Experiments
```bash
cd ../experiments/
python reddit_quick_demo.py        # Quick demo (5% of nodes)
python reddit_neext_experiment.py  # Full experiment (configurable)
```

## 📊 Dataset Overview

- **Nodes**: 232,965 Reddit posts from September 2014
- **Edges**: 11,606,919 connections (posts with shared commenters)
- **Features**: 602-dimensional (GloVe embeddings + metadata)
- **Task**: Predict subreddit (41 classes) from local graph structure
- **Splits**: Train (152K) / Val (24K) / Test (55K)

## 🔬 NEExT Integration

The integration uses NEExT's **EgonetCollection** to transform node classification into a graph-level task:

1. **Graph Loading**: NetworkX graph with node attributes for features and labels
2. **Egonet Decomposition**: Create k-hop neighborhoods around each node
3. **Feature Computation**: Structural features on each egonet
4. **Graph Embeddings**: Wasserstein distance-based embeddings
5. **Classification**: XGBoost for subreddit prediction

## 🎯 Key Features

- **Stratified Sampling**: Preserves class distribution and train/val/test splits
- **Memory Efficient**: 5% sample reduces from 3.3GB to 158MB
- **NEExT Compatible**: Proper node attributes for features and labels
- **Flexible Configuration**: Adjustable k-hop, sampling rate, feature sets

## 📈 Results

With 5% sample (11,648 nodes):
- Successfully creates 582 egonets (5% sampling)
- Computes structural features in <1 second
- Generates embeddings in ~2 seconds
- 41-class classification requires larger samples for training

## 🔧 Configuration Options

### `reddit_neext_experiment.py`
```python
K_HOP = 2              # Neighborhood size (1, 2, or 3)
SAMPLE_FRACTION = 0.2  # Fraction of nodes to use (0.1 to 1.0)
EMBEDDING_DIM = 20     # Embedding dimensions
```

### `reddit_quick_demo.py`
```python
K_HOP = 1              # 1-hop for speed
SAMPLE_FRACTION = 0.05 # 5% for quick testing
EMBEDDING_DIM = 10     # Smaller embeddings
```

## 📝 Notes

- Large-scale experiments should use the full dataset or larger samples
- Class imbalance (41 classes) requires sufficient samples per class
- For best results, use at least 10-20% of nodes for training
- The pipeline demonstrates NEExT's capability for real-world node classification

## 🔗 References

- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)
- [NEExT Framework](https://github.com/ashdehghan/NEExT)
- Reddit dataset: Posts from September 2014, labeled by subreddit