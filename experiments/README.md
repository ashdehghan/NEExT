# NEExT Experiments

This directory contains various experiments using the NEExT framework. Each experiment is implemented as a Python script and demonstrates different aspects of the framework's capabilities.

## Available Experiments

### 1. Node Sampling Experiment (`node_sampling_experiments.py`)

This experiment investigates the effect of node sampling rate on classifier accuracy when using graph embeddings.

**Purpose:**
- Understand how reducing the number of nodes in graphs affects model performance
- Find optimal sampling rates that balance computational efficiency and accuracy
- Demonstrate the node sampling functionality of NEExT

**Experiment Details:**
- Dataset: BZR (Benzene Ring dataset)
- Sampling rates: 1.0 to 0.1 (in decrements of 0.1)
- Features: All available node features
- Embedding: Approximate Wasserstein
- Model: Classifier with 50 train/test splits per sampling rate
- Visualization: Box plot of accuracy vs sampling rate

**Requirements:**
- NEExT framework with experiments dependencies:
  ```bash
  pip install -e ".[experiments]"
  ```

**Running the Experiment:**
1. Navigate to the experiments directory:
   ```bash
   cd examples/experiments
   ```
2. Run the experiment script:
   ```bash
   python node_sampling_experiments.py
   ```

**Output Files:**
- `node_sampling_results.png`: Box plot visualization showing accuracy distribution for each sampling rate
- `node_sampling_summary.csv`: Summary statistics with mean, standard deviation, min, and max accuracy for each rate
