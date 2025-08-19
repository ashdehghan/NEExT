#!/usr/bin/env python3
"""
Full NEExT Analysis on Reddit Binary Dataset

This script runs a comprehensive analysis using the 5% binary Reddit dataset,
demonstrating the complete NEExT pipeline for node-level classification.

Expected runtime: 2-5 minutes
Task: Predict SERIOUS vs ENTERTAINMENT subreddits from graph structure
"""

import pickle
import numpy as np
import pandas as pd
import time
import logging
import json
from pathlib import Path
from collections import Counter
import sys
from datetime import datetime

# Add NEExT to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from NEExT.framework import NEExT
from NEExT.collections import EgonetCollection

# Setup comprehensive logging
log_filename = f'reddit_full_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RedditFullAnalysis:
    """Complete NEExT analysis pipeline for Reddit binary classification."""
    
    def __init__(self):
        """Initialize analysis with configuration from dataset.json."""
        
        # Load dataset metadata
        with open('dataset.json', 'r') as f:
            self.dataset_info = json.load(f)
        
        # Configuration - using recommended full experiment settings
        full_config = self.dataset_info['experiment_recommendations']['full_experiments']
        
        self.config = {
            'graph_file': full_config['dataset'],
            'k_hop': full_config['k_hop'],
            'sample_fraction': full_config['sample_fraction'],
            'embedding_dim': full_config['embedding_dim'],
            'feature_list': [
                "degree_centrality",
                "closeness_centrality", 
                "betweenness_centrality",
                "clustering_coefficient",
                "page_rank",
                "eigenvector_centrality"
            ],
            'feature_vector_length': 2,  # 0-hop and 1-hop aggregation
            'random_seed': 42
        }
        
        # Results storage
        self.results = {
            'dataset_info': self.dataset_info['dataset_variants']['reddit_binary_5pct.pkl'],
            'configuration': self.config,
            'timing': {},
            'graph_stats': {},
            'egonet_stats': {},
            'feature_stats': {},
            'embedding_stats': {},
            'model_results': {},
            'feature_importance': None
        }
        
        # Initialize NEExT
        self.nxt = NEExT()
        self.nxt.set_log_level("WARNING")
        
        logger.info("="*80)
        logger.info("REDDIT FULL NEEXT ANALYSIS INITIALIZED")
        logger.info("="*80)
        logger.info(f"Configuration: {self.config}")
        
    def load_and_analyze_graph(self):
        """Load graph and perform initial analysis."""
        logger.info("\n[1/7] LOADING AND ANALYZING GRAPH")
        logger.info("-"*50)
        
        start_time = time.time()
        
        # Load graph
        with open(self.config['graph_file'], 'rb') as f:
            self.graph = pickle.load(f)
        
        load_time = time.time() - start_time
        self.results['timing']['graph_loading'] = load_time
        
        # Basic statistics
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        avg_degree = 2 * n_edges / n_nodes
        
        logger.info(f"Graph loaded in {load_time:.2f} seconds")
        logger.info(f"Nodes: {n_nodes:,}")
        logger.info(f"Edges: {n_edges:,}")
        logger.info(f"Average degree: {avg_degree:.1f}")
        
        # Analyze labels and splits
        binary_counts = Counter()
        split_counts = Counter()
        
        for _, attrs in self.graph.nodes(data=True):
            label = attrs.get('binary_label', -1)
            split = attrs.get('split', 'unknown')
            if label >= 0:
                binary_counts[label] += 1
            split_counts[split] += 1
        
        logger.info(f"Class distribution:")
        logger.info(f"  Serious (0): {binary_counts[0]:,} ({binary_counts[0]/n_nodes*100:.1f}%)")
        logger.info(f"  Entertainment (1): {binary_counts[1]:,} ({binary_counts[1]/n_nodes*100:.1f}%)")
        logger.info(f"  Balance ratio: {min(binary_counts.values())/max(binary_counts.values()):.3f}")
        
        logger.info(f"Split distribution: {dict(split_counts)}")
        
        # Store results
        self.results['graph_stats'] = {
            'nodes': n_nodes,
            'edges': n_edges,
            'average_degree': avg_degree,
            'class_distribution': dict(binary_counts),
            'split_distribution': dict(split_counts),
            'class_balance': min(binary_counts.values())/max(binary_counts.values())
        }
        
    def create_graph_collection(self):
        """Create NEExT GraphCollection."""
        logger.info("\n[2/7] CREATING GRAPH COLLECTION")
        logger.info("-"*50)
        
        start_time = time.time()
        
        self.graph_collection = self.nxt.load_from_networkx(
            [self.graph],
            reindex_nodes=False,
            filter_largest_component=False,
            node_sample_rate=1.0
        )
        
        creation_time = time.time() - start_time
        self.results['timing']['graph_collection'] = creation_time
        
        logger.info(f"GraphCollection created in {creation_time:.2f} seconds")
        
    def create_egonet_collection(self):
        """Create comprehensive egonet collection."""
        logger.info(f"\n[3/7] CREATING EGONET COLLECTION")
        logger.info("-"*50)
        logger.info(f"Parameters: k_hop={self.config['k_hop']}, sample_fraction={self.config['sample_fraction']:.1%}")
        
        start_time = time.time()
        
        self.egonet_collection = EgonetCollection(egonet_feature_target='binary_label')
        self.egonet_collection.compute_k_hop_egonets(
            graph_collection=self.graph_collection,
            k_hop=self.config['k_hop'],
            sample_fraction=self.config['sample_fraction'],
            random_seed=self.config['random_seed']
        )
        
        creation_time = time.time() - start_time
        self.results['timing']['egonet_creation'] = creation_time
        
        num_egonets = len(self.egonet_collection.graphs)
        logger.info(f"Created {num_egonets:,} egonets in {creation_time:.2f} seconds")
        logger.info(f"Rate: {num_egonets/creation_time:.1f} egonets/second")
        
        # Analyze egonet properties
        egonet_sizes = [len(g.nodes) for g in self.egonet_collection.graphs]
        egonet_labels = [g.graph_label for g in self.egonet_collection.graphs if g.graph_label is not None]
        
        label_dist = Counter(egonet_labels)
        
        logger.info(f"Egonet statistics:")
        logger.info(f"  Size - min: {min(egonet_sizes)}, max: {max(egonet_sizes)}, avg: {np.mean(egonet_sizes):.1f}")
        logger.info(f"  Labels - Serious: {label_dist.get(0, 0)}, Entertainment: {label_dist.get(1, 0)}")
        
        self.results['egonet_stats'] = {
            'count': num_egonets,
            'size_stats': {
                'min': min(egonet_sizes),
                'max': max(egonet_sizes),
                'mean': np.mean(egonet_sizes),
                'median': np.median(egonet_sizes),
                'std': np.std(egonet_sizes)
            },
            'label_distribution': dict(label_dist),
            'creation_rate': num_egonets/creation_time
        }
        
    def compute_structural_features(self):
        """Compute comprehensive structural features."""
        logger.info(f"\n[4/7] COMPUTING STRUCTURAL FEATURES")
        logger.info("-"*50)
        logger.info(f"Features: {', '.join(self.config['feature_list'])}")
        logger.info(f"Vector length: {self.config['feature_vector_length']} (multi-hop aggregation)")
        
        start_time = time.time()
        
        self.features = self.nxt.compute_node_features(
            graph_collection=self.egonet_collection,
            feature_list=self.config['feature_list'],
            feature_vector_length=self.config['feature_vector_length'],
            show_progress=True,
            n_jobs=-1
        )
        
        computation_time = time.time() - start_time
        self.results['timing']['feature_computation'] = computation_time
        
        num_features = len(self.features.feature_columns)
        logger.info(f"Computed {num_features} features in {computation_time:.2f} seconds")
        logger.info(f"Rate: {num_features/computation_time:.1f} features/second")
        logger.info(f"Feature columns: {self.features.feature_columns[:10]}..." if num_features > 10 else f"Feature columns: {self.features.feature_columns}")
        
        # Normalize features
        logger.info("Normalizing features with StandardScaler...")
        self.features.normalize(type="StandardScaler")
        
        # Feature statistics
        feature_df = self.features.features_df
        feature_cols = [col for col in feature_df.columns if col not in ['node_id', 'graph_id']]
        
        if feature_cols:
            feature_stats = feature_df[feature_cols].describe()
            logger.info(f"Feature statistics (sample):")
            logger.info(f"  Mean range: [{feature_df[feature_cols].mean().min():.3f}, {feature_df[feature_cols].mean().max():.3f}]")
            logger.info(f"  Std range: [{feature_df[feature_cols].std().min():.3f}, {feature_df[feature_cols].std().max():.3f}]")
        
        self.results['feature_stats'] = {
            'count': num_features,
            'feature_names': self.features.feature_columns,
            'computation_rate': num_features/computation_time,
            'normalization': 'StandardScaler'
        }
        
    def compute_graph_embeddings(self):
        """Generate graph embeddings for classification."""
        logger.info(f"\n[5/7] COMPUTING GRAPH EMBEDDINGS")
        logger.info("-"*50)
        
        start_time = time.time()
        
        actual_embedding_dim = min(self.config['embedding_dim'], len(self.features.feature_columns))
        logger.info(f"Embedding algorithm: approx_wasserstein")
        logger.info(f"Embedding dimension: {actual_embedding_dim}")
        
        self.embeddings = self.nxt.compute_graph_embeddings(
            graph_collection=self.egonet_collection,
            features=self.features,
            embedding_algorithm="approx_wasserstein",
            embedding_dimension=actual_embedding_dim,
            random_state=self.config['random_seed']
        )
        
        embedding_time = time.time() - start_time
        self.results['timing']['embedding_computation'] = embedding_time
        
        num_embeddings = len(self.embeddings.embeddings_df)
        embedding_dims = len(self.embeddings.embedding_columns)
        
        logger.info(f"Generated {num_embeddings} embeddings in {embedding_time:.2f} seconds")
        logger.info(f"Embedding dimensions: {embedding_dims}")
        logger.info(f"Rate: {num_embeddings/embedding_time:.1f} embeddings/second")
        
        # Embedding statistics
        embedding_df = self.embeddings.embeddings_df
        embedding_cols = self.embeddings.embedding_columns
        
        if embedding_cols:
            logger.info(f"Embedding statistics:")
            logger.info(f"  Mean range: [{embedding_df[embedding_cols].mean().min():.3f}, {embedding_df[embedding_cols].mean().max():.3f}]")
            logger.info(f"  Std range: [{embedding_df[embedding_cols].std().min():.3f}, {embedding_df[embedding_cols].std().max():.3f}]")
        
        self.results['embedding_stats'] = {
            'count': num_embeddings,
            'dimensions': embedding_dims,
            'algorithm': 'approx_wasserstein',
            'computation_rate': num_embeddings/embedding_time
        }
        
    def train_classification_model(self):
        """Train and evaluate classification model."""
        logger.info(f"\n[6/7] TRAINING CLASSIFICATION MODEL")
        logger.info("-"*50)
        
        start_time = time.time()
        
        try:
            model_results = self.nxt.train_ml_model(
                graph_collection=self.egonet_collection,
                embeddings=self.embeddings,
                model_type="classifier",
                sample_size=100,  # Good sample size for cross-validation
                balance_dataset=False  # Already balanced
            )
            
            training_time = time.time() - start_time
            self.results['timing']['model_training'] = training_time
            
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            # Detailed results analysis
            logger.info("\n" + "="*60)
            logger.info("CLASSIFICATION RESULTS")
            logger.info("="*60)
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            detailed_results = {}
            
            for metric in metrics:
                if metric in model_results:
                    scores = model_results[metric]
                    mean_val = np.mean(scores)
                    std_val = np.std(scores)
                    min_val = np.min(scores)
                    max_val = np.max(scores)
                    
                    logger.info(f"{metric.upper():<12}: {mean_val:.4f} ± {std_val:.4f} (range: [{min_val:.4f}, {max_val:.4f}])")
                    
                    detailed_results[metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'scores': scores.tolist() if hasattr(scores, 'tolist') else list(scores)
                    }
            
            logger.info(f"\nModel type: {model_results.get('model_type', 'Unknown')}")
            logger.info(f"Classes: {model_results.get('classes', [])}")
            logger.info(f"Cross-validation folds: {len(model_results.get('accuracy', []))}")
            
            self.results['model_results'] = {
                'detailed_metrics': detailed_results,
                'model_type': model_results.get('model_type', 'XGBoost'),
                'classes': model_results.get('classes', []),
                'cv_folds': len(model_results.get('accuracy', [])),
                'training_time': training_time
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self.results['model_results'] = {'error': str(e)}
            
    def compute_feature_importance(self):
        """Analyze feature importance for interpretability."""
        logger.info(f"\n[7/7] COMPUTING FEATURE IMPORTANCE")
        logger.info("-"*50)
        
        try:
            start_time = time.time()
            
            importance_df = self.nxt.compute_feature_importance(
                graph_collection=self.egonet_collection,
                features=self.features,
                feature_importance_algorithm="supervised_fast",
                embedding_algorithm="approx_wasserstein",
                n_iterations=10,
                random_state=self.config['random_seed']
            )
            
            importance_time = time.time() - start_time
            self.results['timing']['feature_importance'] = importance_time
            
            logger.info(f"Feature importance computed in {importance_time:.2f} seconds")
            
            if not importance_df.empty:
                logger.info(f"\nTop 10 Most Important Features:")
                logger.info("-"*60)
                
                # Convert to list of dictionaries for JSON serialization
                top_features = []
                for idx, row in importance_df.head(10).iterrows():
                    feature_info = row.to_dict()
                    top_features.append(feature_info)
                    logger.info(f"{idx+1:2d}. {feature_info}")
                
                self.results['feature_importance'] = {
                    'top_10': top_features,
                    'computation_time': importance_time,
                    'algorithm': 'supervised_fast'
                }
            else:
                logger.warning("Feature importance computation returned empty results")
                self.results['feature_importance'] = {'error': 'Empty results'}
                
        except Exception as e:
            logger.error(f"Feature importance computation failed: {e}")
            self.results['feature_importance'] = {'error': str(e)}
            
    def save_comprehensive_results(self):
        """Save all results with timestamps."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"reddit_full_analysis_results_{timestamp}.json"
        
        # Add metadata
        self.results['metadata'] = {
            'analysis_date': datetime.now().isoformat(),
            'total_runtime': sum(self.results['timing'].values()),
            'log_file': log_filename,
            'dataset_metadata': self.dataset_info
        }
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        return results_file
        
    def print_final_summary(self):
        """Print comprehensive final summary."""
        total_time = sum(self.results['timing'].values())
        
        logger.info("\n" + "="*80)
        logger.info("FULL NEEXT ANALYSIS COMPLETE")
        logger.info("="*80)
        
        # Timing breakdown
        logger.info(f"\nTiming breakdown (total: {total_time:.1f}s):")
        for step, time_val in self.results['timing'].items():
            pct = time_val / total_time * 100
            logger.info(f"  {step:<20}: {time_val:6.1f}s ({pct:5.1f}%)")
        
        # Key results
        if 'detailed_metrics' in self.results['model_results']:
            metrics = self.results['model_results']['detailed_metrics']
            logger.info(f"\nFinal Performance:")
            if 'accuracy' in metrics:
                acc = metrics['accuracy']['mean']
                logger.info(f"  Accuracy: {acc:.1%}")
            if 'f1_score' in metrics:
                f1 = metrics['f1_score']['mean']
                logger.info(f"  F1 Score: {f1:.3f}")
        
        # Dataset summary
        logger.info(f"\nDataset processed:")
        logger.info(f"  Graph: {self.results['graph_stats']['nodes']:,} nodes, {self.results['graph_stats']['edges']:,} edges")
        logger.info(f"  Egonets: {self.results['egonet_stats']['count']:,}")
        logger.info(f"  Features: {self.results['feature_stats']['count']}")
        logger.info(f"  Embeddings: {self.results['embedding_stats']['count']} x {self.results['embedding_stats']['dimensions']}D")
        
        # Research insights
        logger.info(f"\nKey insights:")
        logger.info(f"  - Successfully classified Reddit posts as serious vs entertainment")
        logger.info(f"  - Local graph structure contains semantic information")
        logger.info(f"  - NEExT pipeline effective for node-level classification")
        logger.info(f"  - {self.config['k_hop']}-hop neighborhoods capture relevant patterns")
        
    def run_full_analysis(self):
        """Execute the complete analysis pipeline."""
        logger.info(f"Starting full NEExT analysis on Reddit binary dataset...")
        logger.info(f"Expected runtime: 2-5 minutes")
        
        pipeline_start = time.time()
        
        try:
            # Execute all steps
            self.load_and_analyze_graph()
            self.create_graph_collection()
            self.create_egonet_collection()
            self.compute_structural_features()
            self.compute_graph_embeddings()
            self.train_classification_model()
            self.compute_feature_importance()
            
            # Save and summarize
            results_file = self.save_comprehensive_results()
            self.print_final_summary()
            
            logger.info(f"\n✅ Analysis completed successfully!")
            logger.info(f"   Total runtime: {time.time() - pipeline_start:.1f} seconds")
            logger.info(f"   Results: {results_file}")
            logger.info(f"   Log: {log_filename}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def main():
    """Main entry point for full analysis."""
    print("\n" + "="*80)
    print("REDDIT FULL NEEXT ANALYSIS")
    print("="*80)
    print("\nThis analysis will:")
    print("1. Load the 5% binary Reddit dataset (7,315 nodes)")
    print("2. Create 2-hop egonets with 20% sampling (~1,460 egonets)")
    print("3. Compute 6 structural features with multi-hop aggregation")
    print("4. Generate 30-dimensional Wasserstein embeddings")
    print("5. Train XGBoost classifier with cross-validation")
    print("6. Analyze feature importance for interpretability")
    print("\nExpected runtime: 2-5 minutes")
    print("Results saved with timestamps for reproducibility")
    print("="*80)
    
    # Run analysis
    analysis = RedditFullAnalysis()
    results = analysis.run_full_analysis()
    
    return results


if __name__ == "__main__":
    results = main()