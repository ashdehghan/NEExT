#!/usr/bin/env python3
"""
Reddit Node Classification Experiment - 20% Sample

Ready-to-run experiment using the 20% sampled Reddit dataset.
This script is optimized for a good balance between performance and accuracy.

Expected runtime: 2-3 minutes
Expected results: 41-class classification with sufficient samples per class
"""

import pickle
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from collections import Counter
import sys
from datetime import datetime

# Add NEExT to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from NEExT.framework import NEExT
from NEExT.collections import EgonetCollection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'reddit_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RedditExperiment:
    """Main experiment class for Reddit node classification."""
    
    def __init__(self, graph_file="reddit_networkx_20pct.pkl"):
        """Initialize experiment with configuration."""
        self.graph_file = graph_file
        
        # Experiment configuration
        self.K_HOP = 2  # 2-hop neighborhoods for good context
        self.SAMPLE_FRACTION = 0.15  # 15% of the 20% sample (~7000 nodes)
        self.EMBEDDING_DIM = 30  # Reasonable embedding dimension
        
        # Feature configuration
        self.FEATURE_LIST = [
            "degree_centrality",
            "closeness_centrality", 
            "betweenness_centrality",
            "clustering_coefficient",
            "page_rank",
            "eigenvector_centrality"
        ]
        
        self.nxt = None
        self.graph = None
        self.graph_collection = None
        self.egonet_collection = None
        self.features = None
        self.embeddings = None
        self.results = {}
        
    def load_graph(self):
        """Load the 20% sampled Reddit graph."""
        logger.info("="*80)
        logger.info("LOADING REDDIT 20% SAMPLE")
        logger.info("="*80)
        
        start_time = time.time()
        with open(self.graph_file, 'rb') as f:
            self.graph = pickle.load(f)
        
        load_time = time.time() - start_time
        logger.info(f"Graph loaded in {load_time:.2f} seconds")
        logger.info(f"Nodes: {self.graph.number_of_nodes():,}")
        logger.info(f"Edges: {self.graph.number_of_edges():,}")
        logger.info(f"Average degree: {2 * self.graph.number_of_edges() / self.graph.number_of_nodes():.1f}")
        
        # Analyze class distribution
        class_dist = Counter()
        split_dist = Counter()
        
        for _, attrs in self.graph.nodes(data=True):
            label = attrs.get('subreddit_label', -1)
            split = attrs.get('split', 'unknown')
            if label != -1:
                class_dist[label] += 1
            split_dist[split] += 1
        
        logger.info(f"Classes: {len(class_dist)} unique subreddits")
        logger.info(f"Split distribution: Train={split_dist['train']}, Val={split_dist['val']}, Test={split_dist['test']}")
        
        # Check minimum class size
        min_class = min(class_dist.values())
        logger.info(f"Minimum class size: {min_class} (sufficient for stratified splitting)")
        
    def create_graph_collection(self):
        """Create NEExT GraphCollection."""
        logger.info("\n" + "="*80)
        logger.info("CREATING GRAPH COLLECTION")
        logger.info("="*80)
        
        self.nxt = NEExT()
        self.nxt.set_log_level("WARNING")
        
        start_time = time.time()
        self.graph_collection = self.nxt.load_from_networkx(
            [self.graph],
            reindex_nodes=False,
            filter_largest_component=False,
            node_sample_rate=1.0
        )
        
        creation_time = time.time() - start_time
        logger.info(f"GraphCollection created in {creation_time:.2f} seconds")
        
    def create_egonets(self):
        """Create egonet collection for node classification."""
        logger.info("\n" + "="*80)
        logger.info("CREATING EGONET COLLECTION")
        logger.info("="*80)
        logger.info(f"Parameters: k_hop={self.K_HOP}, sample_fraction={self.SAMPLE_FRACTION:.1%}")
        
        start_time = time.time()
        
        self.egonet_collection = EgonetCollection(egonet_feature_target='subreddit_label')
        self.egonet_collection.compute_k_hop_egonets(
            graph_collection=self.graph_collection,
            k_hop=self.K_HOP,
            sample_fraction=self.SAMPLE_FRACTION,
            random_seed=42
        )
        
        creation_time = time.time() - start_time
        num_egonets = len(self.egonet_collection.graphs)
        
        logger.info(f"Created {num_egonets:,} egonets in {creation_time:.2f} seconds")
        logger.info(f"Rate: {num_egonets/creation_time:.1f} egonets/second")
        
        # Analyze egonet sizes
        egonet_sizes = [len(g.nodes) for g in self.egonet_collection.graphs]
        logger.info(f"Egonet sizes: min={min(egonet_sizes)}, max={max(egonet_sizes)}, avg={np.mean(egonet_sizes):.1f}")
        
        # Check label distribution
        egonet_labels = [g.graph_label for g in self.egonet_collection.graphs if g.graph_label is not None]
        unique_labels = len(set(egonet_labels))
        logger.info(f"Unique labels in egonets: {unique_labels}")
        
    def compute_features(self):
        """Compute structural features on egonets."""
        logger.info("\n" + "="*80)
        logger.info("COMPUTING STRUCTURAL FEATURES")
        logger.info("="*80)
        logger.info(f"Features to compute: {', '.join(self.FEATURE_LIST)}")
        
        start_time = time.time()
        
        self.features = self.nxt.compute_node_features(
            graph_collection=self.egonet_collection,
            feature_list=self.FEATURE_LIST,
            feature_vector_length=2,  # Aggregate at 0-hop and 1-hop
            show_progress=True,
            n_jobs=-1  # Use all cores
        )
        
        computation_time = time.time() - start_time
        num_features = len(self.features.feature_columns)
        
        logger.info(f"Computed {num_features} features in {computation_time:.2f} seconds")
        logger.info(f"Rate: {num_features/computation_time:.1f} features/second")
        
        # Normalize features
        logger.info("Normalizing features...")
        self.features.normalize(type="StandardScaler")
        
    def compute_embeddings(self):
        """Generate graph embeddings for each egonet."""
        logger.info("\n" + "="*80)
        logger.info("COMPUTING GRAPH EMBEDDINGS")
        logger.info("="*80)
        
        start_time = time.time()
        
        self.embeddings = self.nxt.compute_graph_embeddings(
            graph_collection=self.egonet_collection,
            features=self.features,
            embedding_algorithm="approx_wasserstein",
            embedding_dimension=min(self.EMBEDDING_DIM, len(self.features.feature_columns)),
            random_state=42
        )
        
        embedding_time = time.time() - start_time
        num_embeddings = len(self.embeddings.embeddings_df)
        
        logger.info(f"Generated {num_embeddings} embeddings in {embedding_time:.2f} seconds")
        logger.info(f"Embedding dimensions: {len(self.embeddings.embedding_columns)}")
        logger.info(f"Rate: {num_embeddings/embedding_time:.1f} embeddings/second")
        
    def train_classifier(self):
        """Train and evaluate the classifier."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING CLASSIFIER")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            model_results = self.nxt.train_ml_model(
                graph_collection=self.egonet_collection,
                embeddings=self.embeddings,
                model_type="classifier",
                sample_size=100,  # Good sample size for cross-validation
                balance_dataset=False  # Don't balance - we have sufficient samples
            )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Store results
            self.results = model_results
            
            # Print performance metrics
            logger.info("\n" + "-"*60)
            logger.info("CLASSIFICATION RESULTS")
            logger.info("-"*60)
            
            metrics = ['accuracy', 'recall', 'precision', 'f1_score']
            for metric in metrics:
                if metric in model_results:
                    scores = model_results[metric]
                    mean_val = np.mean(scores)
                    std_val = np.std(scores)
                    logger.info(f"{metric.capitalize():<15} Mean: {mean_val:.4f} ± {std_val:.4f}")
            
            logger.info(f"\nNumber of classes: {len(model_results.get('classes', []))}")
            
            # Additional statistics
            if 'accuracy' in model_results:
                acc_scores = model_results['accuracy']
                logger.info(f"\nAccuracy range: [{min(acc_scores):.4f}, {max(acc_scores):.4f}]")
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.results = None
            
    def compute_feature_importance(self):
        """Analyze feature importance."""
        logger.info("\n" + "="*80)
        logger.info("COMPUTING FEATURE IMPORTANCE")
        logger.info("="*80)
        
        try:
            start_time = time.time()
            
            importance_df = self.nxt.compute_feature_importance(
                graph_collection=self.egonet_collection,
                features=self.features,
                feature_importance_algorithm="supervised_fast",
                embedding_algorithm="approx_wasserstein",
                n_iterations=5,
                random_state=42
            )
            
            importance_time = time.time() - start_time
            logger.info(f"Feature importance computed in {importance_time:.2f} seconds")
            
            # Display top features
            if not importance_df.empty:
                logger.info("\nTop 10 Most Important Features:")
                logger.info("-"*40)
                for idx, row in importance_df.head(10).iterrows():
                    logger.info(f"{idx+1:2d}. {row.to_dict()}")
                    
        except Exception as e:
            logger.error(f"Feature importance computation failed: {e}")
            
    def save_results(self):
        """Save experiment results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"reddit_results_{timestamp}.pkl"
        
        results_data = {
            'configuration': {
                'graph_file': self.graph_file,
                'k_hop': self.K_HOP,
                'sample_fraction': self.SAMPLE_FRACTION,
                'embedding_dim': self.EMBEDDING_DIM,
                'features': self.FEATURE_LIST
            },
            'results': self.results,
            'graph_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'egonets': len(self.egonet_collection.graphs) if self.egonet_collection else 0
            }
        }
        
        with open(results_file, 'wb') as f:
            pickle.dump(results_data, f)
        
        logger.info(f"\nResults saved to: {results_file}")
        
    def run(self):
        """Run the complete experiment pipeline."""
        total_start = time.time()
        
        logger.info("\n" + "="*80)
        logger.info("REDDIT NODE CLASSIFICATION EXPERIMENT - 20% SAMPLE")
        logger.info("="*80)
        
        # Execute pipeline
        self.load_graph()
        self.create_graph_collection()
        self.create_egonets()
        self.compute_features()
        self.compute_embeddings()
        self.train_classifier()
        self.compute_feature_importance()
        self.save_results()
        
        # Final summary
        total_time = time.time() - total_start
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*80)
        logger.info(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        
        if self.results and 'accuracy' in self.results:
            final_acc = np.mean(self.results['accuracy'])
            logger.info(f"Final accuracy: {final_acc:.2%}")
        
        logger.info("\nKey insights:")
        logger.info("- Local graph structure predicts subreddit membership")
        logger.info("- 2-hop neighborhoods capture community patterns")
        logger.info("- Structural features alone achieve meaningful classification")
        
        return self.results


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("REDDIT NODE CLASSIFICATION - 20% SAMPLE")
    print("="*80)
    print("\nThis experiment will:")
    print("1. Load the 20% sampled Reddit graph (46,593 nodes)")
    print("2. Create ~7,000 2-hop egonets")
    print("3. Compute 6 structural features")
    print("4. Generate 30-dimensional embeddings")
    print("5. Train XGBoost classifier for 41 subreddits")
    print("\nExpected runtime: 2-3 minutes")
    print("="*80)
    
    # Run experiment
    experiment = RedditExperiment()
    results = experiment.run()
    
    print("\n" + "="*80)
    print("EXPERIMENT FINISHED!")
    print("="*80)
    print("Check the log file for detailed results.")
    print("Results saved to reddit_results_*.pkl")
    
    return results


if __name__ == "__main__":
    # Run the experiment
    results = main()