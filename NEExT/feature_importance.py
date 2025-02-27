from typing import List, Optional, Literal, Dict
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import time
from tqdm import tqdm
from .graph_collection import GraphCollection
from .features import Features
from .graph_embeddings import GraphEmbeddings
from .ml_models import MLModels
from .embeddings import Embeddings

class FeatureImportanceConfig(BaseModel):
    """Configuration for feature importance analysis"""
    algorithm: Literal["supervised_greedy", "supervised_fast", "unsupervised"]
    embedding_algorithm: str = "approx_wasserstein"
    model_name: Literal["xgboost", "random_forest"] = "random_forest"
    random_state: int = 42
    sample_size: int = 5

class FeatureImportance:
    """
    A class for analyzing feature importance in graph embeddings.
    
    This class provides methods to determine the importance of node features
    based on their predictive power in both supervised and unsupervised settings.
    
    Attributes:
        graph_collection (GraphCollection): Collection of graphs to analyze
        features (Features): Features object containing node features
        config (FeatureImportanceConfig): Configuration for importance analysis
    
    Example:
        >>> importance = FeatureImportance(
        ...     graph_collection=collection,
        ...     features=features,
        ...     algorithm="supervised_greedy",
        ...     embedding_algorithm="approx_wasserstein"
        ... )
        >>> results_df = importance.compute()
    """
    
    def __init__(
        self,
        graph_collection: GraphCollection,
        features: Features,
        algorithm: str,
        embedding_algorithm: str = "approx_wasserstein",
        random_state: int = 42,
        n_iterations: int = 5  # Keep for backward compatibility
    ):
        """Initialize the FeatureImportance analyzer."""
        self.config = FeatureImportanceConfig(
            algorithm=algorithm,
            embedding_algorithm=embedding_algorithm,
            random_state=random_state,
            sample_size=n_iterations  # Use n_iterations as sample_size
        )
        self.graph_collection = graph_collection
        self.features = features
        
        # Define available algorithms
        self.available_algorithms = {
            "supervised_greedy": self._supervised_greedy,
            "supervised_fast": self._supervised_fast,
            "unsupervised": self._unsupervised
        }
    
    def compute(self) -> pd.DataFrame:
        """
        Compute feature importance based on the configured algorithm.
        
        Returns:
            pd.DataFrame: DataFrame containing feature importance results
        """
        if self.config.algorithm not in self.available_algorithms:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
        
        start_time = time.time()
        results = self.available_algorithms[self.config.algorithm]()
        total_time = time.time() - start_time
        
        # Add total computation time to results
        results['total_time'] = total_time
        
        return results
    
    def _supervised_greedy(self) -> pd.DataFrame:
        """
        Compute feature importance using supervised greedy algorithm.
        
        This method determines feature importance by iteratively selecting features
        that maximize model performance when combined with previously selected features.
        
        Returns:
            pd.DataFrame: Results containing:
                - feature_name: Name of the feature
                - avg_performance: Average model performance at each step
                - embedding_algorithm: Name of embedding algorithm used
                - total_time: Total computation time in seconds
        """
        available_features = self.features.feature_columns.copy()
        selected_features = []
        performance_scores = []
        
        # Create progress bar for the outer loop
        pbar = tqdm(
            total=len(available_features),
            desc="Selecting features",
            position=0,
            leave=True
        )
        
        while available_features:
            best_score = float('-inf')
            best_feature = None
            
            # Create progress bar for the inner loop
            inner_pbar = tqdm(
                available_features,
                desc=f"Testing features (selected: {len(selected_features)})",
                position=1,
                leave=False
            )
            
            # Try each remaining feature
            for feature in inner_pbar:
                current_features = selected_features + [feature]
                inner_pbar.set_postfix({'testing': feature}, refresh=True)
                
                # Create embeddings using current feature set
                embeddings = GraphEmbeddings(
                    graph_collection=self.graph_collection,
                    features=self.features,
                    embedding_algorithm=self.config.embedding_algorithm,
                    embedding_dimension=len(current_features),
                    feature_columns=current_features,
                    random_state=self.config.random_state
                ).compute()
                
                # Train and evaluate model
                scores = []
                ml_model = MLModels(
                    graph_collection=self.graph_collection,
                    embeddings=embeddings,
                    model_type="classifier" if isinstance(
                        self.graph_collection.graphs[0].graph_label, 
                        (int, np.integer)
                    ) else "regressor",
                    random_state=self.config.random_state,
                    sample_size=self.config.sample_size  # Use sample_size instead of n_iterations
                )
                results = ml_model.compute()
                
                # Get performance metric
                if results["model_type"] == "classifier":
                    scores = results["accuracy"]  # Use all scores directly
                else:
                    scores = [-score for score in results["rmse"]]  # Negative RMSE for maximization
                
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_feature = feature
            
            # Close inner progress bar
            inner_pbar.close()
            
            # Add best feature to selected features
            selected_features.append(best_feature)
            available_features.remove(best_feature)
            performance_scores.append(abs(best_score))  # Convert back to positive RMSE if needed
            
            # Update outer progress bar
            pbar.update(1)
            pbar.set_postfix({'best_feature': best_feature, 'score': abs(best_score)}, refresh=True)
        
        # Close outer progress bar
        pbar.close()
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'feature_name': selected_features,
            'avg_performance': performance_scores,
            'embedding_algorithm': self.config.embedding_algorithm
        })
        
        return results_df
    
    def _supervised_fast(self) -> pd.DataFrame:
        """
        Compute feature importance using supervised fast algorithm.
        
        This method:
        1. Determines feature importance order using Random Forest on 1D embeddings
        2. Evaluates performance by iteratively building models with increasing feature sets
        3. Returns results in same format as greedy method for consistency
        
        Returns:
            pd.DataFrame: Results containing:
                - feature_name: Name of the feature in order of importance
                - avg_performance: Performance using features up to this point
                - embedding_algorithm: Name of embedding algorithm used
                - total_time: Total computation time in seconds
        """
        start_time = time.time()
        feature_embeddings = []
        
        # Create progress bar for initial embeddings
        pbar = tqdm(
            self.features.feature_columns,
            desc="Computing initial embeddings",
            position=0,
            leave=True
        )
        
        # Generate 1D embeddings for each feature
        for feature in pbar:
            pbar.set_postfix({'feature': feature}, refresh=True)
            
            embeddings = GraphEmbeddings(
                graph_collection=self.graph_collection,
                features=self.features,
                embedding_algorithm=self.config.embedding_algorithm,
                embedding_dimension=1,
                feature_columns=[feature],
                random_state=self.config.random_state
            ).compute()
            
            embedding_df = embeddings.embeddings_df.copy()
            embedding_df.rename(columns={'emb_0': feature}, inplace=True)
            feature_embeddings.append(embedding_df)
        
        # Merge all embeddings
        merged_df = feature_embeddings[0]
        for df in feature_embeddings[1:]:
            merged_df = pd.merge(merged_df, df, on='graph_id', how='outer')
        
        # Get feature importance order using Random Forest
        embeddings = Embeddings(
            embeddings_df=merged_df,
            embedding_name=self.config.embedding_algorithm,
            embedding_columns=self.features.feature_columns
        )
        
        ml_model = MLModels(
            graph_collection=self.graph_collection,
            embeddings=embeddings,
            model_type="classifier" if isinstance(
                self.graph_collection.graphs[0].graph_label, 
                (int, np.integer)
            ) else "regressor",
            model_name="random_forest",
            compute_feature_importance=True,
            sample_size=self.config.sample_size,
            random_state=self.config.random_state
        )
        
        results = ml_model.compute()
        ordered_features = results['feature_importance'].index.tolist()
        
        # Evaluate performance iteratively
        performance_scores = []
        
        # Create progress bar for performance evaluation
        pbar = tqdm(
            range(len(ordered_features)),
            desc="Evaluating feature combinations",
            position=0,
            leave=True
        )
        
        # Evaluate each feature combination
        for i in pbar:
            current_features = ordered_features[:i+1]
            pbar.set_postfix({'n_features': len(current_features)}, refresh=True)
            
            # Create embeddings using current feature set
            embeddings = GraphEmbeddings(
                graph_collection=self.graph_collection,
                features=self.features,
                embedding_algorithm=self.config.embedding_algorithm,
                embedding_dimension=len(current_features),  # Embedding size matches feature count
                feature_columns=current_features,
                random_state=self.config.random_state
            ).compute()
            
            # Train and evaluate model
            ml_model = MLModels(
                graph_collection=self.graph_collection,
                embeddings=embeddings,
                model_type="classifier" if isinstance(
                    self.graph_collection.graphs[0].graph_label, 
                    (int, np.integer)
                ) else "regressor",
                model_name="random_forest",
                sample_size=self.config.sample_size,
                random_state=self.config.random_state
            )
            
            results = ml_model.compute()
            
            # Get performance metric
            if results["model_type"] == "classifier":
                score = np.mean(results["accuracy"])
            else:
                score = -np.mean(results["rmse"])  # Negative RMSE for consistency
            
            performance_scores.append(abs(score))  # Convert back to positive RMSE if needed
        
        total_time = time.time() - start_time
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'feature_name': ordered_features,
            'avg_performance': performance_scores,
            'embedding_algorithm': self.config.embedding_algorithm,
            'total_time': total_time
        })
        
        return results_df
    
    def _unsupervised(self) -> pd.DataFrame:
        """Compute feature importance using unsupervised algorithm."""
        pass
