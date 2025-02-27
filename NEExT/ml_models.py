from typing import Dict, List, Optional, Union, Literal, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from pydantic import BaseModel, Field, validator
import logging
from .graph_collection import GraphCollection
from .embeddings import Embeddings

# Set up logger
logger = logging.getLogger(__name__)

# Try to import XGBoost, but provide fallbacks if it fails
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available. Using sklearn models as fallback.")
    XGBOOST_AVAILABLE = False

# Try to import SMOTE, but handle if it's not available
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    logger.warning("imbalanced-learn not available. Dataset balancing will be disabled.")
    SMOTE_AVAILABLE = False

# Import sklearn models as fallbacks
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

class MLModelsConfig(BaseModel):
    """
    Configuration for machine learning models.
    
    Attributes:
        model_type: Type of model to train ("classifier" or "regressor")
        model_name: Name of the model algorithm
        balance_dataset: Whether to balance the dataset for classification
        compute_feature_importance: Whether to compute feature importance
        sample_size: Number of training/testing iterations
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        parallel_backend: Parallelization backend ("process" or "thread")
    """
    model_type: Literal["classifier", "regressor"]
    model_name: Literal["xgboost", "random_forest"] = "xgboost"
    balance_dataset: bool = Field(default=False, description="Whether to balance the dataset for classification")
    compute_feature_importance: bool = Field(default=False, description="Whether to compute feature importance")
    sample_size: int = Field(default=5, description="Number of training/testing iterations")
    test_size: float = Field(default=0.3, description="Proportion of the dataset to include in the test split")
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    n_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 for all CPUs)")
    parallel_backend: Literal["process", "thread"] = Field(default="process", description="Parallelization backend")
    
    @validator('sample_size')
    def validate_sample_size(cls, v):
        """Validate that sample_size is positive."""
        if v <= 0:
            raise ValueError("sample_size must be positive")
        return v
    
    @validator('test_size')
    def validate_test_size(cls, v):
        """Validate that test_size is between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError("test_size must be between 0 and 1")
        return v
    
    @validator('n_jobs')
    def validate_n_jobs(cls, v):
        """Convert n_jobs=-1 to the number of available CPUs."""
        if v == -1:
            return multiprocessing.cpu_count()
        if v <= 0 and v != -1:
            raise ValueError("n_jobs must be positive or -1")
        return v

class MLModels:
    """
    A class for training and evaluating machine learning models on graph embeddings.
    
    This class provides methods to train classification or regression models
    using graph embeddings as features and graph labels as targets. It supports
    parallel processing for model training and evaluation.
    
    Attributes:
        graph_collection (GraphCollection): Collection of graphs with labels
        embeddings_df (pd.DataFrame): DataFrame containing graph embeddings
        config (MLModelsConfig): Configuration for model training and evaluation
        data_df (pd.DataFrame): Merged DataFrame with embeddings and labels
        label_encoder (LabelEncoder): Encoder for categorical labels (classification only)
        num_classes (int): Number of unique classes (classification only)
    """
    
    def __init__(
        self,
        graph_collection: GraphCollection,
        embeddings: Embeddings,
        model_type: str = "classifier",
        model_name: str = "xgboost",
        balance_dataset: bool = False,
        compute_feature_importance: bool = False,
        sample_size: int = 5,
        test_size: float = 0.3,
        random_state: int = 42,
        n_jobs: int = -1,
        parallel_backend: str = "process"
    ):
        """Initialize the MLModels processor."""
        self.config = MLModelsConfig(
            model_type=model_type,
            model_name=model_name,
            balance_dataset=balance_dataset,
            compute_feature_importance=compute_feature_importance,
            sample_size=sample_size,
            test_size=test_size,
            random_state=random_state,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend
        )
        self.graph_collection = graph_collection
        self.data_df = embeddings.embeddings_df
        
        # Check if graphs have labels
        if not self._check_graph_labels():
            raise ValueError("Graph collection must have labels for all graphs")
        
        # Prepare labels DataFrame
        self.labels_df = self._prepare_labels_df()
        
        # Merge embeddings with labels
        self.data_df = pd.merge(
            self.data_df, 
            self.labels_df, 
            on="graph_id"
        )
        
        # Encode labels for classification
        if self.config.model_type == "classifier":
            self._encode_labels()
    
    def _check_graph_labels(self) -> bool:
        """
        Check if all graphs in the collection have labels.
        
        Returns:
            bool: True if all graphs have labels, False otherwise
        """
        return all(g.graph_label is not None for g in self.graph_collection.graphs)
    
    def _prepare_labels_df(self) -> pd.DataFrame:
        """
        Prepare DataFrame with graph IDs and labels.
        
        Returns:
            pd.DataFrame: DataFrame with graph_id and label columns
        """
        graph_ids = []
        graph_labels = []
        
        for graph in self.graph_collection.graphs:
            graph_ids.append(graph.graph_id)
            graph_labels.append(graph.graph_label)
        
        return pd.DataFrame({
            "graph_id": graph_ids,
            "label": graph_labels
        })
    
    def _encode_labels(self) -> None:
        """
        Encode categorical labels as integers for classification tasks.
        
        This method creates a new column 'encoded_label' in the data_df
        and sets the num_classes attribute.
        """
        self.label_encoder = LabelEncoder()
        self.data_df["encoded_label"] = self.label_encoder.fit_transform(self.data_df["label"])
        self.num_classes = len(self.label_encoder.classes_)
    
    def compute(self) -> Dict[str, Any]:
        """
        Train and evaluate models based on the configuration.
        
        Returns:
            Dict: Dictionary containing model information and evaluation metrics
                - model_type: Type of model ("classifier" or "regressor")
                - accuracy, recall, precision, f1_score: Lists of metrics (classifier)
                - rmse, mae: Lists of metrics (regressor)
                - model: Trained model (last iteration)
                - classes: List of class labels (classifier only)
                - feature_columns: List of feature columns used for training
        """
        if self.config.model_type == "classifier":
            return self._compute_classifier()
        else:
            return self._compute_regressor()
    
    def _train_classifier_iteration(self, iteration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate a classifier for a single iteration."""
        i = iteration_data['i']
        X = iteration_data['X']
        y = iteration_data['y']
        feature_cols = iteration_data['feature_cols']
        test_size = iteration_data['test_size']
        random_state = iteration_data['random_state'] + i
        balance_dataset = iteration_data['balance_dataset']
        num_classes = iteration_data['num_classes']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X[feature_cols], y, test_size=test_size, random_state=random_state
        )
        
        # Balance dataset if requested
        if balance_dataset:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Train model based on selected algorithm
        if self.config.model_name == "xgboost" and XGBOOST_AVAILABLE:
            model = XGBClassifier(random_state=random_state, n_jobs=1)
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=1
            )
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get feature importance if requested
        feature_importance = None
        if self.config.compute_feature_importance:
            if self.config.model_name == "xgboost" and XGBOOST_AVAILABLE:
                importance = model.feature_importances_
            else:
                importance = model.feature_importances_
            
            # Create feature importance ranking
            importance_dict = dict(zip(feature_cols, importance))
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            feature_importance = {feat: rank for rank, (feat, _) in enumerate(sorted_features, 1)}
        
        # Return results
        return {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred, average='macro'),
            'precision': precision_score(y_test, y_pred, average='macro'),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'feature_importance': feature_importance
        }
    
    def _compute_classifier(self) -> Dict[str, Any]:
        """
        Train and evaluate classification models in parallel.
        
        Returns:
            Dict: Dictionary containing model information and evaluation metrics
                - model_type: "classifier"
                - accuracy, recall, precision, f1_score: Lists of metrics
                - model: Trained model (last iteration)
                - classes: List of class labels
                - feature_columns: List of feature columns used for training
        """
        # Prepare feature columns (all columns except graph_id, label, and encoded_label)
        feature_cols = [col for col in self.data_df.columns 
                       if col not in ["graph_id", "label", "encoded_label"]]
        
        # Prepare iteration data
        iteration_data = [
            {
                'i': i,
                'X': self.data_df,
                'y': self.data_df["encoded_label"],
                'feature_cols': feature_cols,
                'test_size': self.config.test_size,
                'random_state': self.config.random_state,
                'balance_dataset': self.config.balance_dataset,
                'num_classes': self.num_classes
            }
            for i in range(self.config.sample_size)
        ]
        
        # Run iterations in parallel
        executor_class = ProcessPoolExecutor if self.config.parallel_backend == "process" else ThreadPoolExecutor
        results = []
        
        with executor_class(max_workers=self.config.n_jobs) as executor:
            results = list(executor.map(self._train_classifier_iteration, iteration_data))
        
        # Collect results
        models = [r['model'] for r in results]
        accuracy_scores = [r['accuracy'] for r in results]
        recall_scores = [r['recall'] for r in results]
        precision_scores = [r['precision'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        
        # Process feature importance if computed
        feature_importance_df = None
        if self.config.compute_feature_importance:
            # Collect all rankings
            all_rankings = []
            for r in results:
                all_rankings.append(r['feature_importance'])
            
            # Calculate average rank for each feature
            feature_ranks = {}
            for feature in feature_cols:
                ranks = [r[feature] for r in all_rankings]
                feature_ranks[feature] = np.mean(ranks)
            
            # Create sorted DataFrame
            feature_importance_df = pd.DataFrame(
                {'average_rank': feature_ranks}
            ).sort_values('average_rank')
        
        # Return results with means and standard deviations
        return {
            "model_type": "classifier",
            "accuracy": accuracy_scores,
            "accuracy_mean": np.mean(accuracy_scores),
            "accuracy_std": np.std(accuracy_scores),
            "recall": recall_scores,
            "recall_mean": np.mean(recall_scores),
            "recall_std": np.std(recall_scores),
            "precision": precision_scores,
            "precision_mean": np.mean(precision_scores),
            "precision_std": np.std(precision_scores),
            "f1_score": f1_scores,
            "f1_score_mean": np.mean(f1_scores),
            "f1_score_std": np.std(f1_scores),
            "model": models[-1],
            "classes": self.label_encoder.classes_,
            "feature_columns": feature_cols,
            "feature_importance": feature_importance_df
        }
    
    def _train_regressor_iteration(self, iteration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train and evaluate a regressor for a single iteration."""
        i = iteration_data['i']
        X = iteration_data['X']
        y = iteration_data['y']
        feature_cols = iteration_data['feature_cols']
        test_size = iteration_data['test_size']
        random_state = iteration_data['random_state'] + i
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X[feature_cols], y, test_size=test_size, random_state=random_state
        )
        
        # Train model based on selected algorithm
        if self.config.model_name == "xgboost" and XGBOOST_AVAILABLE:
            model = XGBRegressor(random_state=random_state, n_jobs=1)
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=1
            )
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Get feature importance if requested
        feature_importance = None
        if self.config.compute_feature_importance:
            if self.config.model_name == "xgboost" and XGBOOST_AVAILABLE:
                importance = model.feature_importances_
            else:
                importance = model.feature_importances_
            
            # Create feature importance ranking
            importance_dict = dict(zip(feature_cols, importance))
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            feature_importance = {feat: rank for rank, (feat, _) in enumerate(sorted_features, 1)}
        
        # Return results
        return {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'feature_importance': feature_importance
        }
    
    def _compute_regressor(self) -> Dict[str, Any]:
        """
        Train and evaluate regression models in parallel.
        
        Returns:
            Dict: Dictionary containing model information and evaluation metrics
                - model_type: "regressor"
                - rmse, mae: Lists of metrics
                - model: Trained model (last iteration)
                - feature_columns: List of feature columns used for training
        """
        # Prepare feature columns (all columns except graph_id and label)
        feature_cols = [col for col in self.data_df.columns 
                       if col not in ["graph_id", "label"]]
        
        # Prepare iteration data
        iteration_data = [
            {
                'i': i,
                'X': self.data_df,
                'y': self.data_df["label"],
                'feature_cols': feature_cols,
                'test_size': self.config.test_size,
                'random_state': self.config.random_state
            }
            for i in range(self.config.sample_size)
        ]
        
        # Run iterations in parallel
        executor_class = ProcessPoolExecutor if self.config.parallel_backend == "process" else ThreadPoolExecutor
        results = []
        
        with executor_class(max_workers=self.config.n_jobs) as executor:
            results = list(executor.map(self._train_regressor_iteration, iteration_data))
        
        # Collect results
        models = [r['model'] for r in results]
        rmse_scores = [r['rmse'] for r in results]
        mae_scores = [r['mae'] for r in results]
        
        # Process feature importance if computed
        feature_importance_df = None
        if self.config.compute_feature_importance:
            # Collect all rankings
            all_rankings = []
            for r in results:
                all_rankings.append(r['feature_importance'])
            
            # Calculate average rank for each feature
            feature_ranks = {}
            for feature in feature_cols:
                ranks = [r[feature] for r in all_rankings]
                feature_ranks[feature] = np.mean(ranks)
            
            # Create sorted DataFrame
            feature_importance_df = pd.DataFrame(
                {'average_rank': feature_ranks}
            ).sort_values('average_rank')
        
        # Return results
        return {
            "model_type": "regressor",
            "rmse": rmse_scores,
            "rmse_mean": np.mean(rmse_scores),
            "rmse_std": np.std(rmse_scores),
            "mae": mae_scores,
            "mae_mean": np.mean(mae_scores),
            "mae_std": np.std(mae_scores),
            "model": models[-1],
            "feature_columns": feature_cols,
            "feature_importance": feature_importance_df
        }
