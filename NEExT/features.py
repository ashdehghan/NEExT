from typing import List, Optional, Literal
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class FeaturesConfig(BaseModel):
    """Configuration for features normalization"""
    scaler_type: Literal["StandardScaler", "MinMaxScaler", "RobustScaler"] = "StandardScaler"

class Features:
    """
    A class for managing feature data and operations.
    
    This class provides a container for feature data and operations like
    normalization and merging of features.
    
    Attributes:
        features_df (pd.DataFrame): DataFrame containing the features
        feature_columns (List[str]): List of feature column names
    
    Example:
        >>> features = Features(df, ["page_rank", "degree_centrality"])
        >>> features.normalize(type="StandardScaler")
        >>> merged = features1 + features2
    """
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        feature_columns: List[str]
    ):
        """Initialize the Features object."""
        self.features_df = features_df
        self.feature_columns = feature_columns
        
        # Validate that all feature columns exist in DataFrame
        missing_cols = [col for col in feature_columns if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Features {missing_cols} not found in DataFrame")
    
    def normalize(self, type: str = "StandardScaler") -> None:
        """
        Normalize features using the specified scaler.
        
        Args:
            type: Type of scaler to use ("StandardScaler", "MinMaxScaler", "RobustScaler")
        """
        config = FeaturesConfig(scaler_type=type)
        
        if not self.features_df.empty:
            # Get feature columns (exclude node_id and graph_id)
            feature_cols = [col for col in self.features_df.columns 
                          if col not in ['node_id', 'graph_id']]
            
            if feature_cols:
                # Initialize scaler
                scalers = {
                    "StandardScaler": StandardScaler(),
                    "MinMaxScaler": MinMaxScaler(),
                    "RobustScaler": RobustScaler()
                }
                scaler = scalers[config.scaler_type]
                
                # Fit and transform the feature columns
                self.features_df[feature_cols] = scaler.fit_transform(
                    self.features_df[feature_cols]
                )
    
    def __add__(self, other: 'Features') -> 'Features':
        """
        Merge two Features objects.
        
        Args:
            other: Another Features object to merge with
            
        Returns:
            Features: New Features object containing merged data
        """
        if not isinstance(other, Features):
            raise TypeError("Can only add Features objects together")
        
        # Merge DataFrames on node_id and graph_id
        merged_df = pd.merge(
            self.features_df,
            other.features_df,
            on=['node_id', 'graph_id'],
            how='outer'
        )
        
        # Combine feature columns
        merged_columns = list(set(self.feature_columns + other.feature_columns))
        
        return Features(merged_df, merged_columns)

