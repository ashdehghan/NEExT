from typing import List
import pandas as pd
from pydantic import BaseModel

class EmbeddingsConfig(BaseModel):
    """Configuration for embeddings"""
    embedding_name: str
    embedding_columns: List[str]

class Embeddings:
    """
    A class for managing graph embeddings data and operations.
    
    This class provides a container for embeddings data and operations like
    merging different types of embeddings.
    
    Attributes:
        embeddings_df (pd.DataFrame): DataFrame containing the embeddings
        embedding_name (str): Name of the embedding algorithm used
        embedding_columns (List[str]): List of embedding column names
    
    Example:
        >>> embeddings1 = Embeddings(df1, "wasserstein", ["emb_0", "emb_1"])
        >>> embeddings2 = Embeddings(df2, "approx_wasserstein", ["emb_0", "emb_1"])
        >>> merged = embeddings1 + embeddings2  # Combines embeddings with unique column names
    """
    
    def __init__(
        self,
        embeddings_df: pd.DataFrame,
        embedding_name: str,
        embedding_columns: List[str]
    ):
        """Initialize the Embeddings object."""
        self.embeddings_df = embeddings_df
        self.embedding_name = embedding_name
        self.embedding_columns = embedding_columns
        
        # Validate that all embedding columns exist in DataFrame
        missing_cols = [col for col in embedding_columns if col not in embeddings_df.columns]
        if missing_cols:
            raise ValueError(f"Embedding columns {missing_cols} not found in DataFrame")
    
    def __add__(self, other: 'Embeddings') -> 'Embeddings':
        """
        Merge two Embeddings objects.
        
        Args:
            other: Another Embeddings object to merge with
            
        Returns:
            Embeddings: New Embeddings object containing merged data with unique column names
        """
        if not isinstance(other, Embeddings):
            raise TypeError("Can only add Embeddings objects together")
        
        # Create copies of DataFrames to avoid modifying originals
        df1 = self.embeddings_df.copy()
        df2 = other.embeddings_df.copy()
        
        # Rename columns to include embedding name prefix
        rename_dict1 = {
            col: f"{self.embedding_name}_{col}" 
            for col in self.embedding_columns
        }
        rename_dict2 = {
            col: f"{other.embedding_name}_{col}" 
            for col in other.embedding_columns
        }
        
        df1.rename(columns=rename_dict1, inplace=True)
        df2.rename(columns=rename_dict2, inplace=True)
        
        # Merge DataFrames on graph_id
        merged_df = pd.merge(
            df1,
            df2,
            on='graph_id',
            how='outer'
        )
        
        # Create new column names list
        new_columns = list(rename_dict1.values()) + list(rename_dict2.values())
        
        # Create new embedding name
        new_name = f"{self.embedding_name}+{other.embedding_name}"
        
        return Embeddings(merged_df, new_name, new_columns)