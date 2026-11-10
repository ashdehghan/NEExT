from typing import Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


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
        feature_groups (Dict[str, List[str]]): Columns that belong together and are selected as one unit
            (e.g. the d columns of a node embedding). Columns not named in any group are their own group.

    Example:
        >>> features = Features(df, ["page_rank", "degree_centrality"])
        >>> features.normalize(type="StandardScaler")
        >>> merged = features1 + features2
    """

    def __init__(self, features_df: pd.DataFrame, feature_columns: List[str], feature_groups: Optional[Dict[str, List[str]]] = None):
        """Initialize the Features object."""
        self.features_df = features_df
        self.feature_columns = feature_columns
        self.feature_groups = {name: list(cols) for name, cols in (feature_groups or {}).items()}

        # Validate that all feature columns exist in DataFrame
        missing_cols = [col for col in feature_columns if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Features {missing_cols} not found in DataFrame")
        unknown = [col for cols in self.feature_groups.values() for col in cols if col not in feature_columns]
        if unknown:
            raise ValueError(f"feature_groups name columns that are not feature columns: {unknown}")

    def selection_units(self, columns: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """The units feature selection adds or removes at once: each group (restricted to `columns`, default all
        feature columns) and every remaining column on its own, in feature-column order."""
        columns = list(self.feature_columns if columns is None else columns)
        wanted = set(columns)
        group_of = {col: name for name, cols in self.feature_groups.items() for col in cols}
        units: Dict[str, List[str]] = {}
        for col in columns:
            name = group_of.get(col, col)
            units.setdefault(name, [])
            if col in wanted:
                units[name].append(col)
        return units

    def normalize(self, type: str = "StandardScaler") -> None:
        """
        Normalize features using the specified scaler.

        Args:
            type: Type of scaler to use ("StandardScaler", "MinMaxScaler", "RobustScaler")
        """
        config = FeaturesConfig(scaler_type=type)

        if not self.features_df.empty:
            # Get feature columns (exclude node_id and graph_id)
            feature_cols = [col for col in self.features_df.columns if col not in ["node_id", "graph_id"]]

            if feature_cols:
                # Initialize scaler
                scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}
                scaler = scalers[config.scaler_type]

                # Fit and transform the feature columns
                self.features_df[feature_cols] = scaler.fit_transform(self.features_df[feature_cols])

    def __add__(self, other: "Features") -> "Features":
        """
        Merge two Features objects.

        Args:
            other: Another Features object to merge with

        Returns:
            Features: New Features object containing merged data
        """
        if not isinstance(other, Features):
            raise TypeError("Can only add Features objects together")

        # Detect overlapping feature columns and use suffixes if needed
        overlap = set(self.feature_columns) & set(other.feature_columns)
        if overlap:
            suffixes = ("_left", "_right")
        else:
            suffixes = ("", "")

        # Merge DataFrames on node_id and graph_id
        merged_df = pd.merge(
            self.features_df,
            other.features_df,
            on=["node_id", "graph_id"],
            how="outer",
            suffixes=suffixes,
        )

        # Derive feature columns from actual DataFrame column order
        merged_columns = [col for col in merged_df.columns if col not in ("node_id", "graph_id")]

        groups = dict(self.feature_groups)
        if not overlap:
            groups.update(other.feature_groups)
        return Features(merged_df, merged_columns, groups if not overlap else None)
