import copy
import time
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from NEExT.collections import GraphCollection
from NEExT.embeddings import Embeddings, GraphEmbeddings
from NEExT.features import Features
from NEExT.ml_models import MLModels
from NEExT.ml_models.ml_models import XGBOOST_AVAILABLE

if XGBOOST_AVAILABLE:
    from xgboost import XGBClassifier, XGBRegressor


class FeatureImportanceConfig(BaseModel):
    """Configuration for feature importance analysis"""

    algorithm: Literal["supervised_greedy", "supervised_fast", "unsupervised"]
    embedding_algorithm: str = "approx_wasserstein"
    model_name: Literal["xgboost", "random_forest"] = "random_forest"
    random_state: int = 42
    sample_size: int = 5
    n_jobs: int = -1
    parallel_backend: Literal["process", "thread"] = "process"
    holdout_size: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    unsupervised_nodes_per_graph: int = Field(default=50, ge=2)
    unsupervised_n_graphs: int = Field(default=200, ge=2)
    unsupervised_n_pairs: int = Field(default=1000, ge=2)


class FeatureImportance:
    """
    A class for analyzing feature importance in graph embeddings.

    This class provides methods to determine the importance of node features
    based on their predictive power in both supervised and unsupervised settings.

    When ``holdout_size`` is set, a stratified fraction of the graphs is held out
    before any selection happens. Feature selection (embeddings, model training and
    scoring) then only ever sees the remaining training graphs, and the returned
    ``heldout_performance`` column scores each prefix of the selected feature order
    on the held-out graphs, which are never used for selection. The embedding step
    is label-free and is computed over all graphs, as in the rest of NEExT.

    Attributes:
        graph_collection (GraphCollection): Collection of graphs to analyze
        features (Features): Features object containing node features
        config (FeatureImportanceConfig): Configuration for importance analysis
        train_graph_ids (List): Graph IDs used for selection
        heldout_graph_ids (List): Graph IDs held out for evaluation (empty without holdout)

    Example:
        >>> importance = FeatureImportance(
        ...     graph_collection=collection,
        ...     features=features,
        ...     algorithm="supervised_greedy",
        ...     embedding_algorithm="approx_wasserstein",
        ...     holdout_size=0.2,
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
        n_iterations: int = 5,  # Keep for backward compatibility
        n_jobs: int = -1,
        parallel_backend: str = "process",
        holdout_size: Optional[float] = None,
        unsupervised_nodes_per_graph: int = 50,
        unsupervised_n_graphs: int = 200,
        unsupervised_n_pairs: int = 1000,
    ):
        """Initialize the FeatureImportance analyzer."""
        self.config = FeatureImportanceConfig(
            algorithm=algorithm,
            embedding_algorithm=embedding_algorithm,
            random_state=random_state,
            sample_size=n_iterations,  # Use n_iterations as sample_size
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            holdout_size=holdout_size,
            unsupervised_nodes_per_graph=unsupervised_nodes_per_graph,
            unsupervised_n_graphs=unsupervised_n_graphs,
            unsupervised_n_pairs=unsupervised_n_pairs,
        )
        self.graph_collection = graph_collection
        self.features = features

        # Define available algorithms
        self.available_algorithms = {
            "supervised_greedy": self._supervised_greedy,
            "supervised_fast": self._supervised_fast,
            "unsupervised": self._unsupervised,
        }

        self.train_graph_ids, self.heldout_graph_ids = self._split_graph_ids()
        self.train_collection, self.train_features = self._restrict_to_graphs(self.train_graph_ids)

    def compute(self) -> pd.DataFrame:
        """
        Compute feature importance based on the configured algorithm.

        Returns:
            pd.DataFrame: DataFrame containing feature importance results. With
                ``holdout_size`` set, it also contains ``heldout_performance``: the
                held-out score of the embedding built from the first ``rank + 1``
                features of the returned order.
        """
        if self.config.algorithm not in self.available_algorithms:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        start_time = time.time()
        results = self.available_algorithms[self.config.algorithm]()

        if self.heldout_graph_ids:
            results["heldout_performance"] = self.evaluate_feature_order(results["feature_name"].tolist())

        total_time = time.time() - start_time

        # Add total computation time to results
        results["total_time"] = total_time

        return results

    def _is_classifier(self) -> bool:
        return isinstance(self.graph_collection.graphs[0].graph_label, (int, np.integer))

    def _split_graph_ids(self) -> Tuple[List, List]:
        """Split graph IDs into selection (train) and held-out sets."""
        graph_ids = [graph.graph_id for graph in self.graph_collection.graphs]
        if self.config.holdout_size is None:
            return graph_ids, []

        labels = [graph.graph_label for graph in self.graph_collection.graphs]
        if any(label is None for label in labels):
            if self.config.algorithm != "unsupervised":
                raise ValueError("Graph collection must have labels for all graphs to use a supervised holdout")
            stratify = None
        else:
            stratify = labels if self._is_classifier() else None
        train_ids, heldout_ids = train_test_split(
            graph_ids,
            test_size=self.config.holdout_size,
            random_state=self.config.random_state,
            stratify=stratify,
            shuffle=True,
        )
        return list(train_ids), list(heldout_ids)

    def _restrict_to_graphs(self, graph_ids: List) -> Tuple[GraphCollection, Features]:
        """Return a collection and features view containing only ``graph_ids``."""
        if not self.heldout_graph_ids:
            return self.graph_collection, self.features
        keep = set(graph_ids)
        collection = copy.copy(self.graph_collection)
        collection.graphs = [graph for graph in self.graph_collection.graphs if graph.graph_id in keep]
        features_df = self.features.features_df[self.features.features_df["graph_id"].isin(keep)].reset_index(drop=True)
        return collection, Features(features_df, list(self.features.feature_columns), self.features.feature_groups)

    def evaluate_feature_order(self, ordered_features: List[str]) -> List[float]:
        """
        Score each prefix of ``ordered_features`` on the held-out graphs.

        Names may be feature groups (``Features.feature_groups``), which contribute all their columns.
        For prefix j, an embedding of the prefix's column count is built from the first j names,
        a model is fit on the training graphs only, and accuracy (classifier) or RMSE
        (regressor) is measured on the held-out graphs.

        Returns:
            List[float]: One held-out score per prefix length.
        """
        if not self.heldout_graph_ids:
            raise ValueError("evaluate_feature_order requires holdout_size to be set")

        labels = {graph.graph_id: graph.graph_label for graph in self.graph_collection.graphs}
        train_ids = set(self.train_graph_ids)
        heldout_ids = set(self.heldout_graph_ids)
        is_classifier = self._is_classifier()
        encoder = LabelEncoder().fit([labels[gid] for gid in self.train_graph_ids]) if is_classifier else None

        units = self.features.selection_units()
        scores = []
        for j in range(1, len(ordered_features) + 1):
            # A name is a feature group (all its columns) or a single column.
            prefix = [col for name in ordered_features[:j] for col in units.get(name, [name])]
            embeddings = GraphEmbeddings(
                graph_collection=self.graph_collection,
                features=self.features,
                embedding_algorithm=self.config.embedding_algorithm,
                embedding_dimension=len(prefix),
                feature_columns=prefix,
                random_state=self.config.random_state,
            ).compute()
            emb_df = embeddings.embeddings_df
            emb_cols = embeddings.embedding_columns
            train_df = emb_df[emb_df["graph_id"].isin(train_ids)]
            test_df = emb_df[emb_df["graph_id"].isin(heldout_ids)]
            y_train = train_df["graph_id"].map(labels).to_numpy()
            y_test = test_df["graph_id"].map(labels).to_numpy()

            if is_classifier:
                y_train = encoder.transform(y_train)
                y_test = encoder.transform(y_test)
                if XGBOOST_AVAILABLE:
                    model = XGBClassifier(random_state=self.config.random_state, n_jobs=1)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state, n_jobs=1)
                model.fit(train_df[emb_cols].to_numpy(), y_train)
                scores.append(float(accuracy_score(y_test, model.predict(test_df[emb_cols].to_numpy()))))
            else:
                if XGBOOST_AVAILABLE:
                    model = XGBRegressor(random_state=self.config.random_state, n_jobs=1)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state, n_jobs=1)
                model.fit(train_df[emb_cols].to_numpy(), y_train)
                predictions = model.predict(test_df[emb_cols].to_numpy())
                scores.append(float(np.sqrt(mean_squared_error(y_test, predictions))))
        return scores

    def _supervised_greedy(self) -> pd.DataFrame:
        """
        Compute feature importance using supervised greedy algorithm.

        This method determines feature importance by iteratively selecting features
        that maximize model performance when combined with previously selected features.
        Only the training graphs are used (all graphs when no holdout is configured).

        Returns:
            pd.DataFrame: Results containing:
                - feature_name: Name of the feature
                - avg_performance: Average model performance at each step
                - embedding_algorithm: Name of embedding algorithm used
                - total_time: Total computation time in seconds
        """
        # Selection units: a feature group (e.g. all columns of one node embedding) enters as a block,
        # every other column on its own.
        units = self.train_features.selection_units()
        available_features = list(units)
        selected_features = []
        selected_columns: List[str] = []
        performance_scores = []

        # Create progress bar for the outer loop
        pbar = tqdm(total=len(available_features), desc="Selecting features", position=0, leave=True)

        while available_features:
            best_score = float("-inf")
            best_feature = None

            # Create progress bar for the inner loop
            inner_pbar = tqdm(available_features, desc=f"Testing features (selected: {len(selected_features)})", position=1, leave=False)

            # Try each remaining feature
            for feature in inner_pbar:
                current_features = selected_columns + units[feature]
                inner_pbar.set_postfix({"testing": feature}, refresh=True)

                # Create embeddings using current feature set
                embeddings = GraphEmbeddings(
                    graph_collection=self.train_collection,
                    features=self.train_features,
                    embedding_algorithm=self.config.embedding_algorithm,
                    embedding_dimension=len(current_features),
                    feature_columns=current_features,
                    random_state=self.config.random_state,
                ).compute()

                # Train and evaluate model
                ml_model = MLModels(
                    graph_collection=self.train_collection,
                    embeddings=embeddings,
                    model_type="classifier" if self._is_classifier() else "regressor",
                    random_state=self.config.random_state,
                    sample_size=self.config.sample_size,  # Use sample_size instead of n_iterations
                    n_jobs=self.config.n_jobs,
                    parallel_backend=self.config.parallel_backend,
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
            selected_columns.extend(units[best_feature])
            available_features.remove(best_feature)
            performance_scores.append(abs(best_score))  # Convert back to positive RMSE if needed

            # Update outer progress bar
            pbar.update(1)
            pbar.set_postfix({"best_feature": best_feature, "score": abs(best_score)}, refresh=True)

        # Close outer progress bar
        pbar.close()

        # Create results DataFrame
        results_df = pd.DataFrame(
            {
                "feature_name": selected_features,
                "n_columns": [len(units[name]) for name in selected_features],
                "avg_performance": performance_scores,
                "embedding_algorithm": self.config.embedding_algorithm,
            }
        )

        return results_df

    def _supervised_fast(self) -> pd.DataFrame:
        """
        Compute feature importance using the supervised fast algorithm.

        Each node feature is turned into a 1D graph embedding, the per-feature embeddings are merged
        into a single matrix (one column per feature), and a Random Forest is trained on that matrix.
        The forest's ``feature_importances_`` give a genuine, differentiated importance weight per
        node feature, averaged over ``sample_size`` iterations. Higher weight = more predictive.
        Only the training graphs are used (all graphs when no holdout is configured).

        Returns:
            pd.DataFrame: Results containing:
                - feature_name: Name of the node feature
                - importance: Mean Random Forest importance weight (0..1, sums to ~1 across features)
                - embedding_algorithm: Name of embedding algorithm used
                - total_time: Total computation time in seconds
        """
        start_time = time.time()
        feature_embeddings = []

        # Generate 1D embeddings for each feature
        for feature in tqdm(self.features.feature_columns, desc="Computing per-feature embeddings"):
            embeddings = GraphEmbeddings(
                graph_collection=self.train_collection,
                features=self.train_features,
                embedding_algorithm=self.config.embedding_algorithm,
                embedding_dimension=1,
                feature_columns=[feature],
                random_state=self.config.random_state,
            ).compute()

            embedding_df = embeddings.embeddings_df.copy()
            embedding_df.rename(columns={"emb_0": feature}, inplace=True)
            feature_embeddings.append(embedding_df)

        # Merge all per-feature embeddings into one matrix (one column per node feature)
        merged_df = feature_embeddings[0]
        for df in feature_embeddings[1:]:
            merged_df = pd.merge(merged_df, df, on="graph_id", how="outer")

        embeddings = Embeddings(
            embeddings_df=merged_df, embedding_name=self.config.embedding_algorithm, embedding_columns=self.features.feature_columns
        )

        # Train a Random Forest and read its real per-feature importance weights
        ml_model = MLModels(
            graph_collection=self.train_collection,
            embeddings=embeddings,
            model_type="classifier" if self._is_classifier() else "regressor",
            model_name="random_forest",
            compute_feature_importance=True,
            sample_size=self.config.sample_size,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            parallel_backend=self.config.parallel_backend,
        )

        results = ml_model.compute()
        importance_df = results["feature_importance"]  # index=feature, column 'importance', desc

        total_time = time.time() - start_time

        results_df = pd.DataFrame(
            {
                "feature_name": importance_df.index.tolist(),
                "importance": importance_df["importance"].tolist(),
                "embedding_algorithm": self.config.embedding_algorithm,
                "total_time": total_time,
            }
        )

        return results_df

    def _unsupervised(self) -> pd.DataFrame:
        """
        Order features without labels by how much their distributions vary across graphs.

        1. Sample up to ``unsupervised_n_graphs`` graphs and up to ``unsupervised_nodes_per_graph``
           nodes from each (training graphs only when a holdout is configured).
        2. Standardise each feature over the sampled nodes so scores are comparable across features.
        3. For a fixed random set of ``unsupervised_n_pairs`` graph pairs, compute the 1D Wasserstein
           distance between the two graphs' node-feature distributions, per feature. This gives each
           feature a vector of pairwise distances.
        4. The seed feature is the one whose distance vector has the largest variance.
        5. Each next feature maximises ``variance * (1 - max |corr|)``, where the correlation is
           between its distance vector and those of the already selected features, so variable but
           non-redundant features come first.

        Returns:
            pd.DataFrame: Results containing:
                - feature_name: Name of the feature, in selection order
                - variance: Variance of the feature's pairwise Wasserstein distances
                - max_abs_correlation: Largest |correlation| with an earlier-selected feature
                - score: Selection score at the step the feature was picked
                - embedding_algorithm: Name of embedding algorithm (used for held-out scoring)
        """
        rng = np.random.default_rng(self.config.random_state)
        feature_cols = list(self.features.feature_columns)
        features_df = self.train_features.features_df

        graph_ids = np.array(sorted(features_df["graph_id"].unique()))
        if len(graph_ids) < 2:
            raise ValueError("Unsupervised feature selection needs at least two graphs")
        if len(graph_ids) > self.config.unsupervised_n_graphs:
            graph_ids = rng.choice(graph_ids, size=self.config.unsupervised_n_graphs, replace=False)

        grouped = features_df[features_df["graph_id"].isin(set(graph_ids.tolist()))].groupby("graph_id")
        samples = {}
        for graph_id, group in grouped:
            values = group[feature_cols].to_numpy(dtype=float)
            if len(values) > self.config.unsupervised_nodes_per_graph:
                values = values[rng.choice(len(values), size=self.config.unsupervised_nodes_per_graph, replace=False)]
            samples[graph_id] = values
        sampled_ids = list(samples.keys())

        stacked = np.vstack([samples[gid] for gid in sampled_ids])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std[std == 0] = 1.0
        for gid in sampled_ids:
            samples[gid] = (samples[gid] - mean) / std

        n_graphs = len(sampled_ids)
        all_pairs = n_graphs * (n_graphs - 1) // 2
        if all_pairs <= self.config.unsupervised_n_pairs:
            pairs = [(a, b) for a in range(n_graphs) for b in range(a + 1, n_graphs)]
        else:
            pairs = set()
            while len(pairs) < self.config.unsupervised_n_pairs:
                a, b = rng.choice(n_graphs, size=2, replace=False)
                pairs.add((min(a, b), max(a, b)))
            pairs = sorted(pairs)

        distances = np.zeros((len(feature_cols), len(pairs)))
        for p, (a, b) in enumerate(pairs):
            left = samples[sampled_ids[a]]
            right = samples[sampled_ids[b]]
            for f in range(len(feature_cols)):
                distances[f, p] = wasserstein_distance(left[:, f], right[:, f])

        variances = distances.var(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = np.nan_to_num(np.corrcoef(distances), nan=0.0)
        if correlations.ndim == 0:
            correlations = np.array([[1.0]])

        remaining = list(range(len(feature_cols)))
        order, chosen_variance, chosen_corr, chosen_score = [], [], [], []
        while remaining:
            best_index, best_score, best_corr = None, float("-inf"), 0.0
            for f in remaining:
                max_corr = float(np.max(np.abs(correlations[f, order]))) if order else 0.0
                score = float(variances[f] * (1.0 - max_corr))
                if score > best_score:
                    best_index, best_score, best_corr = f, score, max_corr
            order.append(best_index)
            remaining.remove(best_index)
            chosen_variance.append(float(variances[best_index]))
            chosen_corr.append(best_corr)
            chosen_score.append(best_score)

        return pd.DataFrame(
            {
                "feature_name": [feature_cols[f] for f in order],
                "variance": chosen_variance,
                "max_abs_correlation": chosen_corr,
                "score": chosen_score,
                "embedding_algorithm": self.config.embedding_algorithm,
            }
        )
