from collections.abc import Sequence
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from NEExT.embeddings.embeddings import Embeddings


def compute_embedding_similarity(
    embeddings: Embeddings,
    reference_graph_ids: Sequence[Union[str, int]],
) -> pd.DataFrame:
    """
    Compute cosine similarity of every embedding row to the centroid of a reference set.

    The centroid is the mean of the reference rows' embedding vectors; a single
    reference id is the trivial one-row case. Raw cosine similarity is rescaled to
    [0, 1] with the fixed affine map (cosine + 1) / 2, so 1.0 means identical
    direction, 0.5 orthogonal, and 0.0 opposite. Zero-norm vectors get cosine 0.0
    (similarity 0.5), matching scikit-learn's convention.

    Args:
        embeddings: Embeddings object whose embeddings_df holds one row per graph_id
        reference_graph_ids: graph_id values (as found in embeddings_df) whose rows
            form the reference set

    Returns:
        pd.DataFrame with columns:
            graph_id: as in embeddings.embeddings_df
            cosine_similarity: raw cosine in [-1, 1]
            similarity: (cosine_similarity + 1) / 2, in [0, 1]
    """
    if len(reference_graph_ids) == 0:
        raise ValueError("reference_graph_ids must contain at least one graph id")

    embeddings_df = embeddings.embeddings_df
    embedding_columns = [column for column in embeddings.embedding_columns if pd.api.types.is_numeric_dtype(embeddings_df[column])]
    if not embedding_columns:
        raise ValueError("Embeddings contain no numeric embedding columns")

    graph_ids = embeddings_df["graph_id"].astype(str)
    reference_ids = {str(reference_id) for reference_id in reference_graph_ids}
    reference_mask = graph_ids.isin(reference_ids)
    missing_ids = sorted(reference_ids - set(graph_ids[reference_mask]))
    if missing_ids:
        raise ValueError(f"Reference graph ids not found in embeddings: {missing_ids}")

    vectors = embeddings_df[embedding_columns].to_numpy(dtype=float)
    centroid = vectors[reference_mask.to_numpy()].mean(axis=0)
    cosine = cosine_similarity(vectors, centroid.reshape(1, -1)).reshape(-1)
    cosine = np.clip(cosine, -1.0, 1.0)

    return pd.DataFrame(
        {
            "graph_id": embeddings_df["graph_id"],
            "cosine_similarity": cosine,
            "similarity": (cosine + 1.0) / 2.0,
        }
    )
