import numpy as np
import pandas as pd
import pytest

from NEExT.embeddings import Embeddings, compute_embedding_similarity


def make_embeddings(rows: dict[str, list[float]]) -> Embeddings:
    columns = [f"emb_{index}" for index in range(len(next(iter(rows.values()))))]
    frame = pd.DataFrame([{"graph_id": graph_id, **dict(zip(columns, vector))} for graph_id, vector in rows.items()])
    return Embeddings(frame, "test", columns)


def similarity_by_graph(result: pd.DataFrame) -> dict[str, float]:
    return {str(row["graph_id"]): row["similarity"] for row in result.to_dict(orient="records")}


def test_identical_vector_reference_is_one():
    embeddings = make_embeddings({"0": [1.0, 0.0], "1": [1.0, 0.0], "2": [0.0, 1.0]})
    result = compute_embedding_similarity(embeddings, ["0"])
    values = similarity_by_graph(result)
    assert values["0"] == pytest.approx(1.0)
    assert values["1"] == pytest.approx(1.0)


def test_orthogonal_vector_is_half_and_opposite_is_zero():
    embeddings = make_embeddings({"0": [1.0, 0.0], "1": [0.0, 1.0], "2": [-1.0, 0.0]})
    result = compute_embedding_similarity(embeddings, ["0"])
    values = similarity_by_graph(result)
    assert values["1"] == pytest.approx(0.5)
    assert values["2"] == pytest.approx(0.0)


def test_centroid_of_multiple_references():
    embeddings = make_embeddings({"0": [1.0, 0.0], "1": [0.0, 1.0], "2": [1.0, 1.0], "3": [-1.0, -1.0]})
    result = compute_embedding_similarity(embeddings, ["0", "1"])
    values = similarity_by_graph(result)
    # Centroid of (1,0) and (0,1) is (0.5, 0.5); (1,1) is parallel to it.
    assert values["2"] == pytest.approx(1.0)
    assert values["3"] == pytest.approx(0.0)
    assert values["0"] == pytest.approx((np.cos(np.pi / 4) + 1) / 2)


def test_integer_reference_ids_match_string_graph_ids():
    embeddings = make_embeddings({"7": [1.0, 0.0], "8": [0.0, 1.0]})
    result = compute_embedding_similarity(embeddings, [7])
    assert similarity_by_graph(result)["7"] == pytest.approx(1.0)


def test_output_columns_and_ranges():
    embeddings = make_embeddings({"0": [1.0, 2.0], "1": [-3.0, 0.5], "2": [0.0, 0.0]})
    result = compute_embedding_similarity(embeddings, ["0"])
    assert list(result.columns) == ["graph_id", "cosine_similarity", "similarity"]
    assert result["cosine_similarity"].between(-1.0, 1.0).all()
    assert result["similarity"].between(0.0, 1.0).all()
    # Zero-norm vector maps to cosine 0.0 -> similarity 0.5.
    assert similarity_by_graph(result)["2"] == pytest.approx(0.5)


def test_empty_reference_rejected():
    embeddings = make_embeddings({"0": [1.0, 0.0]})
    with pytest.raises(ValueError, match="at least one"):
        compute_embedding_similarity(embeddings, [])


def test_missing_reference_ids_listed():
    embeddings = make_embeddings({"0": [1.0, 0.0], "1": [0.0, 1.0]})
    with pytest.raises(ValueError, match=r"\['9'\]"):
        compute_embedding_similarity(embeddings, ["0", "9"])
