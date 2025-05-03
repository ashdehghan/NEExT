from typing import List, Literal

from pydantic import BaseModel, Field


class Param(BaseModel):
    comment: str
    egonet_target: str = "is_outlier"
    egonet_skip_features: List = Field(default_factory=list)
    filter_largest_component: bool = True
    show_progress: bool = False
    # alternative models that do not use NEExT
    n2v: bool = False
    random: bool = False

    ## khop dimension
    egonet_k_hop: int = 1
    # NEExT embedding configuration
    ## global structural features
    global_structural_feature_list: List[str] = []
    global_feature_vector_length: int = 1
    ## local structural features
    local_structural_feature_list: List[str] = []
    local_feature_vector_length: int = 1
    ## local node features
    local_node_features: List[str] = []
    ## embedding approach
    embeddings_dimension: int = 5
    embeddings_strategy: Literal[
        "structural_embeddings",
        "feature_embeddings",
        "separate_embeddings",
        "combined_embeddings",
        "structural_with_node_features",
        "feature_with_node_features",
        "only_egonet_node_features",
    ] = "feature_embeddings"
    # node positional embedding
    egonet_position: bool = False
    position_one_hot: bool = False
