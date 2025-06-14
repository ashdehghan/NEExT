from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Params(BaseModel):
    comment: str
    egonet_target: str = "is_outlier"
    egonet_skip_features: List = Field(default_factory=list)
    filter_largest_component: bool = True
    show_progress: bool = False
    # alternative models that do not use NEExT
    n2v: bool = False
    random: bool = False

    ## egonet
    egonet_k_hop: int = 1
    
    ## global structural features
    global_structural_feature_list: List[str] = []
    global_feature_vector_length: int = 1
    
    ## local features
    local_node_features: List[str] = []
    local_structural_feature_list: List[str] = []
    local_feature_vector_length: int = 1
    
    ## embedding
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
    egonet_position: Optional[Literal["distance", "inv_distance", "inv_exp_distance"]] = None
    include_position: bool = False
    one_hot_encoding: bool = False
    position_encoding: bool = False
