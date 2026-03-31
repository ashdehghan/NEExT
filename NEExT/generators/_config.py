"""Pydantic configuration models for the synthetic graph generation module."""

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


class GeneratorSpec(BaseModel):
    """Defines a single graph generator specification.

    Attributes:
        generator_type: Name of the generator (e.g., "erdos_renyi", "barabasi_albert").
            Open string, not restricted to built-ins — supports custom registered generators.
        params: Parameters passed to the generator function.
        label: Graph classification label assigned to generated graphs.
        n_range: Optional (min, max) tuple for variable-size graphs. When set,
            'n' is sampled uniformly from this range for each graph.
    """

    model_config = {"arbitrary_types_allowed": True}

    generator_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    label: Union[int, float] = 0
    n_range: Optional[Tuple[int, int]] = None

    @field_validator("n_range")
    @classmethod
    def validate_n_range(cls, v: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        if v is not None:
            if v[0] < 1 or v[1] < v[0]:
                raise ValueError(f"n_range must satisfy 1 <= min <= max, got {v}")
        return v


class CollectionSpec(BaseModel):
    """Specification for generating a mixed collection of graphs.

    Attributes:
        specs: List of (GeneratorSpec, count) pairs defining what to generate.
        seed: Master random seed for reproducibility.
        shuffle: Whether to shuffle the final list of graphs.
    """

    specs: List[Tuple[GeneratorSpec, int]]
    seed: Optional[int] = 42
    shuffle: bool = True


class AttributeConfig(BaseModel):
    """Configuration for synthetic node/edge attribute generation.

    Attributes:
        strategy: How to generate attributes.
        n_features: Number of features to generate.
        noise_level: Standard deviation of additive noise.
        custom_fn: User-provided callable for the "custom" strategy.
    """

    model_config = {"arbitrary_types_allowed": True}

    strategy: Literal[
        "random_uniform",
        "random_normal",
        "community_correlated",
        "label_informative",
        "degree_correlated",
        "custom",
    ] = "random_normal"
    n_features: int = Field(default=5, ge=1)
    noise_level: float = Field(default=0.1, ge=0.0)
    custom_fn: Optional[Callable] = None

    @field_validator("custom_fn")
    @classmethod
    def validate_custom_fn(cls, v, info):
        if info.data.get("strategy") == "custom" and v is None:
            raise ValueError("custom_fn must be provided when strategy is 'custom'")
        return v


class AnomalyConfig(BaseModel):
    """Configuration for anomaly injection.

    Attributes:
        anomaly_type: Type of anomaly to inject.
        fraction: Fraction of nodes to make anomalous.
        severity: Controls the intensity of the anomaly (higher = more extreme).
    """

    anomaly_type: Literal[
        "structural_hub",
        "structural_clique",
        "structural_bridge",
        "structural_rewire",
        "contextual_feature",
        "contextual_mixed",
    ] = "structural_hub"
    fraction: float = Field(default=0.05, gt=0.0, lt=1.0)
    severity: float = Field(default=3.0, gt=0.0)
