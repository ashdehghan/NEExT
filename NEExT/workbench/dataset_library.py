"""Curated built-in dataset catalog for NEExT Workbench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .schemas import DatasetCatalogEntry

CATALOG_SOURCE = "AnomalyPoint/NEExT_datasets"
CSV_BUNDLE_BASE_URL = "https://raw.githubusercontent.com/AnomalyPoint/NEExT_datasets/refs/heads/main/real_world_networks/csv_format"
WORKBENCH_CATALOG_SOURCE = "NEExT Workbench curated examples"
HF_CATALOG_SOURCE = "anomalypoint/NEExT (Hugging Face)"
HF_RESOLVE_BASE_URL = "https://huggingface.co/datasets/anomalypoint/NEExT/resolve/main"
CATALOG_DATA_DIR = Path(__file__).with_name("catalog_data")


@dataclass(frozen=True)
class CatalogDataset:
    id: str
    name: str
    description: str
    domain: str
    files: dict[str, str]
    graph_count: int
    node_count: int
    edge_count: int
    source: str = CATALOG_SOURCE
    source_type: str = "neext_csv_bundle"
    source_graph_shape: str = "graph_collection"
    graph_shape: str = "graph_collection"
    node_attribute_columns: tuple[str, ...] = ()

    def to_public_entry(self) -> DatasetCatalogEntry:
        return DatasetCatalogEntry(
            id=self.id,
            name=self.name,
            description=self.description,
            source=self.source,
            domain=self.domain,
            source_type=self.source_type,
            source_graph_shape=self.source_graph_shape,
            graph_shape=self.graph_shape,
            graph_count=self.graph_count,
            node_count=self.node_count,
            edge_count=self.edge_count,
            has_graph_labels="graph_labels" in self.files,
            has_node_features="node_features" in self.files or "nodes" in self.files and bool(self.node_attribute_columns),
            has_edge_features="edge_features" in self.files,
            node_attribute_columns=list(self.node_attribute_columns),
        )


def _bundle_files(dataset: str, *, node_features: bool = False, edge_features: bool = False) -> dict[str, str]:
    base_url = f"{CSV_BUNDLE_BASE_URL}/{dataset}"
    files = {
        "edges": f"{base_url}/edges.csv",
        "node_graph_mapping": f"{base_url}/node_graph_mapping.csv",
        "graph_labels": f"{base_url}/graph_labels.csv",
    }
    if node_features:
        files["node_features"] = f"{base_url}/node_features.csv"
    if edge_features:
        files["edge_features"] = f"{base_url}/edge_features.csv"
    return files


def _single_graph_files(dataset: str) -> dict[str, str]:
    dataset_dir = CATALOG_DATA_DIR / dataset
    return {
        "nodes": str(dataset_dir / "nodes.csv"),
        "edges": str(dataset_dir / "edges.csv"),
    }


def _hf_single_graph_files(path: str) -> dict[str, str]:
    base_url = f"{HF_RESOLVE_BASE_URL}/{path}"
    return {
        "nodes": f"{base_url}/nodes.csv",
        "edges": f"{base_url}/edges.csv",
    }


def _hf_entry(
    entry_id: str, name: str, path: str, description: str, domain: str, node_count: int, edge_count: int, label_column: str
) -> CatalogDataset:
    return CatalogDataset(
        id=entry_id,
        name=name,
        description=description,
        domain=domain,
        files=_hf_single_graph_files(path),
        graph_count=1,
        node_count=node_count,
        edge_count=edge_count,
        source=HF_CATALOG_SOURCE,
        source_type="neext_single_graph_csv",
        source_graph_shape="single_graph",
        node_attribute_columns=(label_column,),
    )


_HF_SINGLE_GRAPH_DATASETS: tuple[tuple[str, str, str, str, str, int, int, str], ...] = (
    (
        "ACTOR",
        "Actor Co-occurrence",
        "actor/neext",
        "Actor category classification; 5 classes, label 'actor_class'.",
        "Web graphs",
        7600,
        26659,
        "actor_class",
    ),
    (
        "AIRPORTS_BRAZIL",
        "Air Traffic — Brazil",
        "airports/neext/brazil",
        "Airport activity-level classification; 4 classes, label 'activity_quartile'.",
        "Transportation",
        131,
        1003,
        "activity_quartile",
    ),
    (
        "AIRPORTS_EUROPE",
        "Air Traffic — Europe",
        "airports/neext/europe",
        "Airport activity-level classification; 4 classes, label 'activity_quartile'.",
        "Transportation",
        399,
        5993,
        "activity_quartile",
    ),
    (
        "AIRPORTS_USA",
        "Air Traffic — USA",
        "airports/neext/usa",
        "Airport activity-level classification; 4 classes, label 'activity_quartile'.",
        "Transportation",
        1190,
        13599,
        "activity_quartile",
    ),
    (
        "AMAZON_COMPUTERS",
        "Amazon Computers",
        "amazon-computers/neext",
        "Product category classification (co-purchase); 10 classes, label 'category'.",
        "Co-purchase",
        13752,
        245861,
        "category",
    ),
    (
        "AMAZON_PHOTO",
        "Amazon Photo",
        "amazon-photo/neext",
        "Product category classification (co-purchase); 8 classes, label 'category'.",
        "Co-purchase",
        7650,
        119081,
        "category",
    ),
    (
        "AMAZON_RATINGS",
        "Amazon Ratings",
        "amazon-ratings/neext",
        "Product rating-class prediction; 5 classes, label 'rating_class'.",
        "Co-purchase",
        24492,
        93050,
        "rating_class",
    ),
    (
        "BITCOIN_OTC",
        "Bitcoin OTC",
        "bitcoin-otc/neext",
        "Fraudulent-user detection (derived labels); 3 classes, label 'trust_label'.",
        "Finance",
        5881,
        21492,
        "trust_label",
    ),
    (
        "BLOGCATALOG",
        "BlogCatalog",
        "blogcatalog/neext",
        "Blogger interest-group classification; 38 classes, label 'group'.",
        "Social networks",
        7460,
        131034,
        "group",
    ),
    (
        "BOOKS",
        "Books (PyGOD)",
        "books/neext",
        "Outlier book detection (Amazon co-purchase); 2 classes, label 'is_outlier'.",
        "Fraud & anomaly",
        1418,
        3695,
        "is_outlier",
    ),
    (
        "CITESEER",
        "Citeseer",
        "citeseer/neext",
        "Paper topic classification (citation network); 6 classes, label 'subject'.",
        "Citation networks",
        3312,
        4536,
        "subject",
    ),
    (
        "COAUTHOR_CS",
        "Coauthor CS",
        "coauthor-cs/neext",
        "Research-field classification (co-authorship); 15 classes, label 'field'.",
        "Co-authorship",
        18333,
        81894,
        "field",
    ),
    (
        "COAUTHOR_PHYSICS",
        "Coauthor Physics",
        "coauthor-physics/neext",
        "Research-field classification (co-authorship); 5 classes, label 'field'.",
        "Co-authorship",
        34493,
        247962,
        "field",
    ),
    (
        "CORA",
        "Cora",
        "cora/neext",
        "Paper topic classification (citation network); 7 classes, label 'subject'.",
        "Citation networks",
        2708,
        5278,
        "subject",
    ),
    (
        "DEEZER_EUROPE",
        "Deezer Europe",
        "deezer-europe/neext",
        "User gender classification; 2 classes, label 'gender'.",
        "Social networks",
        28281,
        92752,
        "gender",
    ),
    (
        "DISNEY",
        "Disney (PyGOD)",
        "disney/neext",
        "Outlier movie detection (co-purchase); 2 classes, label 'is_outlier'.",
        "Fraud & anomaly",
        124,
        335,
        "is_outlier",
    ),
    (
        "EMAIL_EU_CORE",
        "Email-EU-core",
        "email-eu-core/neext",
        "Department classification from email traffic; 42 classes, label 'department'.",
        "Email networks",
        1005,
        16064,
        "department",
    ),
    ("ENRON", "Enron (PyGOD)", "enron/neext", "Email spam detection; 2 classes, label 'is_outlier'.", "Fraud & anomaly", 13533, 176987, "is_outlier"),
    (
        "FACEBOOK_PAGE_PAGE",
        "Facebook Page-Page",
        "facebook-page-page/neext",
        "Page category classification; 4 classes, label 'page_type'.",
        "Social networks",
        22470,
        170823,
        "page_type",
    ),
    (
        "GITHUB_DEVELOPERS",
        "GitHub Developers",
        "github-developers/neext",
        "Web vs ML developer classification; 2 classes, label 'ml_developer'.",
        "Social networks",
        37700,
        289003,
        "ml_developer",
    ),
    (
        "LASTFM_ASIA",
        "LastFM Asia",
        "lastfm-asia/neext",
        "User country classification; 18 classes, label 'country'.",
        "Social networks",
        7624,
        27806,
        "country",
    ),
    (
        "MINESWEEPER",
        "Minesweeper",
        "minesweeper/neext",
        "Mine prediction on a synthetic grid; 2 classes, label 'is_mine'.",
        "Synthetic",
        10000,
        39402,
        "is_mine",
    ),
    (
        "OGBN_ARXIV",
        "ogbn-arxiv",
        "ogbn-arxiv/neext",
        "arXiv subject-area classification; 40 classes, label 'arxiv_category'. Large download (~200 MB).",
        "Citation networks",
        169343,
        1157799,
        "arxiv_category",
    ),
    (
        "POLBLOGS",
        "Political Blogs",
        "polblogs/neext",
        "Political-leaning classification; 2 classes, label 'leaning'.",
        "Web graphs",
        1490,
        16715,
        "leaning",
    ),
    (
        "PUBMED",
        "Pubmed",
        "pubmed/neext",
        "Paper topic classification (citation network); 3 classes, label 'diabetes_type'.",
        "Citation networks",
        19717,
        44324,
        "diabetes_type",
    ),
    (
        "QUESTIONS",
        "Questions (Yandex Q)",
        "questions/neext",
        "User churn prediction; 2 classes, label 'is_active'.",
        "Fraud & anomaly",
        48921,
        153540,
        "is_active",
    ),
    (
        "REDDIT_GRAPHSAGE",
        "Reddit (GraphSAGE)",
        "reddit-graphsage/neext",
        "Subreddit classification of posts; 41 classes, label 'subreddit'. Large download (~170 MB).",
        "Social networks",
        232965,
        11606919,
        "subreddit",
    ),
    (
        "REDDIT_PYGOD",
        "Reddit (PyGOD)",
        "reddit-pygod/neext",
        "Banned-user detection; 2 classes, label 'is_outlier'.",
        "Fraud & anomaly",
        10984,
        78516,
        "is_outlier",
    ),
    (
        "ROMAN_EMPIRE",
        "Roman Empire",
        "roman-empire/neext",
        "Syntactic-role classification; 18 classes, label 'syntactic_role'.",
        "Text graphs",
        22662,
        32927,
        "syntactic_role",
    ),
    (
        "TOLOKERS",
        "Tolokers",
        "tolokers/neext",
        "Banned crowdworker prediction; 2 classes, label 'is_banned'.",
        "Fraud & anomaly",
        11758,
        519000,
        "is_banned",
    ),
    (
        "TWITCH_DE",
        "Twitch — DE",
        "twitch/neext/de",
        "Explicit-content streamer classification; 2 classes, label 'mature'.",
        "Social networks",
        9498,
        153138,
        "mature",
    ),
    (
        "TWITCH_ENGB",
        "Twitch — ENGB",
        "twitch/neext/engb",
        "Explicit-content streamer classification; 2 classes, label 'mature'.",
        "Social networks",
        7126,
        35324,
        "mature",
    ),
    (
        "TWITCH_ES",
        "Twitch — ES",
        "twitch/neext/es",
        "Explicit-content streamer classification; 2 classes, label 'mature'.",
        "Social networks",
        4648,
        59382,
        "mature",
    ),
    (
        "TWITCH_FR",
        "Twitch — FR",
        "twitch/neext/fr",
        "Explicit-content streamer classification; 2 classes, label 'mature'.",
        "Social networks",
        6549,
        112666,
        "mature",
    ),
    (
        "TWITCH_PTBR",
        "Twitch — PTBR",
        "twitch/neext/ptbr",
        "Explicit-content streamer classification; 2 classes, label 'mature'.",
        "Social networks",
        1912,
        31299,
        "mature",
    ),
    (
        "TWITCH_RU",
        "Twitch — RU",
        "twitch/neext/ru",
        "Explicit-content streamer classification; 2 classes, label 'mature'.",
        "Social networks",
        4385,
        37304,
        "mature",
    ),
    (
        "WEBKB_CORNELL",
        "WebKB — Cornell",
        "webkb/neext/cornell",
        "University web-page classification; 5 classes, label 'page_class'.",
        "Web graphs",
        183,
        277,
        "page_class",
    ),
    (
        "WEBKB_TEXAS",
        "WebKB — Texas",
        "webkb/neext/texas",
        "University web-page classification; 5 classes, label 'page_class'.",
        "Web graphs",
        183,
        279,
        "page_class",
    ),
    (
        "WEBKB_WISCONSIN",
        "WebKB — Wisconsin",
        "webkb/neext/wisconsin",
        "University web-page classification; 5 classes, label 'page_class'.",
        "Web graphs",
        251,
        450,
        "page_class",
    ),
    ("WEIBO", "Weibo (PyGOD)", "weibo/neext", "Social spam detection; 2 classes, label 'is_outlier'.", "Fraud & anomaly", 8405, 377271, "is_outlier"),
    ("WIKICS", "WikiCS", "wikics/neext", "CS article branch classification; 10 classes, label 'category'.", "Web graphs", 11701, 215603, "category"),
    (
        "WIKIPEDIA_ARTICLES_CHAMELEON",
        "Wikipedia — Chameleon",
        "wikipedia-articles/neext/chameleon",
        "Traffic-level classification (binned); 5 classes, label 'traffic_quintile'.",
        "Web graphs",
        2277,
        31371,
        "traffic_quintile",
    ),
    (
        "WIKIPEDIA_ARTICLES_CROCODILE",
        "Wikipedia — Crocodile",
        "wikipedia-articles/neext/crocodile",
        "Traffic-level classification (binned); 5 classes, label 'traffic_quintile'.",
        "Web graphs",
        11631,
        170773,
        "traffic_quintile",
    ),
    (
        "WIKIPEDIA_ARTICLES_SQUIRREL",
        "Wikipedia — Squirrel",
        "wikipedia-articles/neext/squirrel",
        "Traffic-level classification (binned); 5 classes, label 'traffic_quintile'.",
        "Web graphs",
        5201,
        198353,
        "traffic_quintile",
    ),
)

DATASET_CATALOG: tuple[CatalogDataset, ...] = (
    CatalogDataset(
        id="MUTAG",
        name="MUTAG",
        description="Molecule graph collection for graph classification workflows.",
        domain="Molecules",
        files=_bundle_files("MUTAG"),
        graph_count=188,
        node_count=3371,
        edge_count=7442,
    ),
    CatalogDataset(
        id="NCI1",
        name="NCI1",
        description="Chemical compound graph collection with graph-level labels.",
        domain="Molecules",
        files=_bundle_files("NCI1"),
        graph_count=4110,
        node_count=122747,
        edge_count=265506,
    ),
    CatalogDataset(
        id="BZR",
        name="BZR",
        description="Benzodiazepine receptor molecule graph collection with node features.",
        domain="Molecules",
        files=_bundle_files("BZR", node_features=True),
        graph_count=405,
        node_count=14479,
        edge_count=31070,
    ),
    CatalogDataset(
        id="PROTEINS",
        name="PROTEINS",
        description="Protein graph collection with graph-level labels and node features.",
        domain="Bioinformatics",
        files=_bundle_files("PROTEINS", node_features=True),
        graph_count=1113,
        node_count=43471,
        edge_count=162088,
    ),
    CatalogDataset(
        id="IMDB",
        name="IMDB",
        description="Movie collaboration graph collection with graph-level labels.",
        domain="Social networks",
        files=_bundle_files("IMDB"),
        graph_count=1000,
        node_count=19773,
        edge_count=386124,
    ),
    CatalogDataset(
        id="KARATE_CLUB",
        name="Zachary Karate Club",
        description="Single social network with node-level club labels for egonet graph classification workflows.",
        domain="Social networks",
        files=_single_graph_files("karate_club"),
        graph_count=1,
        node_count=34,
        edge_count=78,
        source=WORKBENCH_CATALOG_SOURCE,
        source_type="neext_single_graph_csv",
        source_graph_shape="single_graph",
        node_attribute_columns=("club",),
    ),
) + tuple(_hf_entry(*row) for row in _HF_SINGLE_GRAPH_DATASETS)


def list_catalog_entries() -> list[DatasetCatalogEntry]:
    return [dataset.to_public_entry() for dataset in DATASET_CATALOG]


def get_catalog_dataset(catalog_id: str) -> Optional[CatalogDataset]:
    normalized = catalog_id.strip().casefold()
    for dataset in DATASET_CATALOG:
        if dataset.id.casefold() == normalized:
            return dataset
    return None
