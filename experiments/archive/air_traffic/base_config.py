class Config:
    def __init__(self):
        # Graph
        self.dataset = "usa"
        if self.dataset == "brazil":
            self.edge_list_url = "https://raw.githubusercontent.com/node-embedding/ffstruc2vec/refs/heads/main/graph/brazil-airports.edgelist"
            self.label_url = "https://raw.githubusercontent.com/node-embedding/ffstruc2vec/refs/heads/main/graph/labels-brazil-airports.txt"
        elif self.dataset == "europe":
            self.edge_list_url = "https://raw.githubusercontent.com/node-embedding/ffstruc2vec/refs/heads/main/graph/europe-airports.edgelist"
            self.label_url = "https://raw.githubusercontent.com/node-embedding/ffstruc2vec/refs/heads/main/graph/labels-europe-airports.txt"
        elif self.dataset == "usa":
            self.edge_list_url = "https://raw.githubusercontent.com/node-embedding/ffstruc2vec/refs/heads/main/graph/usa-airports.edgelist"
            self.label_url = "https://raw.githubusercontent.com/node-embedding/ffstruc2vec/refs/heads/main/graph/labels-usa-airports.txt"
        self.k_hop = 1
        self.skip_features = []
        self.egonet_feature_target = "label"
        self.graph_type = "igraph"
        self.filter_largest_component = False

        # Features
        self.global_structural_features = {
            "feature_list": ["all"],
            "feature_vector_length": 3,
        }
        self.egonet_structural_features = {
            "feature_list": ["all"],
            "feature_vector_length": self.k_hop,
        }
        self.egonet_node_features = {
            "feature_list": [],
        }

        # Embeddings
        self.embeddings = {
            "embeddings_dimension": 10,
            "strategy": "separate_embeddings",
            # "strategy": "only_egonet_node_features",
        }

        # Modeling
        self.modeling = {
            "n_trials": 10,
            "n_experiments": 10,
            "test_size": 0.2,
            "random_state": 42,
            "eval_metric": "multi_logloss",
        }
