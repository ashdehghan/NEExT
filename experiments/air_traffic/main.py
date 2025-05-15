from base_config import Config
from load_data import load_dataframes, generate_egonets, generate_embeddings
from modeling import run_experiments


def main():
    config = Config()

    # Load and prepare data
    edges_df, features_df, mapping_df = load_dataframes(config)

    # Extract features and embeddings
    graph_collection, egonet_collection, global_structural_features = generate_egonets(config, edges_df, features_df, mapping_df)
    embeddings, dataset = generate_embeddings(config, global_structural_features, graph_collection, egonet_collection)

    # Run model training + optimization
    run_experiments(config, dataset)


if __name__ == "__main__":
    main()
