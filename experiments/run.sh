# global features experiment
for i in $(seq 1 6)
do
    python ego_abcdo.py \
        --output_path results/ego_abcdo_17042025.parquet \
        --comment p1\
        --global_structural_feature_list all \
        --global_feature_vector_length $i \
        --embeddings_strategy feature_embeddings
done


# TIMING TEST

features=("page_rank" "degree_centrality" "closeness_centrality" "betweenness_centrality" "eigenvector_centrality" "clustering_coefficient" "local_efficiency" "lsme" "load_centrality" "basic_expansion")
for feature in "${features[@]}"
do
  for i in {1..5}
  do
    for (( k=1; k<=i; k++ ))
    do
      python ego_abcdo.py \
        --output_path results/ego_abcdo_17042025.parquet \
        --comment timing_experiment_2\
        --local_structural_feature_list $feature \
        --egonet_k_hop $i \
        --local_feature_vector_length $k \
        --embeddings_strategy structural_embeddings \
        --embeddings_dimension $k
    done
  done
done

# LOCAL MODEL TEST

python ego_abcdo.py \
    --output_path results/ego_abcdo_17042025.parquet \
    --comment local_model\
    --local_structural_feature_list betweenness_centrality closeness_centrality clustering_coefficient degree_centrality eigenvector_centrality load_centrality local_efficiency lsme \
    --egonet_k_hop 1 \
    --local_feature_vector_length 1 \
    --embeddings_strategy structural_embeddings

python ego_abcdo.py \
    --output_path results/ego_abcdo_17042025.parquet \
    --comment local_model\
    --local_structural_feature_list closeness_centrality degree_centrality eigenvector_centrality lsme \
    --egonet_k_hop 2 \
    --local_feature_vector_length 2 \
    --embeddings_strategy structural_embeddings

# EMBEDDING COMBINATIONS TEST

python ego_abcdo.py \
    --output_path results/ego_abcdo_17042025.parquet \
    --comment global_and_local\
    --global_structural_feature_list all \
    --global_feature_vector_length 4 \
    --local_structural_feature_list betweenness_centrality closeness_centrality clustering_coefficient degree_centrality eigenvector_centrality load_centrality local_efficiency lsme \
    --egonet_k_hop 1 \
    --local_feature_vector_length 1 \
    --embeddings_strategy combined_embeddings

python ego_abcdo.py \
    --output_path results/ego_abcdo_17042025.parquet \
    --comment global_and_local\
    --global_structural_feature_list all \
    --global_feature_vector_length 4 \
    --local_structural_feature_list closeness_centrality degree_centrality eigenvector_centrality lsme \
    --egonet_k_hop 2 \
    --local_feature_vector_length 2 \
    --embeddings_strategy combined_embeddings

python ego_abcdo.py \
    --output_path results/ego_abcdo_17042025.parquet \
    --comment global_and_local\
    --global_structural_feature_list all \
    --global_feature_vector_length 4 \
    --local_structural_feature_list betweenness_centrality closeness_centrality clustering_coefficient degree_centrality eigenvector_centrality load_centrality local_efficiency lsme \
    --egonet_k_hop 1 \
    --local_feature_vector_length 1 \
    --embeddings_strategy separate_embeddings

python ego_abcdo.py \
    --output_path results/ego_abcdo_17042025.parquet \
    --comment global_and_local\
    --global_structural_feature_list all \
    --global_feature_vector_length 4 \
    --local_structural_feature_list closeness_centrality degree_centrality eigenvector_centrality lsme \
    --egonet_k_hop 2 \
    --local_feature_vector_length 2 \
    --embeddings_strategy separate_embeddings

python ego_abcdo.py \
    --output_path results/ego_abcdo_30042025.parquet \
    --comment global_and_local \
    --global_structural_feature_list all \
    --global_feature_vector_length 4 \
    --local_structural_feature_list betweenness_centrality closeness_centrality clustering_coefficient degree_centrality eigenvector_centrality load_centrality local_efficiency lsme \
    --egonet_k_hop 1 \
    --local_feature_vector_length 1 \
    --embeddings_strategy separate_embeddings \
    --egonet_position




python ego_abcdo.py \
    --output_path results/ego_abcdo_30042025.parquet \
    --comment 0hop \
    --global_structural_feature_list all \
    --global_feature_vector_length 4 \
    --egonet_k_hop 0 \
    --local_feature_vector_length 1 \
    --embeddings_strategy feature_embeddings \
