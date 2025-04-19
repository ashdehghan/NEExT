# echo "starting global features experiments"
# for i in $(seq 1 10)
# do
#     python ego_abcdo.py \
#         --output_path results/ego_abcdo_17042025.parquet \
#         --comment p1\
#         --global_structural_feature_list all \
#         --global_feature_vector_length $i \
#         --embeddings_strategy feature_embeddings
# done

# echo "starting k_hop local features experiments"
# for i in $(seq 1 5)
# do
#     python ego_abcdo.py \
#         --output_path results/ego_abcdo_17042025.parquet \
#         --comment p2\
#         --local_structural_feature_list degree_centrality \
#         --egonet_k_hop $i \
#         --local_feature_vector_length 2 \
#         --embeddings_strategy structural_embeddings \
#         --embeddings_dimension 2
# done


echo "starting k_hop local features experiments"
for i in $(seq 1 5)
do
    python ego_abcdo.py \
        --output_path results/ego_abcdo_17042025.parquet \
        --comment p3\
        --local_structural_feature_list clustering_coefficient \
        --egonet_k_hop $i \
        --local_feature_vector_length 2 \
        --embeddings_strategy structural_embeddings \
        --embeddings_dimension 2
done