# NEExT

### Network Embedding Exploration Tool

NEExT is a tool for exploring and building graph embeddings. This tool allows for:
* Cleansing and standardizing a collection of graph data.
* Creating node and structural features for nodes in the graph collection.
* Creating embeddings for graphs.

### Installation Process
NEExT uses Python 3.x (currently tested using Python 3.11).
You can install NEExT using the following:
```console
pip install NEExT
```

### Graph Data Format
You can use a few different data formats to upload data into NEExT. Currently, it allows for:
* CSV files
* NetworkX Objects (comming soon)
See below for examples of using different data formats.

#### Using CSV Files
Data can be categorized into the following groups:
* Edge File (captures which nodes are connected to which nodes)
* Node Graph Mapping (captures which belongs to which graph)
* Graph Label Mapping [optional] (captures labels for each graph)
* Node Features [optional] (captures the features for each node)

Below we show example of how each of the above files should be formatted:

##### Edge File:
|node_a|node_b|
|---|---|
|1|2|
|3|2|
|.|.|

#### Node Graph Mapping:
|node_id|graph_id|
|---|---|
|0|1|
|1|1|
|2|1|
|3|2|
|4|2|
|.|.|

#### Graph Label Mapping:
|graph_id|graph_label|
|---|---|
|0|0|
|1|0|
|2|1|
|3|0|
|4|1|
|.|.|

#### Node Features:
|node_id|node_feat_0|node_feat_1|...|
|---|---|---|---|
|0|0.34| 3.2| .|
|1|0.1| 2.9| .|
|2|1.9| 1.3| .|
|3|0.0| 2.2| .|
|4|11.2| 12.3| .|
|.|.| .| .|

Note that NEExT can not handle non-numerical features. Some feature engineering on the node features must be done by the end-user.
Data standardization, however, will be done.






# NEExT Tutorial [Getting Started]

In this notebook, we showcase how to use NEExT to analyze graph embeddings.


```python
from NEExT.NEExT import NEExT
```

The following are link to some graph data, which we will use in this tutorial.
Note that we have Graph Labels in this dataset, which are optional data, for using NEExT. The datasets were genearted using the ABCD Framework found here (https://github.com/bkamins/ABCDGraphGenerator.jl)

## Loading Data

First we deine a path to the datasets. They are `csv` files, with format as defined in the README file.


```python
edge_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/xi_n/edge_file.csv"
graph_label_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/xi_n/graph_label_mapping_file.csv"
node_graph_mapping_file = "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/xi_n/node_graph_mapping_file.csv"
```

Now we can instantiate a NEExT object.


```python
nxt = NEExT(quiet_mode="on")
```

You can load data using the `load_data_from_csv` method:


```python
nxt.load_data_from_csv(edge_file=edge_file, node_graph_mapping_file=node_graph_mapping_file, graph_label_file=graph_label_file)
```

## Building Features

You can now compute various features on nodes of the subgraphs in the graph collection loaded above.<br>
This can be done using the method `compute_graph_feature`. <br>
To get the list of available node features, you can use the function `get_list_of_graph_features`.


```python
nxt.get_list_of_graph_features()
```




    ['lsme',
     'self_walk',
     'basic_expansion',
     'basic_node_features',
     'page_rank',
     'degree_centrality',
     'closeness_centrality',
     'load_centrality',
     'eigenvector_centrality',
     'anomaly_score_CADA',
     'normalized_anomaly_score_CADA',
     'community_association_strength',
     'normalized_within_module_degree',
     'participation_coefficient']



These are the type of node features you can compute on every node on each graph in the graph collection. <br>
So for example, let's compute `page_rank`. We also need to defined what the feature vector size should be.


```python
nxt.compute_graph_feature(feat_name="page_rank", feat_vect_len=4)
```

To compute additional features, simply use the same function, and provide the length of the vector size.<br>
Let's add degree centrality to the list of computed features.


```python
nxt.compute_graph_feature(feat_name="degree_centrality", feat_vect_len=4)
```

## Building Global Feature Object

Right now, we have 2 features computed on every node, for every graph. We can use these features to construct a overall pooled feature vector, which can be used to construct graph embeddings. <br>
To do this, we can pool the features using the `pool_grpah_features` method.


```python
nxt.pool_graph_features(pool_method="concat")
```

The overall feature (which we call global feature) is a concatenated vector of whatever features you have computed on the graph. In this example it would be a 8 dimensional vector of `page_rank` and `degree_centrality`.<br>
You can access the global vector by using the `get_global_feature_vector` method.


```python
df = nxt.get_global_feature_vector()
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node_id</th>
      <th>graph_id</th>
      <th>feat_degree_centrality_0</th>
      <th>feat_degree_centrality_1</th>
      <th>feat_degree_centrality_2</th>
      <th>feat_degree_centrality_3</th>
      <th>feat_page_rank_0</th>
      <th>feat_page_rank_1</th>
      <th>feat_page_rank_2</th>
      <th>feat_page_rank_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>4.094288</td>
      <td>1.723672</td>
      <td>2.023497</td>
      <td>2.122162</td>
      <td>4.014656</td>
      <td>1.825315</td>
      <td>2.003575</td>
      <td>2.062771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2.682074</td>
      <td>1.689427</td>
      <td>2.023497</td>
      <td>2.082461</td>
      <td>2.651835</td>
      <td>1.745939</td>
      <td>2.042548</td>
      <td>2.045115</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2.682074</td>
      <td>1.578132</td>
      <td>2.120736</td>
      <td>2.131372</td>
      <td>2.672592</td>
      <td>1.696518</td>
      <td>2.058271</td>
      <td>2.071704</td>
    </tr>
  </tbody>
</table>
</div>



## Dimensionality Reduction

We may wish to reduce the number of dimensions of our data, which could help downstream tasks such as Embedding generation or machine learning tasks. This can be done using the `apply_dim_reduc_to_graph_feats`.


```python
nxt.apply_dim_reduc_to_graph_feats(dim_size=4, reducer_type="pca")
```

If we take a look at the `global feature vector` we can see that it is upaded with the new size of dimension.


```python
df = nxt.get_global_feature_vector()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node_id</th>
      <th>graph_id</th>
      <th>feat_0</th>
      <th>feat_1</th>
      <th>feat_2</th>
      <th>feat_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2.560267</td>
      <td>3.463473</td>
      <td>1.190519</td>
      <td>1.075334</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2.219077</td>
      <td>1.481326</td>
      <td>1.365794</td>
      <td>0.532872</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2.225506</td>
      <td>1.488865</td>
      <td>1.925751</td>
      <td>0.467088</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>2.129568</td>
      <td>0.363245</td>
      <td>-0.311636</td>
      <td>1.752476</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>2.009336</td>
      <td>0.529704</td>
      <td>3.251858</td>
      <td>-1.665345</td>
    </tr>
  </tbody>
</table>
</div>



You still have access to the pre-dimensionality reduction global vector by using the method `get_archived_global_feature_vector`.


```python
df = nxt.get_archived_global_feature_vector()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node_id</th>
      <th>graph_id</th>
      <th>feat_degree_centrality_0</th>
      <th>feat_degree_centrality_1</th>
      <th>feat_degree_centrality_2</th>
      <th>feat_degree_centrality_3</th>
      <th>feat_page_rank_0</th>
      <th>feat_page_rank_1</th>
      <th>feat_page_rank_2</th>
      <th>feat_page_rank_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>4.094288</td>
      <td>1.723672</td>
      <td>2.023497</td>
      <td>2.122162</td>
      <td>4.014656</td>
      <td>1.825315</td>
      <td>2.003575</td>
      <td>2.062771</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2.682074</td>
      <td>1.689427</td>
      <td>2.023497</td>
      <td>2.082461</td>
      <td>2.651835</td>
      <td>1.745939</td>
      <td>2.042548</td>
      <td>2.045115</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2.682074</td>
      <td>1.578132</td>
      <td>2.120736</td>
      <td>2.131372</td>
      <td>2.672592</td>
      <td>1.696518</td>
      <td>2.058271</td>
      <td>2.071704</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>1.975967</td>
      <td>2.082671</td>
      <td>1.851304</td>
      <td>2.161749</td>
      <td>1.968745</td>
      <td>2.028736</td>
      <td>1.879435</td>
      <td>2.154684</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>1.975967</td>
      <td>1.355541</td>
      <td>2.346133</td>
      <td>1.966471</td>
      <td>1.940827</td>
      <td>1.384500</td>
      <td>2.274468</td>
      <td>1.972817</td>
    </tr>
  </tbody>
</table>
</div>



## Building Graph Embeddings

This function returns a Pandas DataFrame, with the collection features and how they map to the graphs and nodes. <br>
One thing to note is that the data is standardized across all graphs.

We can use the features computed on the graphs to build graph embeddings. To see what graph embedding engines are available to use, we can use the `get_list_of_graph_embedding_engines` function.


```python
nxt.get_list_of_graph_embedding_engines()
```




    ['approx_wasserstein', 'wasserstein', 'sinkhornvectorizer']



Now, let's build a 3 dimensional embedding for every graph in graph collection using the Approximate Wasserstein embedding engine. This can be done by using the method `build_graph_embedding`.


```python
nxt.build_graph_embedding(emb_dim_len=3, emb_engine="approx_wasserstein")
```

You can access the embedding results by using the method `get_graph_embeddings`.


```python
df = nxt.get_graph_embeddings()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>emb_0</th>
      <th>emb_1</th>
      <th>emb_2</th>
      <th>graph_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.121117</td>
      <td>1.715908</td>
      <td>0.420738</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.927579</td>
      <td>1.293987</td>
      <td>1.120732</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.070954</td>
      <td>1.024027</td>
      <td>0.343173</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.682049</td>
      <td>0.990811</td>
      <td>0.106760</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.281254</td>
      <td>0.773949</td>
      <td>0.212976</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



## Visualize Embeddings

You can use the builtin visualization function to gain quick insights into the performance of your embeddings. This can be done by using the method `visualize_graph_embedding`. If you have labels for your graph (like the case here), we can color the embedding distributions using the labels. By default, embeddings are not colored.

## Using Sampled Sub-Graphs

We may often have to deal with large graphs, both in the number of sub-graphs in the collection, and also the size of each graph. To allow for faster computation, we can sample each sub-graph and compute metrics and features for a fraction of nodes on each sub-graph. This can be done by using the method `build_node_sample_collection`. It takes as input the fraction of sampled nodes. Once this method is called all further computation will use the sampled node collection.


```python
nxt.build_node_sample_collection(sample_rate=0.1)
```

## Adding Custom Node Feature Function

You can define and load into `NEExT` you own node feature function. Below, we show an example of loading a custom function into `NEExT` and using it to compute node features. The only thing to keep in mind is the interface for the function, meaning the input and output format.

#### Input:
Your custom function should have the following inputs:
```
func(G, feat_vect_len):
    ...
```
Where `G` is a `NetworkX` graph object and `feat_vect_len` is an `int` indicating the length of the node feature vector.
#### Output:
Your function should have the following output:
```
func(G, feat_vect_len):
    ...
    feat_vect = {node_id : [v1, v2, v3, ...], ...}
    return feat_vect
```
Where the output is a `dict`, where the keys are the node ids and the values of list, with elements being the values of features for that node. The length of the feature vector should be the same as the input `feat_vect_len`.

One thing to note is that, if you have applied `sampling` (as we have done above), `NEExT` will automatically load a sampled version of the graphs into your custom function. The NetworkX graph G passed to your function is a sub-graph with only a fraction of nodes, as defined by the sampling rate.

Below, we show an example of:
* Creating a custom function, where we have a feature vector of only zero (you can do something more complicated)
* Loading the custom function into NEExT using `load_custom_node_feature_function` with two parameters (function and function_name)
* Calling the custom function to compute features.
* Concatinating the new features with the old one (from above)
* Displaying the new gloabl feature DataFrame.


```python
def my_custom_node_feature(G, feat_vect_len):
    feat_vect = {}
    for i in G.nodes:
        feat_vect[i] = [0]*feat_vect_len
    return feat_vect

nxt.load_custom_node_feature_function(function=my_custom_node_feature, function_name="my_custom_node_feature")
nxt.get_list_of_graph_features()
```




    ['lsme',
     'self_walk',
     'basic_expansion',
     'basic_node_features',
     'page_rank',
     'degree_centrality',
     'closeness_centrality',
     'load_centrality',
     'eigenvector_centrality',
     'anomaly_score_CADA',
     'normalized_anomaly_score_CADA',
     'community_association_strength',
     'normalized_within_module_degree',
     'participation_coefficient',
     'my_custom_node_feature']




```python
nxt.compute_graph_feature(feat_name="my_custom_node_feature", feat_vect_len=4)
```


```python
nxt.pool_graph_features(pool_method="concat")
```


```python
df = nxt.get_global_feature_vector()
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>node_id</th>
      <th>graph_id</th>
      <th>feat_degree_centrality_0</th>
      <th>feat_degree_centrality_1</th>
      <th>feat_degree_centrality_2</th>
      <th>feat_degree_centrality_3</th>
      <th>feat_page_rank_0</th>
      <th>feat_page_rank_1</th>
      <th>feat_page_rank_2</th>
      <th>feat_page_rank_3</th>
      <th>feat_my_custom_node_feature_0</th>
      <th>feat_my_custom_node_feature_1</th>
      <th>feat_my_custom_node_feature_2</th>
      <th>feat_my_custom_node_feature_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>0</td>
      <td>2.005123</td>
      <td>1.354005</td>
      <td>2.347066</td>
      <td>1.959329</td>
      <td>1.977187</td>
      <td>1.385398</td>
      <td>2.273995</td>
      <td>1.966959</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14</td>
      <td>0</td>
      <td>2.005123</td>
      <td>2.268527</td>
      <td>2.061383</td>
      <td>1.982474</td>
      <td>1.972862</td>
      <td>2.141328</td>
      <td>2.009107</td>
      <td>2.031782</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>0</td>
      <td>0.553677</td>
      <td>1.141169</td>
      <td>1.825850</td>
      <td>2.434965</td>
      <td>0.834835</td>
      <td>1.422558</td>
      <td>1.834757</td>
      <td>2.336032</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
