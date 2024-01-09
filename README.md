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
     'eigenvector_centrality']



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
nxt.get_global_feature_vector()
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
      <th>feat_page_rank_0</th>
      <th>feat_page_rank_1</th>
      <th>feat_page_rank_2</th>
      <th>feat_page_rank_3</th>
      <th>feat_degree_centrality_0</th>
      <th>feat_degree_centrality_1</th>
      <th>feat_degree_centrality_2</th>
      <th>feat_degree_centrality_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>4.014656</td>
      <td>1.645432</td>
      <td>1.825315</td>
      <td>2.003575</td>
      <td>4.094288</td>
      <td>1.632019</td>
      <td>1.723672</td>
      <td>2.023497</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2.651835</td>
      <td>1.999918</td>
      <td>1.745939</td>
      <td>2.042548</td>
      <td>2.682074</td>
      <td>2.024244</td>
      <td>1.689427</td>
      <td>2.023497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2.672592</td>
      <td>1.917080</td>
      <td>1.696518</td>
      <td>2.058271</td>
      <td>2.682074</td>
      <td>1.915292</td>
      <td>1.578132</td>
      <td>2.120736</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>1.968745</td>
      <td>1.937933</td>
      <td>2.028736</td>
      <td>1.879435</td>
      <td>1.975967</td>
      <td>1.993115</td>
      <td>2.082671</td>
      <td>1.851304</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>1.940827</td>
      <td>2.407239</td>
      <td>1.384500</td>
      <td>2.274468</td>
      <td>1.975967</td>
      <td>2.491178</td>
      <td>1.355541</td>
      <td>2.346133</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7495</th>
      <td>395</td>
      <td>24</td>
      <td>-0.829459</td>
      <td>-1.399973</td>
      <td>-1.030338</td>
      <td>-0.951132</td>
      <td>-0.853770</td>
      <td>-1.468206</td>
      <td>-1.059412</td>
      <td>-0.963018</td>
    </tr>
    <tr>
      <th>7496</th>
      <td>396</td>
      <td>24</td>
      <td>-1.149447</td>
      <td>-1.507187</td>
      <td>-1.014597</td>
      <td>-0.950959</td>
      <td>-1.205938</td>
      <td>-1.598620</td>
      <td>-1.042336</td>
      <td>-0.959924</td>
    </tr>
    <tr>
      <th>7497</th>
      <td>397</td>
      <td>24</td>
      <td>-1.155132</td>
      <td>-1.514255</td>
      <td>-0.920724</td>
      <td>-0.995089</td>
      <td>-1.205938</td>
      <td>-1.598620</td>
      <td>-0.936739</td>
      <td>-1.010254</td>
    </tr>
    <tr>
      <th>7498</th>
      <td>398</td>
      <td>24</td>
      <td>-1.159577</td>
      <td>-1.429980</td>
      <td>-1.003237</td>
      <td>-0.983561</td>
      <td>-1.205938</td>
      <td>-1.511677</td>
      <td>-1.031103</td>
      <td>-0.995703</td>
    </tr>
    <tr>
      <th>7499</th>
      <td>399</td>
      <td>24</td>
      <td>-1.142376</td>
      <td>-1.585013</td>
      <td>-0.961202</td>
      <td>-0.894351</td>
      <td>-1.205938</td>
      <td>-1.685562</td>
      <td>-0.983921</td>
      <td>-0.895910</td>
    </tr>
  </tbody>
</table>
<p>7500 rows × 10 columns</p>
</div>



## Dimensionality Reduction

We may wish to reduce the number of dimensions of our data, which could help downstream tasks such as Embedding generation or machine learning tasks. This can be done using the `apply_dim_reduc_to_graph_feats`.


```python
nxt.apply_dim_reduc_to_graph_feats(dim_size=4, reducer_type="pca")
```

If we take a look at the `global feature vector` we can see that it is upaded with the new size of dimension.


```python
nxt.get_global_feature_vector()
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
      <th>emb_0</th>
      <th>emb_1</th>
      <th>emb_2</th>
      <th>emb_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>2.471714</td>
      <td>3.577450</td>
      <td>0.394070</td>
      <td>0.779143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2.232913</td>
      <td>1.420164</td>
      <td>0.969629</td>
      <td>0.912235</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2.202837</td>
      <td>1.494916</td>
      <td>0.809437</td>
      <td>1.537148</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>2.102230</td>
      <td>0.403983</td>
      <td>0.199739</td>
      <td>-0.931054</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>2.164103</td>
      <td>0.202613</td>
      <td>2.194223</td>
      <td>3.052554</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7495</th>
      <td>395</td>
      <td>24</td>
      <td>-1.150994</td>
      <td>0.263208</td>
      <td>-1.147955</td>
      <td>0.737321</td>
    </tr>
    <tr>
      <th>7496</th>
      <td>396</td>
      <td>24</td>
      <td>-1.258108</td>
      <td>-0.154415</td>
      <td>-1.594372</td>
      <td>0.813288</td>
    </tr>
    <tr>
      <th>7497</th>
      <td>397</td>
      <td>24</td>
      <td>-1.245211</td>
      <td>-0.174141</td>
      <td>-1.731441</td>
      <td>0.256937</td>
    </tr>
    <tr>
      <th>7498</th>
      <td>398</td>
      <td>24</td>
      <td>-1.243488</td>
      <td>-0.198692</td>
      <td>-1.372653</td>
      <td>0.566846</td>
    </tr>
    <tr>
      <th>7499</th>
      <td>399</td>
      <td>24</td>
      <td>-1.247078</td>
      <td>-0.143294</td>
      <td>-1.960032</td>
      <td>0.931437</td>
    </tr>
  </tbody>
</table>
<p>7500 rows × 6 columns</p>
</div>



You still have access to the pre-dimensionality reduction global vector by using the method `get_archived_global_feature_vector`.


```python
nxt.get_archived_global_feature_vector()
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
      <th>feat_page_rank_0</th>
      <th>feat_page_rank_1</th>
      <th>feat_page_rank_2</th>
      <th>feat_page_rank_3</th>
      <th>feat_degree_centrality_0</th>
      <th>feat_degree_centrality_1</th>
      <th>feat_degree_centrality_2</th>
      <th>feat_degree_centrality_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>4.014656</td>
      <td>1.645432</td>
      <td>1.825315</td>
      <td>2.003575</td>
      <td>4.094288</td>
      <td>1.632019</td>
      <td>1.723672</td>
      <td>2.023497</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2.651835</td>
      <td>1.999918</td>
      <td>1.745939</td>
      <td>2.042548</td>
      <td>2.682074</td>
      <td>2.024244</td>
      <td>1.689427</td>
      <td>2.023497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>2.672592</td>
      <td>1.917080</td>
      <td>1.696518</td>
      <td>2.058271</td>
      <td>2.682074</td>
      <td>1.915292</td>
      <td>1.578132</td>
      <td>2.120736</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>1.968745</td>
      <td>1.937933</td>
      <td>2.028736</td>
      <td>1.879435</td>
      <td>1.975967</td>
      <td>1.993115</td>
      <td>2.082671</td>
      <td>1.851304</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0</td>
      <td>1.940827</td>
      <td>2.407239</td>
      <td>1.384500</td>
      <td>2.274468</td>
      <td>1.975967</td>
      <td>2.491178</td>
      <td>1.355541</td>
      <td>2.346133</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7495</th>
      <td>395</td>
      <td>24</td>
      <td>-0.829459</td>
      <td>-1.399973</td>
      <td>-1.030338</td>
      <td>-0.951132</td>
      <td>-0.853770</td>
      <td>-1.468206</td>
      <td>-1.059412</td>
      <td>-0.963018</td>
    </tr>
    <tr>
      <th>7496</th>
      <td>396</td>
      <td>24</td>
      <td>-1.149447</td>
      <td>-1.507187</td>
      <td>-1.014597</td>
      <td>-0.950959</td>
      <td>-1.205938</td>
      <td>-1.598620</td>
      <td>-1.042336</td>
      <td>-0.959924</td>
    </tr>
    <tr>
      <th>7497</th>
      <td>397</td>
      <td>24</td>
      <td>-1.155132</td>
      <td>-1.514255</td>
      <td>-0.920724</td>
      <td>-0.995089</td>
      <td>-1.205938</td>
      <td>-1.598620</td>
      <td>-0.936739</td>
      <td>-1.010254</td>
    </tr>
    <tr>
      <th>7498</th>
      <td>398</td>
      <td>24</td>
      <td>-1.159577</td>
      <td>-1.429980</td>
      <td>-1.003237</td>
      <td>-0.983561</td>
      <td>-1.205938</td>
      <td>-1.511677</td>
      <td>-1.031103</td>
      <td>-0.995703</td>
    </tr>
    <tr>
      <th>7499</th>
      <td>399</td>
      <td>24</td>
      <td>-1.142376</td>
      <td>-1.585013</td>
      <td>-0.961202</td>
      <td>-0.894351</td>
      <td>-1.205938</td>
      <td>-1.685562</td>
      <td>-0.983921</td>
      <td>-0.895910</td>
    </tr>
  </tbody>
</table>
<p>7500 rows × 10 columns</p>
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
nxt.get_graph_embeddings()
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
      <td>2.038486</td>
      <td>1.463379</td>
      <td>0.080776</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.874913</td>
      <td>1.535265</td>
      <td>0.475480</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.021950</td>
      <td>0.849217</td>
      <td>-0.418307</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.726050</td>
      <td>0.750470</td>
      <td>-0.317739</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.313531</td>
      <td>0.656964</td>
      <td>0.077666</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.033228</td>
      <td>0.316619</td>
      <td>-0.397285</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.855314</td>
      <td>0.122401</td>
      <td>0.094769</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.028675</td>
      <td>0.177609</td>
      <td>-0.180609</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.708892</td>
      <td>0.214789</td>
      <td>-0.117356</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-1.307526</td>
      <td>0.197537</td>
      <td>0.116854</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.042245</td>
      <td>-0.204380</td>
      <td>0.022146</td>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.858563</td>
      <td>-0.171441</td>
      <td>0.130683</td>
      <td>11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.038904</td>
      <td>-0.273235</td>
      <td>-0.191326</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.703492</td>
      <td>-0.078910</td>
      <td>0.017633</td>
      <td>13</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-1.304300</td>
      <td>-0.153512</td>
      <td>0.000668</td>
      <td>14</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2.055063</td>
      <td>-0.756504</td>
      <td>0.304664</td>
      <td>15</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.863735</td>
      <td>-0.482437</td>
      <td>0.351043</td>
      <td>16</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.046502</td>
      <td>-0.526162</td>
      <td>0.008372</td>
      <td>17</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.704319</td>
      <td>-0.339239</td>
      <td>0.095141</td>
      <td>18</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-1.302084</td>
      <td>-0.286804</td>
      <td>0.075835</td>
      <td>19</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2.039145</td>
      <td>-0.938570</td>
      <td>-0.315529</td>
      <td>20</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.859070</td>
      <td>-0.545298</td>
      <td>0.026595</td>
      <td>21</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.045740</td>
      <td>-0.657624</td>
      <td>-0.056536</td>
      <td>22</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.701785</td>
      <td>-0.471908</td>
      <td>-0.042016</td>
      <td>23</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-1.299585</td>
      <td>-0.387853</td>
      <td>0.158255</td>
      <td>24</td>
    </tr>
  </tbody>
</table>
</div>



## Visualize Embeddings

You can use the builtin visualization function to gain quick insights into the performance of your embeddings. This can be done by using the method `visualize_graph_embedding`. If you have labels for your graph (like the case here), we can color the embedding distributions using the labels. By default, embeddings are not colored.
