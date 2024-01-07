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
* NetworkX Objects
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


