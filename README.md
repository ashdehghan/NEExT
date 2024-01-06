# NEExT

### Network Embedding Exploration Tool

NEExT is a tool for exploring and building graph embeddings. This tool allows for:
* Cleansing and standardizing a collection of graph data.
* Creating node and structural features for nodes in the graph collection.
* Creating embeddings for graphs.

### Instalation Process
NEExT uses Python 3.x (currently tested using Python 3.11).
You can install NEExT using the following:
```console
pip install NEExT
```

### Graph Data Format
You can use a few different data formats to upload data inot NEExT. Currently, it allows for:
* CSV files
* NetworkX Objects
See below for examples of using different data formats.

#### Using CSV Files

UGAF expects input graph data to be in `csv` format. To create a graph collection, you would need two csv files:
The `edge csv file`, contains the relationship between nodes (how they are connected). Here is an example:
|node_a|node_b|
|---|---|
|1|2|
|3|2|
|.|.|

The `node graph mapping csv file` contains the relationship between nodes and graphs. In other words, which node belongs to which graph.
Here is an example:
|node_id|graph_id|
|---|---|
|0|1|
|1|1|
|2|1|
|3|2|
|4|2|
|.|.|

Here, nodes (0, 1 and 2) belong to graph 1 and nodes (3 and 4) belong to graph 2.
** Note that UGAF does not make the assumption that each graph is singhle connected component. Although you could filter for only connected component, as you shall in the example section.

