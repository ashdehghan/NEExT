# NEExT

### Unsupervised Graph Analysis Framework
UGAF is a frameowork for analysis of a collection for graphs. This includes functionality such as:
* Cleansing and standardizing graph data.
* Creating node and structural embedding for nodes in the graph collection.
* Creating embedding for graphs (graph embedding).

### Instalation Process
UGAF uses Python 3.x (currently tested using Python 3.11).
You can install UGAF using the following:
```console
pip install git+https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/elmspace/ugaf.git
```
where `GIT_USERNAME` and `GIT_PASSWORD` are environment variables, which you can set in terminal by using:
```console
export GIT_USERNAME=<your git username>
```
and 
```console
export GIT_PASSWORD=<your git classic token>
```
** Future version of UGAF will be on public PyPi, which would allow you to install it using `pip` directly.

### Graph Data Format
### Graph Collection Data
Here, we cover data format for creating graph collection data:
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

