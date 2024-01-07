"""
	Author: Ash Dehghan
	Description: This class provides some builtin node and structural embedding
	solutions to be used by UGAF. The user could provide their own node embeddings.
"""

# External Libraries
import random
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from karateclub import DeepWalk

# Internal Libraries
from NEExT.helper_functions import get_numb_of_nb_x_hops_away

class Node_Embedding_Engine:


	def __init__(self):
		pass


	def run_node2vec_embedding(self, G, emb_dim):
		"""
			This method takes as input a networkx graph object
			and runs a Node2Vec embedding using default values.
		"""
		node2vec = Node2Vec(G, dimensions=emb_dim, walk_length=30, num_walks=50, workers=4, quiet=True)
		model = node2vec.fit(window=10, min_count=1, batch_words=4)
		embeddings = {node: list(model.wv[node]) for node in G.nodes()}
		return embeddings


	def run_lsme_embedding(self, G, emb_dim):
		"""
			This method takes as input a networkx graph object
			and runs a LSME structural embedding.
		"""
		nodes = list(G.nodes)
		embeddings = []
		node_list = []
		for node in nodes:
			node_list.append(node)
			emb = self.lsme_run_random_walk(node, G, sample_size=50, rw_length=10)
			if len(emb) < emb_dim:
				emb += [0] * (emb_dim - len(emb))
			else:
				emb = emb[0:emb_dim]
			embeddings.append(emb)
		embeddings = pd.DataFrame(embeddings)
		emb_cols = ["feat_lsme_"+str(i) for i in range(embeddings.shape[1])]
		embeddings.columns = emb_cols
		embeddings.insert(0, "node_id", node_list)
		return embeddings


	def lsme_run_random_walk(self, root_node, G, sample_size=50, rw_length=30):
		"""
			This method runs the random-walk for the LSME algorithm
		"""
		walk = {}
		for i in range(sample_size):
			c_node = root_node
			for i in range(rw_length):
				neighbors = [n for n in G.neighbors(c_node)]
				n_node = random.choice(neighbors)
				dist_b = len(nx.shortest_path(G, root_node, c_node)) - 1
				dist_a = len(nx.shortest_path(G, root_node, n_node)) - 1
				c_node = n_node
				if dist_b in walk.keys():
					if dist_a in walk[dist_b].keys():
						walk[dist_b]["total"] += 1
						walk[dist_b][dist_a] += 1
					else:
						walk[dist_b]["total"] += 1
						walk[dist_b][dist_a] = 1
				else:
					walk[dist_b] = {}
					walk[dist_b]["total"] = 1
					walk[dist_b][dist_a] = 1
		max_walk = max(list(walk.keys()))
		emb = []
		for i in range(0, max_walk+1):
			step = walk[i]
			b = i-1
			s = i
			f = i+1
			pb = step[b]/step["total"] if b in step.keys() else 0
			ps = step[s]/step["total"] if s in step.keys() else 0
			pf = step[f]/step["total"] if f in step.keys() else 0
			emb += [pb, ps, pf]
		emb = emb[3:len(emb)+1]
		return emb


	def run_deepwalk_embedding(self, G, emb_dim):
		"""
			This method takes as input a networkx graph object
			and runs a DeepWalk node embedding.
		"""
		model =  DeepWalk(dimensions=emb_dim)
		model.fit(G)
		dw_emb = model.get_embedding()
		embeddings = {}
		for index, node in enumerate(list(G.nodes)):
			embeddings[node] = list(dw_emb[index])
		return embeddings


	def run_expansion_embedding(self, G, emb_dim):
		"""
			This method takes as input a networkx graph object
			and runs a simple expansion property embedding.
		"""
		embeddings = []
		node_list = []
		d = (2 * len(G.edges))/len(G.nodes)
		for node in G.nodes:
			node_list.append(node)
			dist_list = get_numb_of_nb_x_hops_away(G, node, emb_dim)
			norm_list = []
			for i in range(len(dist_list)):
				if i == 0:
					norm_val = 1 * d
				else:
					norm_val = dist_list[i] - 1
					if norm_val <= 0:
						norm_val = 1
				norm_list.append(norm_val * d)
			emb = [dist_list[i]/norm_list[i] for i in range(len(dist_list))]
			embeddings.append(emb)
		embeddings = pd.DataFrame(embeddings)
		emb_cols = ["feat_basic_expansion_"+str(i) for i in range(embeddings.shape[1])]
		embeddings.columns = emb_cols
		embeddings.insert(0, "node_id", node_list)
		return embeddings
					














