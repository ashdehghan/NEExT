
import networkx as nx


def divide_chunks(list, chunks): 
	for i in range(0, len(list), chunks):  
		yield list[i:i + chunks] 


def get_numb_of_nb_x_hops_away(G, node, max_hop_length):
	"""
		This method will compute the number of neighbors x hops away from
		a given node.
	"""
	node_dict = {}
	dist_dict = {}
	node_list = [node]
	keep_going = True
	while keep_going:
		n = node_list.pop(0)
		nbs = G.neighbors(n)
		for nb in nbs:
			if (nb not in node_dict) and (nb != node):
				node_list.append(nb)
				dist_to_source = len(nx.shortest_path(G, source=node, target=nb)) - 1
				node_dict[nb] = dist_to_source
				if dist_to_source not in dist_dict:
					dist_dict[dist_to_source] = [nb]
				else:
					dist_dict[dist_to_source].append(nb)
				if dist_to_source >= max_hop_length:
					keep_going = False
		if len(node_list) == 0:
			keep_going = False
	# Build dist list
	max_hop = max(list(dist_dict.keys()))
	hop_list = []
	for i in range(1, max_hop+1):
		hop_list.append(len(dist_dict[i]))
	hop_list = hop_list + [0]*(max_hop_length-len(hop_list))
	return hop_list


def get_nodes_x_hops_away(G, node, max_hop_length):
	"""
		This method will compute the number of neighbors x hops away from
		a given node.
	"""
	node_dict = {}
	dist_dict = {}
	node_list = [node]
	keep_going = True
	while keep_going:
		n = node_list.pop(0)
		nbs = G.neighbors(n)
		for nb in nbs:
			if (nb not in node_dict) and (nb != node):
				node_list.append(nb)
				dist_to_source = len(nx.shortest_path(G, source=node, target=nb)) - 1
				node_dict[nb] = dist_to_source
				if dist_to_source not in dist_dict:
					dist_dict[dist_to_source] = [nb]
				else:
					dist_dict[dist_to_source].append(nb)
				if dist_to_source >= max_hop_length:
					keep_going = False
		if len(node_list) == 0:
			keep_going = False
	return dist_dict