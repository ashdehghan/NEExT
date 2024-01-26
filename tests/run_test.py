# %%
import networkx as nx

import sys
sys.path.append("../")

from NEExT.helper_functions import get_nodes_x_hops_away, get_numb_of_nb_x_hops_away

def test_wrapper(func):
	try:
		res = func()
		if res:
			print(f"Test {func.__name__} passed.")
			return True
		else:
			print(f"Test {func.__name__} failed.")
			return False
			
	except Exception as e:
		print(e)
		return False

def test_get_nodes_x_hops_away():
	G = nx.cycle_graph(6)
	return get_nodes_x_hops_away(G, 0, 3) == [{1, 5}, {2, 4}, {3}]

def test_get_numb_of_nb_x_hops_away():
	G = nx.cycle_graph(6)
	return get_numb_of_nb_x_hops_away(G, 0, 3) == [2, 2, 1]

if __name__ == "__main__":
	methods = [ test_get_nodes_x_hops_away, test_get_numb_of_nb_x_hops_away ]
	results = [ test_wrapper(method) for method in methods ]
	print("--------------------------------------------------")
	print(f"Passed {sum(results)} out of {len(results)} tests.")