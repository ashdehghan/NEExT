

class Graph_Object:

	def __init__(self):
		self.graph_id = None
		self.graph = None
		self.numb_of_nodes = None
		self.numb_of_edges = None
		self.numb_of_connected_components = None
		self.connected_components = None
		self.feature_collection = {}
		self.feature_collection["features"] = {}
		self.feature_collection["pooled_features"] = None
		self.computed_features = set()