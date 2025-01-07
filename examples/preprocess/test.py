import networkx as nx
import pickle

G_nx: nx.Graph = pickle.load(open("../datasets/amazon/graph.pkl", "rb"))
print("# nodes:", G_nx.number_of_nodes())
print("# edges:", G_nx.number_of_edges())

print(f"graph is directed: {G_nx.is_directed()}")
input()
