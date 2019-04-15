import networkx as nx
import matplotlib.pyplot as plt

number_nodes = 10
G_complete = nx.complete_graph(n=number_nodes)
nx.draw_circular(G_complete)
plt.show()

G_sparse = nx.empty_graph(n=number_nodes)
for node in range(number_nodes-1):
    G_sparse.add_edge(node, node+1)
G_sparse.add_edge(number_nodes-1, 0)
nx.draw_circular(G_sparse)
plt.show()
