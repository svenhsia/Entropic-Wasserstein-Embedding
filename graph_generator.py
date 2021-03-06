import networkx as nx
import numpy as np

class GraphGenerator(object):
    def __init__(self, graph_type='scale-free', n_nodes=128, m=6, p=0.4):
        if graph_type == 'scale-free':
            graph = nx.barabasi_albert_graph(n_nodes, m)
        elif graph_type == 'small-world':
            graph = nx.watts_strogatz_graph(n_nodes, m, p)
        # elif graph_type == 'community-structured':
        #     graph = nx.stochastic_block_model()
        elif graph_type == 'random-trees':
            graph = GraphGenerator.random_trees(n_nodes)
        elif graph_type == 'google-web':
            graph = GraphGenerator.web_google(n_nodes)

        self.graph = nx.convert_node_labels_to_integers(graph)
    
    @staticmethod
    def random_trees(n_nodes):
        g = nx.Graph()
        g.add_node(0)
        leaves = [0]
        count = 1
        while count < n_nodes:
            parent = leaves.pop(0)
            n_children = np.random.randint(2, 5)
            for _ in range(n_children):
                if count < n_nodes:
                    g.add_edge(parent, count)
                    count += 1
                else:
                    break
        assert len(g.nodes) == n_nodes
        return g
    
    @staticmethod
    def web_google(n_nodes):
        g = nx.Graph()
        with open('./graphs/web_google/web-Google.txt', 'r') as fin:
            for line in fin:
                if not line.startswith('#'):
                    node1, node2 = line.strip().split()
                    g.add_edge(int(node1), int(node2))
        start_vertex = np.random.choice(g.nodes)
        kept_nodes = [start_vertex]
        to_visit = [start_vertex]
        while len(kept_nodes) < n_nodes:
            visit_node = to_visit.pop(0)
            to_add = [n for n in g[visit_node] if n not in kept_nodes]
            kept_nodes += to_add
            to_visit += to_add
        kept_nodes = kept_nodes[:n_nodes]
        return g.subgraph(kept_nodes)

    def get_node_pairs(self):
        nodes = self.graph.nodes
        node_pairs = [[i, j] for i in range(len(nodes)) for j in range(i+1, len(nodes))]
        self.node_pairs = np.array(node_pairs, dtype=np.int32)
        return self.node_pairs
    
    def get_obj_distances(self):
        obj_distances = np.array([nx.shortest_path_length(self.graph, source=n1, target=n2) for n1, n2 in self.node_pairs])
        self.obj_distances = obj_distances.astype(np.float64)
        return self.obj_distances

# # generate scale free graphs to ensure that all types of embeddings use the same graphs
# num_nodes = 64
# for i in range(20):
#     G = GraphGenerator(graph_type='scale-free', n_nodes=num_nodes, m=2)
#     nx.write_gpickle(G, "./graphs/scale_free_{}_{}.pickle".format(num_nodes, i+1))
#     print("Finishing writing scale free graph {}".format(i+1))

# # test read pickle
# G = nx.read_gpickle("./graphs/scale_free_64_1.pickle")
# print(G.get_node_pairs()[0])
# print(G.get_obj_distances()[0])
