import networkx as nx
import os.path
from random import random, choice, uniform
import math


class Node:
    def __init__(self, _id, x, y):
        self.id = _id
        self.x = x
        self.y = y


class Edge:

    def __init__(self, capacity, flow, residual=False):
        self.residual = residual
        self.capacity = capacity
        self.flow = flow


class Graph:

    def __init__(self, n, r, upperCap, external=False):
        if not external:
            self.meta_data = {}
            self.nodes = []
            for node_id in range(n):
                self.nodes.append(Node(node_id, uniform(0, 1), uniform(0, 1)))

            self.edges = {y: {x: Edge(0, 0) for x in self.nodes} for y in self.nodes}  # dict node to node
            self.generate_sink_source_graph(r, upperCap)
            self.meta_data["n"] = n
            self.meta_data['r'] = r
            self.meta_data['c'] = upperCap
        else:
            pass

    def initialize(self, nodes, edges, meta_data):
        self.nodes = nodes
        self.edges = edges
        self.meta_data = meta_data

    def link_nodes(self, source, target, capacity):
        if self.edges[source][target].capacity != 0:
            return print("Warning, edge already exists, replacing old capacity with 0")
        self.edges[source][target].capacity = capacity
        self.edges[target][source].capacity = capacity
        self.edges[target][source].flow = capacity
        self.edges[target][source].residual = True

    def reset(self):
        for n1 in self.nodes:
            for n2 in self.nodes:
                edge = self.edges[n1][n2]
                if edge.residual:
                    edge.flow = edge.capacity
                else:
                    edge.flow = 0

def load_graphs(path):
    graphs = []
    with open(path, 'r') as f:
        nodes = []
        edges = None
        edge_count = 0
        for line in f.readlines():
            type, line = line.split(":")

            if type == "graph":
                source_id, target_id, target_distance, n, r, c = line.split(";")

                source_id, target_id, target_distance, n, r, c = int(source_id), int(target_id), int(
                    target_distance), int(n), float(r), int(c)

                g = Graph(None, None, None, external=True)
                meta_data = {"s": source_id, 't': target_id, 'target_distance': target_distance, 'n': n, 'r': r,
                             'c': c, 'edge_count': edge_count}
                g.initialize(nodes, edges, meta_data)

                graphs.append(g)

                nodes = []
                edges = None
                edge_count = 0

            elif type == "node":
                node_id, x, y = line.split(";")
                node_id, x, y = int(node_id), float(x), float(y)
                new_node = Node(node_id, x, y)
                nodes.append(new_node)
            elif type == "edge":
                if edges is None:
                    edges = {y: {x: Edge(0, 0) for x in nodes} for y in nodes}
                s_id, t_id, capacity = line.split(";")
                s_id, t_id, capacity = int(s_id), int(t_id), float(capacity)
                edges[nodes[s_id]][nodes[t_id]].capacity = capacity
                edges[nodes[t_id]][nodes[s_id]].capacity = capacity
                edges[nodes[t_id]][nodes[s_id]].flow = capacity
                edges[nodes[t_id]][nodes[s_id]].residual = True
                edge_count += 1

    return graphs


file_path = 'graphs.txt'
graphs = load_graphs(file_path)

G = nx.DiGraph()
for g in graphs:
    for node in g.nodes:
        G.add_node(node.id)

    for u in g.nodes:
        for v in g.nodes:
            if g.edges[u][v].capacity - g.edges[u][v].flow > 0:
                G.add_edge(u.id, v.id, capacity=g.edges[u][v].capacity, flow=g.edges[u][v].flow)

    networkx_max_flow = nx.maximum_flow(G, g.meta_data["s"], g.meta_data["t"],
                                        flow_func=nx.flow.shortest_augmenting_path)
    print(f"NetworkX Max Flow: {networkx_max_flow[0]}")