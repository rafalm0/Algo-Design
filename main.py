import os.path
from random import random, choice, uniform
import matplotlib.pyplot as plt
import math
import networkx as nx


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
            self.nodes = []
            for node_id in range(n):
                self.nodes.append(Node(node_id, uniform(0, 1), uniform(0, 1)))

            self.edges = {y: {x: Edge(0, 0) for x in self.nodes} for y in self.nodes}  # dict node to node
            self.generate_sink_source_graph(r, upperCap)
            self.meta_data = {"n": n, 'r': r, 'c': upperCap}
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

    def generate_sink_source_graph(self, r, upperCap):
        for vertex in self.nodes:
            for vertex_2 in self.nodes:
                if vertex.id != vertex_2.id:
                    comp_x = (vertex.x - vertex_2.x) ** 2
                    comp_y = (vertex.y - vertex_2.y) ** 2
                    if comp_x + comp_y <= r ** 2:
                        rand = random()
                        if rand < .5:
                            if self.edges[vertex][vertex_2].capacity == 0:
                                if self.edges[vertex_2][vertex].capacity == 0:
                                    capacity = uniform(1, upperCap)
                                    self.link_nodes(vertex, vertex_2, capacity)
                        else:
                            if self.edges[vertex_2][vertex].capacity == 0:
                                if self.edges[vertex][vertex_2].capacity == 0:
                                    capacity = uniform(1, upperCap)
                                    self.link_nodes(vertex_2, vertex, capacity)
        return True

    def find_farthest_node(self, source):
        visited = set()
        max_distance = float('-inf')
        target = source

        q = [(source, 0)]

        while q:
            current_node, distance = q.pop(0)
            visited.add(current_node)

            if distance > max_distance:
                max_distance = distance
                target = current_node

            for neighbor in self.edges[current_node].keys():
                if neighbor not in visited:
                    edge = self.edges[current_node][neighbor]
                    if edge.capacity - edge.flow > 0:
                        q.append((neighbor, distance + 1))

        return target, max_distance

    def max_supported_flow(self, path):
        capacities = [self.edges[path[i]][path[i + 1]].capacity -
                      self.edges[path[i]][path[i + 1]].flow
                      for i in range(len(path) - 1)]

        return min(capacities)

    def update_residuals(self, path, flow):
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            self.edges[a][b].flow += flow
            self.edges[b][a].flow -= flow
            # checking for errors:
            if (self.edges[a][b].flow > self.edges[a][b].capacity) or (self.edges[b][a].flow < 0):
                raise ValueError("error")
            # end check
        return True

    def SAP_DFS_RANDOM(self, source, target, method='SAP'):
        visited = set()
        distance = [math.inf for n in range(len(self.nodes))]
        fathers = {n: None for n in self.nodes}
        distance[source.id] = 0

        q = [(source, 0)]
        visited.add(source)
        while q:
            if method == 'SAP':
                current, current_distance = q.pop()
            elif method == 'DFS':
                current, current_distance = q.pop(0)
            elif method == 'RANDOM':
                current, current_distance = q.pop(choice(range(len(q))))
            else:
                print("wrong input for method")
                raise ValueError

            for neighbor, edge in self.edges[current].items():

                if self.edges[current][neighbor].capacity - self.edges[current][neighbor].flow > 0:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        fathers[neighbor] = current
                        if current_distance + 1 < distance[neighbor.id]:
                            distance[neighbor.id] = current_distance + 1
                        q.append((neighbor, distance[neighbor.id]))

        path = []
        current = target
        if fathers[target] is None:
            return [], distance[target.id]

        while current != source:
            path.append(current)
            current = fathers[current]

        path.append(source)

        return path[::-1], distance[target.id]

    def MAXCAP(self, source, target, method=None):
        cap_until = [0 for n in range(len(self.nodes))]
        fathers = {n: None for n in self.nodes}
        cap_until[source.id] = math.inf

        q = [(source, math.inf)]
        while q:

            current, current_cap = max(q, key=lambda a: a[1])
            q.remove((current, current_cap))

            for neighbor, edge in self.edges[current].items():
                if self.edges[current][neighbor].capacity - self.edges[current][neighbor].flow > 0:
                    max_cap_to_neighbour = min(edge.capacity, current_cap)

                    if max_cap_to_neighbour > cap_until[neighbor.id]:
                        cap_until[neighbor.id] = max_cap_to_neighbour
                        fathers[neighbor] = current
                        q.append((neighbor, max_cap_to_neighbour))

        path = []
        current = target
        if fathers[target] is None:
            return [], cap_until[target.id]

        while current != source:
            path.append(current)
            current = fathers[current]

        path.append(source)

        return path[::-1], cap_until[target.id]

    def ford_fulkerson(self, source, target, method='SAP'):
        if method == "MAXCAP":
            func = self.MAXCAP
        else:
            func = self.SAP_DFS_RANDOM
        max_flow = 0
        path, distance = func(source, target, method)

        while len(path) > 1:
            capacity = self.max_supported_flow(path)
            # print(f"augmenting path in {capacity}")  # uncomment if desires to see all augments
            self.update_residuals(path, capacity)
            max_flow += capacity

            path, distance = func(source, target, method)

        return max_flow


def save_graphs(graphs, path):
    with open(path, 'w') as f:
        for graph in graphs:
            s = graph.meta_data['s']
            t = graph.meta_data['t']
            target_distance = graph.meta_data['target_distance']
            n = graph.meta_data['n']
            r = graph.meta_data['r']
            c = graph.meta_data['c']  # upperCap

            for node in g.nodes:
                line = f"node:{node.id};{node.x};{node.y}\n"
                f.write(line)

            for n1 in g.nodes:
                for n2 in g.nodes:
                    if g.edges[n1][n2].capacity - g.edges[n1][n2].flow > 0:
                        line = f"edge:{n1.id};{n2.id};{g.edges[n1][n2].capacity}\n"
                        f.write(line)

            f.write(f"graph:{s.id};{t.id};{target_distance};{n};{r};{c}\n")

    return True


def load_graphs(path):
    graphs = []
    with open(path, 'r') as f:
        nodes = []
        edges = None
        for line in f.readlines():
            type, line = line.split(":")

            if type == "graph":
                source_id, target_id, target_distance, n, r, c = line.split(";")

                source_id, target_id, target_distance, n, r, c = int(source_id), int(target_id), int(
                    target_distance), int(n), float(r), int(c)

                g = Graph(None, None, None, external=True)
                meta_data = {"s": source_id, 't': target_id, 'target_distance': target_distance, 'n': n, 'r': r, 'c': c}
                g.initialize(nodes, edges, meta_data)

                graphs.append(g)

                nodes = []
                edges = None

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

    return graphs


if __name__ == '__main__':
    file_path = 'graphs.txt'
    output_path = 'logs.csv'
    n_values = [100, 200]
    r_values = [.2, .3]
    upper_cap_values = [2, 5]

    graphs = []
    if not os.path.exists(file_path):
        for n in n_values:
            for r in r_values:
                for c in upper_cap_values:
                    print("------------------------------------------------------------------------------------")
                    print(f"Graph n:{n} , r:{r}, c:{c}")
                    g = Graph(n, r, c)

                    source = g.nodes[0]
                    target, distance = g.find_farthest_node(source)

                    print(f"Got distance of {distance} for node {target.id}")

                    shortest_path, target_distance = g.SAP_DFS_RANDOM(source, target, method='SAP')
                    g.reset()

                    print("Shortest Path:", [node.id for node in shortest_path])
                    print("Shortest Distance:", target_distance)
                    my_max_flow = g.ford_fulkerson(source, target, method='MAXCAP')
                    print(f"My implementation Max Flow: {my_max_flow}")
                    g.reset()
                    g.meta_data['s'] = source
                    g.meta_data['t'] = target
                    g.meta_data['target_distance'] = target_distance
                    graphs.append(g)

                    # -------------------------------------------------------------------------------------------
                    # networkX being used ONLY for VALIDATION
                    G = nx.DiGraph()

                    for node in g.nodes:
                        G.add_node(node.id)

                    for u in g.nodes:
                        for v in g.nodes:
                            if g.edges[u][v].capacity - g.edges[u][v].flow > 0:
                                G.add_edge(u.id, v.id, capacity=g.edges[u][v].capacity, flow=g.edges[u][v].flow)

                    networkx_max_flow = nx.maximum_flow(G, source.id, target.id,
                                                        flow_func=nx.flow.shortest_augmenting_path)
                    print(f"NetworkX Max Flow: {networkx_max_flow[0]}")

                    if networkx_max_flow != my_max_flow:
                        print("VALUES DIFFER ON THIS GRAPH, FURTHER ANALYSIS NEEDED")
                    # -------------------------------------------------------------------------------------------

        save_graphs(graphs, file_path)
    else:
        graphs = load_graphs(file_path)
