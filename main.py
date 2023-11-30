import os.path
from random import random, choice, uniform
import math

nx_imported = False  # the code will only use networkx to validate results but if needed to disable, just comment below

# ----------------------------------------------------------------
try:
    import networkx as nx

    nx_imported = True
except ImportError:
    pass


# ----------------------------------------------------------------

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

    def generate_sink_source_graph(self, r, upperCap):
        edge_count = 0
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
                                    edge_count += 1
                                    capacity = uniform(1, upperCap)
                                    self.link_nodes(vertex, vertex_2, capacity)
                        else:
                            if self.edges[vertex_2][vertex].capacity == 0:
                                if self.edges[vertex][vertex_2].capacity == 0:
                                    edge_count += 1
                                    capacity = uniform(1, upperCap)
                                    self.link_nodes(vertex_2, vertex, capacity)
        self.meta_data['edge_count'] = edge_count
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
        error_margin = 0.00000000001
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            self.edges[a][b].flow += flow
            self.edges[b][a].flow = self.edges[a][b].capacity - self.edges[a][b].flow
            # checking for errors:
            if (self.edges[a][b].flow > (self.edges[a][b].capacity + error_margin)) or (
                    (self.edges[b][a].flow + error_margin) < 0):
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
        self.reset()
        if method == "MAXCAP":
            func = self.MAXCAP
        else:
            func = self.SAP_DFS_RANDOM
        max_flow = 0
        path, distance = func(source, target, method)
        self.meta_data['augmenting_path_count'] = 0
        self.meta_data['augmenting_path_length'] = []
        while len(path) > 1:
            self.meta_data['augmenting_path_count'] = self.meta_data['augmenting_path_count'] + 1
            self.meta_data['augmenting_path_length'].append(len(path))
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

            for node in graph.nodes:
                line = f"node:{node.id};{node.x};{node.y}\n"
                f.write(line)

            for n1 in graph.nodes:
                for n2 in graph.nodes:
                    if graph.edges[n1][n2].capacity - graph.edges[n1][n2].flow > 0:
                        line = f"edge:{n1.id};{n2.id};{graph.edges[n1][n2].capacity}\n"
                        f.write(line)

            f.write(f"graph:{s};{t};{target_distance};{n};{r};{c}\n")

    return True


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


if __name__ == '__main__':
    file_path = 'graphs.txt'
    output_path = 'logs_exp1.csv'
    file_path_experiment_2 = 'graphs_2.txt'
    output_path_experiment_2 = 'logs_exp2.csv'

    #  values for experiment 1
    n_values = [100, 200]
    r_values = [.2, .3]
    upper_cap_values = [2, 50]
    enable_visualization = True

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

                    print("Shortest Path:", [node.id for node in shortest_path])
                    print("Shortest Distance:", target_distance)
                    my_max_flow = g.ford_fulkerson(source, target, method='MAXCAP')
                    print(f"My implementation Max Flow: {my_max_flow}")
                    g.reset()
                    g.meta_data['s'] = source.id
                    g.meta_data['t'] = target.id
                    g.meta_data['target_distance'] = target_distance
                    graphs.append(g)

                    # -------------------------------------------------------------------------------------------
                    # networkX being used ONLY for VALIDATION
                    if nx_imported:
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

                        if round(networkx_max_flow[0], 12) != round(my_max_flow, 12):  # checking up to 12 decimals
                            print("VALUES DIFFER ON THIS GRAPH, FURTHER ANALYSIS NEEDED")
                    # -------------------------------------------------------------------------------------------

        save_graphs(graphs, file_path)
    else:
        graphs = load_graphs(file_path)

    print("Starting experiment 1...")

    logs = []
    for i, graph in enumerate(graphs):
        s = graph.nodes[graph.meta_data['s']]
        t = graph.nodes[graph.meta_data['t']]
        target_distance = graph.meta_data['target_distance']
        n = graph.meta_data['n']
        r = graph.meta_data['r']
        c = graph.meta_data['c']  # upperCap
        print(f"-----------------------------------------")
        print(f"Experiment on graph {i + 1}/{len(graphs)}")

        print(f"n:{n}   r:{r}   upperCap:{c}")
        for method in ['SAP', 'DFS', 'MAXCAP', 'RANDOM']:
            paths = 0  # number of augmenting paths
            ML = 0  # avg of all augmenting paths
            MPL = 0  # ML / longest acyclic path from s to t(recorded in "max_dist")
            total_edges = graph.meta_data['edge_count']
            my_max_flow = graph.ford_fulkerson(s, t, method=method)
            paths = graph.meta_data['augmenting_path_count']
            ML = sum(graph.meta_data['augmenting_path_length'])

            ML = ML / paths
            MPL = ML / target_distance
            logs.append([method, n, r, c, paths, ML, MPL, total_edges])
            graph.reset()
            print(f"\t Method: {method}\t Flow: {my_max_flow}")

    print("All experiments finished, saving logs on csv...")
    with open(output_path, 'w') as f:
        f.write('method,n,r,c,paths,ML,MPL,total_edges\n')
        f.writelines([','.join([str(v) for v in x]) + '\n' for x in logs])

    print("Starting experiment 2...")
    #  values for experiment 2

    n_ = [300, 300]
    r_ = [.3, .3]
    c_ = [1, 200]
    logs = []

    if not os.path.exists(file_path_experiment_2):
        graphs = []
        for i in range(len(n_)):
            n = n_[i]
            r = r_[i]
            c = c_[i]
            print("------------------------------------------------------------------------------------")
            print(f"Graph n:{n} , r:{r}, c:{c}")
            g = Graph(n, r, c)

            source = g.nodes[0]
            target, distance = g.find_farthest_node(source)

            print(f"Got distance of {distance} for node {target.id}")

            shortest_path, target_distance = g.SAP_DFS_RANDOM(source, target, method='SAP')

            print("Shortest Path:", [node.id for node in shortest_path])
            print("Shortest Distance:", target_distance)
            my_max_flow = g.ford_fulkerson(source, target, method='MAXCAP')
            print(f"My implementation Max Flow: {my_max_flow}")
            g.reset()
            g.meta_data['s'] = source.id
            g.meta_data['t'] = target.id
            g.meta_data['target_distance'] = target_distance
            graphs.append(g)
        save_graphs(graphs, file_path_experiment_2)
    else:
        graphs = load_graphs(file_path_experiment_2)

    for g in graphs:
        s = g.nodes[g.meta_data['s']]
        t = g.nodes[g.meta_data['t']]
        target_distance = g.meta_data['target_distance']
        n = g.meta_data['n']
        r = g.meta_data['r']
        c = g.meta_data['c']  # upperCap
        print(f"-----------------------------------------")
        print(f"Experiment on graph {i + 1}/{len(graphs)}")

        print(f"n:{n}   r:{r}   upperCap:{c}")
        for method in ['SAP', 'DFS', 'MAXCAP', 'RANDOM']:
            paths = 0  # number of augmenting paths
            ML = 0  # avg of all augmenting paths
            MPL = 0  # ML / longest acyclic path from s to t(recorded in "max_dist")
            total_edges = g.meta_data['edge_count']

            my_max_flow = g.ford_fulkerson(s, t, method=method)
            paths = g.meta_data['augmenting_path_count']
            ML = sum(g.meta_data['augmenting_path_length'])

            ML = ML / paths
            MPL = ML / target_distance
            logs.append([method, n, r, c, paths, ML, MPL, total_edges])
            g.reset()
            print(f"\t Method: {method}\t Flow: {my_max_flow}")

    print("All experiments finished, saving logs on csv...")

    with open(output_path_experiment_2, 'w') as f:
        f.write('method,n,r,c,paths,ML,MPL,total_edges\n')
        f.writelines([','.join([str(v) for v in x]) + '\n' for x in logs])
