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

    def __init__(self, n, r, upperCap):
        self.nodes = []
        for node_id in range(n):
            self.nodes.append(Node(node_id, uniform(0, 1), uniform(0, 1)))

        self.edges = {y: {x: Edge(0, 0) for x in self.nodes} for y in self.nodes}  # dict node to node
        self.generate_sink_source_graph(r, upperCap)

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

    def SAP_DFS(self, source, target, BFS=True):
        visited = set()
        distance = [math.inf for n in range(len(self.nodes))]
        fathers = {n: None for n in self.nodes}
        distance[source.id] = 0

        q = [(source, 0)]
        visited.add(source)
        while q:
            if BFS:
                current, current_distance = q.pop()
            else:
                current, current_distance = q.pop(0)

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

    def MAXCAP(self, source, target):
        visited = set()
        distance = [math.inf for n in range(len(self.nodes))]
        fathers = {n: None for n in self.nodes}
        distance[source.id] = 0

        q = [(source, 0)]
        visited.add(source)
        while q:
            current, current_distance = min(q, key=lambda a: a[1])
            q.remove((current, current_distance))

            if current == target:
                break

            for neighbor, edge in self.edges[current].items():
                if self.edges[current][neighbor].capacity - self.edges[current][neighbor].flow > 0:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if current_distance + 1 < distance[neighbor.id]:
                            distance[neighbor.id] = current_distance + 1
                            fathers[neighbor] = current
                        q.append((neighbor, distance[neighbor.id]))
        else:
            print("No augmenting path found")
            return [], distance[target.id]

        path = []
        current = target

        while current != source:
            path.append(current)
            current = fathers[current]

        path.append(source)

        return path[::-1], distance[target.id]

    def ford_fulkerson(self, source, target, method='SAP'):
        methods = {'MAXCAP': None, 'SAP': self.SAP_DFS, 'DFS': None, 'RANDOM': None}
        max_flow = 0
        path, distance = methods[method](source, target)

        while len(path) > 1:
            capacity = self.max_supported_flow(path)
            print(f"augmenting path in {capacity}")
            self.update_residuals(path, capacity)
            max_flow += capacity

            path, distance = methods[method](source, target)

        return max_flow


g = Graph(100, 0.2, 2)

source = g.nodes[0]
target, distance = g.find_farthest_node(source)

if source == target:
    print("Warning, Source was generated isolated")
    quit()
print(f"got distance of {distance} for node {target.id}")

shortest_path, shortest_distance = g.SAP_DFS(source, target)

G = nx.DiGraph()

for node in g.nodes:
    G.add_node(node.id)

for u in g.nodes:
    for v in g.nodes:
        if g.edges[u][v].capacity > 0:
            G.add_edge(u.id, v.id, capacity=g.edges[u][v].capacity, flow=g.edges[u][v].flow)

print("Shortest Path:", [node.id for node in shortest_path])
print("Shortest Distance:", shortest_distance)

networkx_max_flow = nx.maximum_flow(G, source.id, target.id, flow_func=nx.flow.shortest_augmenting_path)
# Compare the results
print(f"Your Implementation Max Flow: {g.ford_fulkerson(source, target)}")
print(f"NetworkX Max Flow: {networkx_max_flow}")

# Draw the graph
# pos = {node.id: (node.x, node.y) for node in g.nodes}
# edge_labels = {(u, v): f"{round(d['flow'], 2)}/{round(d['capacity'], 2)}" for u, v, d in G.edges(data=True)}
# node_labels = {node.id: node.id for node in g.nodes}
#
# nx.draw(G, pos, with_labels=True, labels=node_labels, font_weight='bold', node_size=700, node_color='lightblue')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
#
# plt.show()
