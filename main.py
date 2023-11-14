from random import random, choice
import math


class Edge:
    def __init__(self, identifier, source=None, target=None, upper_cap=2, cap=None):
        self.id = identifier
        self.source = source
        self.target = target
        if cap is None:
            self.capacity = random() * upper_cap
        else:
            self.capacity = cap


class Node:
    def __init__(self, identifier, x=.0, y=.0, distance=math.inf):
        self.id = identifier
        self.distance = distance
        self.x = x
        self.y = y
        self.edges = {}
        self.father_node = None


class Graph:
    def __init__(self, nodes=None, edges=None):
        if edges is None:
            edges = []
        if nodes is None:
            nodes = []
        self.nodes = nodes
        self.edges = edges

    def reset(self):
        for node in self.nodes:
            node.distance = math.inf
            node.father_node = None


def generate_sink_source_graph(n, r, upper_cap):
    g = Graph()
    nodes = [Node(identifier=_, x=random(), y=random()) for _ in range(n)]
    edge_id = 0
    edges = []
    for n1 in nodes:
        for n2 in nodes:
            if (n1 != n2) and ((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2 <= r ** 2):

                if random() < .5:
                    if (n2 not in n1.edges.keys()) and (n1 not in n2.edges.keys()):
                        new_edge = Edge(edge_id, upper_cap=upper_cap, source=n1, target=n2)
                        n1.edges[n2] = new_edge
                        edge_id += 1
                        edges.append(new_edge)
                else:
                    if (n2 not in n1.edges.keys()) and (n1 not in n2.edges.keys()):
                        new_edge = Edge(edge_id, upper_cap=upper_cap, source=n2, target=n1)
                        n2.edges[n1] = new_edge
                        edge_id += 1
                        edges.append(new_edge)

    g.nodes = nodes
    g.edges = edges

    return g


def bfs(s: Node, target=None):
    source = s
    # I am not currently adding all vertices because this will mostly be used to create target from source
    q = [source]

    distance = {s: 0}
    visited = []

    while len(q) > 0:
        current = q.pop(0)
        visited.append(current)
        for neighbour in current.edges.keys():
            if neighbour not in visited:
                q.append(neighbour)
                if neighbour not in distance.keys():
                    distance[neighbour] = distance[current] + 1
                    neighbour.father = current
                else:
                    if distance[current] + 1 < distance[neighbour]:
                        distance[neighbour] = distance[current] + 1
                        neighbour.father = current

    if target is None:
        target = max(distance, key=distance.get())

    return distance, target


def random_source_target(g: Graph):
    source = choice(g.nodes)
    distances, target = bfs(source)
    return source, target


def dijkstra(g: Graph, s: Node, t: Node):
    source = s
    q = [source]  # I am not currently adding all vertices, this can become a problem down the line

    for node in g.nodes:
        if node != source:
            q.append(node)

    source.distance = 0
    visited = []
    while len(q) > 0:
        current = min(q, key=lambda a: a.distance)
        visited.append(current)
        for neighbour in current.edges.keys():
            if neighbour.distance > current.distance + 1:
                neighbour.distance = current.distance + 1
                neighbour.father = current
            if neighbour == t:
                return True
            if neighbour not in visited:
                q.append(neighbour)

    return False


if __name__ == '__main__':
    n_values = [100, 200]
    r_values = [.2, .3]
    upper_cap_values = [2, 5]

    graphs = []
    for n in n_values:
        for r in r_values:
            for c in upper_cap_values:
                g = generate_sink_source_graph(n, r, c)
                graphs.append(g)
    print("test ok")
