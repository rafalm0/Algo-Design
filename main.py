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
        self.father = None


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
            node.father = None


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
    source.father = source
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

        max_dist = 0
        for n in distance.keys():
            dist = distance[n]
            if dist != math.inf:
                if dist > max_dist:
                    max_dist = dist
                    target = n

    return distance, target


def random_source_target(g: Graph):
    source = choice(g.nodes)
    distances, target = bfs(source)
    return source, target


def get_path_found(g: Graph, s: Node, t: Node):
    max_path = len(g.nodes)

    current = t
    path_followed = []
    for _ in range(max_path):
        if current == s:
            break
        path_followed.append(current)
        current = current.father

    else:
        if current != s:
            print(f"Path is broken")
            return None

    return path_followed[::-1]


def dijkstra_SAP(g: Graph, s: Node, t: Node):
    source = s
    q = [source]  # I am not currently adding all vertices, this can become a problem down the line

    for node in g.nodes:
        if node != source:
            q.append(node)

    source.distance = 0
    visited = []
    while len(q) > 0:
        current = min(q, key=lambda a: a.distance)
        q.remove(current)
        visited.append(current)
        for neighbour in current.edges.keys():
            if neighbour.distance > current.distance + 1:
                neighbour.distance = current.distance + 1
                neighbour.father = current
            if neighbour == t:
                return True
            if (neighbour not in visited) and (neighbour not in q):
                q.append(neighbour)

    return False


def dijkstra_DFS(g: Graph, s: Node, t: Node):
    source = s
    q = [source]  # I am not currently adding all vertices, this can become a problem down the line

    counter = 9999999999  # big number so it will ever get to 0 unless infinite loop
    for node in g.nodes:
        if node != source:
            q.append(node)

    source.distance = 0
    visited = []
    while len(q) > 0:
        current = min(q, key=lambda a: a.distance)
        q.remove(current)
        visited.append(current)
        for neighbour in current.edges.keys():
            if neighbour == t:  # found the path
                return True

            if neighbour.distance == math.inf:  # found another not relaxed node
                neighbour.distance = counter
                counter -= 1
                neighbour.father = current
            elif neighbour.distance > current.distance + 1:
                neighbour.distance = current.distance + 1
                neighbour.father = current

            if (neighbour not in visited) and (neighbour not in q):
                q.append(neighbour)

    return False

def dijkstra_RANDOM(g: Graph, s: Node, t: Node):
    source = s
    q = [source]  # I am not currently adding all vertices, this can become a problem down the line

    counter = 9999999999  # big number so it will ever get to 0 unless infinite loop
    for node in g.nodes:
        if node != source:
            q.append(node)

    source.distance = 0
    visited = []
    while len(q) > 0:
        current = min(q, key=lambda a: a.distance)
        q.remove(current)
        visited.append(current)
        for neighbour in current.edges.keys():
            if neighbour == t:  # found the path
                return True

            if neighbour.distance == math.inf:  # found another not relaxed node
                neighbour.distance = int(random()*counter)
                neighbour.father = current
            elif neighbour.distance > current.distance + 1:
                neighbour.distance = current.distance + 1
                neighbour.father = current

            if (neighbour not in visited) and (neighbour not in q):
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

                s, t = random_source_target(g)
                found_path = dijkstra_SAP(g, s, t)
                if not found_path:
                    print("could not find path in SAP")
                path = get_path_found(g, s, t)
                found_path_2 = dijkstra_DFS(g, s, t)
                if not found_path_2:
                    print("could not find path in DFS like")
                found_path_3 = dijkstra_RANDOM(g, s, t)
                if not found_path_3:
                    print("could not find path in RANDOM")
                graphs.append((g, s, t, found_path, path))
    print("test ok")
