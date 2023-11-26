from random import random, choice
import math


class Edge:
    def __init__(self, identifier, source=None, target=None, upper_cap=2, cap=None, used_cap=0):
        self.id = identifier
        self.source = source
        self.target = target
        if cap is None:
            self.capacity = random() * upper_cap
        else:
            self.capacity = cap
        self.used_cap = used_cap


class Node:
    def __init__(self, identifier, x=.0, y=.0, distance=math.inf):
        self.id = identifier
        self.distance = distance
        self.x = x
        self.y = y
        self.edges = {}
        self.father = None


class Graph:
    def __init__(self, nodes=None, edges=None, sink=None, source=None):
        if edges is None:
            edges = []
        if nodes is None:
            nodes = []
        self.nodes = nodes
        self.edges = edges
        self.sink = sink
        self.source = source

    def reset_path(self):
        for node in self.nodes:
            node.distance = math.inf
            node.father = None
        self.source.distance = 0

    def reset_capacities(self):
        for edge in self.edges:
            if edge.id < 0:
                edge.used_cap = edge.capacity
            else:
                edge.used_cap = 0
        return

    # TODO fix this fill backwards
    def fill_backwards_path(self, sink: Node = None,
                            target: Node = None):  # this concretes the path and makes it permanent for
        # next searches (residual net)
        if sink is None:
            current = self.sink  # here we do the opposite since we are going backwards
        else:
            current = sink

        if target is None:
            final = self.source  # here the final is the source because we are going the opposite way
        else:
            final = target

        if (current is None) or (final is None):
            raise EnvironmentError(f"Error:There is no sink or target set or given, no path can be calculated")

        path_size = 1

        max_flow = math.inf
        edges_to_update = []
        # now we find the max flow that the path calculated can handle
        while current != final:
            if current.father is None: raise ValueError("path found broke ")

            edge_to_father = current.father.edges[current]  # here the current is the target instead of source
            if (edge_to_father.capacity - edge_to_father.used_cap) < max_flow:
                max_flow = edge_to_father.capacity - edge_to_father.used_cap
                edges_to_update.append(edge_to_father)
            current = current.father
            path_size += 1

        for edge in edges_to_update:
            edge.used_cap += max_flow
            if edge.source not in edge.target.edges.keys():  # check if the other path exists
                new_residual_edge = Edge(identifier=edge.id * -1,
                                         residual=True,
                                         source=edge.target,
                                         target=edge.source,
                                         cap=edge.capacity,
                                         used_cap=edge.capacity - max_flow)  # create other way around
                graph.edges.append(new_residual_edge)
                target.edges[edge.source] = new_residual_edge

            else:
                inverted_edge = edge.target.edges[edge.source]
                inverted_edge.used_cap = edge.capacity - edge.used_cap
                if inverted_edge.used_cap < -0.00000000000001: raise ValueError(f"Residual path fell below zero: {inverted_edge.used_cap}")

        return path_size


def generate_graph(n, r, upper_cap, reverts=False):
    g = Graph()
    nodes = [Node(identifier=_, x=random(), y=random()) for _ in range(n)]
    edge_id = 0
    edges = []
    for n1 in nodes:
        for n2 in nodes:
            if (n1 != n2) and ((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2 <= r ** 2):

                if random() < .5:
                    if (n2 not in n1.edges.keys()) and (n1 not in n2.edges.keys()):
                        new_edge = Edge(edge_id,
                                        upper_cap=upper_cap,
                                        source=n1,
                                        target=n2)

                        n1.edges[n2] = new_edge
                        edge_id += 1
                        edges.append(new_edge)
                        if reverts:
                            new_edge_revert = Edge(edge_id * -1,
                                                   cap=new_edge.capacity,
                                                   source=n2,
                                                   target=n1,
                                                   used_cap=new_edge.capacity)
                            n2.edges[n1] = new_edge_revert
                            edges.append(new_edge_revert)

                else:
                    if (n2 not in n1.edges.keys()) and (n1 not in n2.edges.keys()):
                        new_edge = Edge(edge_id, upper_cap=upper_cap, source=n2, target=n1)
                        n2.edges[n1] = new_edge
                        edge_id += 1
                        edges.append(new_edge)
                        if reverts:
                            new_edge_revert = Edge(edge_id * -1,
                                                   cap=new_edge.capacity,
                                                   source=n1,
                                                   target=n2,
                                                   used_cap=new_edge.capacity)
                            n1.edges[n2] = new_edge_revert
                            edges.append(new_edge_revert)

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
                if current.edges[neighbour].used_cap == current.edges[neighbour].capacity:
                    continue  # this is either a residual edge not to be used or is completly full
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

    return distance, target, max_dist


def random_source_target(g: Graph):
    source = choice(g.nodes)
    distances, target, max_dist = bfs(source)
    return source, target, max_dist


def get_path_found(g: Graph, s: Node, t: Node):
    max_path = len(g.nodes)

    current = t
    path_followed = []
    human_readable = []
    for _ in range(max_path):
        if current == s:
            path_followed.append(current)
            human_readable.append(f"Node {current.id} : {current.distance}")
            break
        path_followed.append(current)
        human_readable.append(f"Node {current.id} : {current.distance}")
        edge = current.father.edges[current]
        human_readable.append(f"Edge {edge.id} ({edge.source.id} : {edge.source.distance} -> {edge.capacity}-> {edge.target.id}): {edge.target.distance}")
        current = current.father

    else:
        if current != s:
            raise ValueError(f"Path is broken")

    return path_followed[::-1], human_readable[::-1]


def dijkstra_SAP(g: Graph, s: Node, t: Node):
    source = s
    q = [source]  # I am not currently adding all vertices but i add as soon as needed a few lines down.

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
            if current.edges[neighbour].used_cap == current.edges[neighbour].capacity:
                continue  # this is either a residual edge not to be used or is completly full
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
    q = [source]  # I am not currently adding all vertices but i add as soon as needed a few lines down.

    counter = 9999999999  # big number,so it will ever get to 0 unless infinite loop
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
            if current.edges[neighbour].used_cap == current.edges[neighbour].capacity:
                continue  # this is either a residual edge not to be used or is completly full
            if neighbour == t:  # found the path
                neighbour.father = current
                return True

            if neighbour.distance == math.inf:  # found another not relaxed node
                neighbour.distance = counter
                counter -= 1
                neighbour.father = current
            # elif neighbour.distance > current.distance + 1:
            #     neighbour.distance = current.distance + 1
            #     neighbour.father = current

            if (neighbour not in visited) and (neighbour not in q):
                q.append(neighbour)

    return False


def dijkstra_RANDOM(g: Graph, s: Node, t: Node):
    source = s
    q = [source]  # I am not currently adding all vertices but i add as soon as needed a few lines down.

    counter = 9999999999  # big number,so it will ever get to 0 unless infinite loop
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
            if current.edges[neighbour].used_cap == current.edges[neighbour].capacity:
                continue  # this is either a residual edge not to be used or is completly full
            if neighbour == t:  # found the path
                neighbour.father = current
                return True

            if neighbour.distance == math.inf:  # found another not relaxed node
                neighbour.distance = int(random() * counter)
                neighbour.father = current
            # elif neighbour.distance > current.distance + 1:
            #     neighbour.distance = current.distance + 1
            #     neighbour.father = current

            if (neighbour not in visited) and (neighbour not in q):
                q.append(neighbour)

    return False


def dijkstra_MAXCAP(g: Graph, s: Node, t: Node):
    source = s
    q = [source]  # I am not currently adding all vertices but i add as soon as needed a few lines down.

    for node in g.nodes:
        if node != source:
            node.distance = 0  # here the distance will behave like capacity and we will look for the max value
            q.append(node)

    source.distance = math.inf  # the source has maximum capacity
    visited = []
    while len(q) > 0:
        current = max(q, key=lambda a: a.distance)
        q.remove(current)
        visited.append(current)
        for neighbour in current.edges.keys():
            edge = current.edges[neighbour]
            edge_cap = edge.capacity - edge.used_cap  # used cap represent flow already set as being used by other
            # iterations of this function (this would be run more than once)
            if edge_cap == 0:
                continue  # this path was already exhausted in previous iterations
            # here we see what would be the max deliverable flow to that neighbour is
            if current.distance > edge_cap:
                max_to_neighbour = edge_cap
            else:
                max_to_neighbour = current.distance

            if neighbour.distance < max_to_neighbour:
                neighbour.distance = max_to_neighbour
                neighbour.father = current
            if neighbour == t:
                return True
            if (neighbour not in visited) and (neighbour not in q):
                q.append(neighbour)

    return False


methods = {'MAXCAP': dijkstra_MAXCAP, 'RANDOM': dijkstra_RANDOM, 'DFS': dijkstra_DFS, 'SAP': dijkstra_SAP}

if __name__ == '__main__':
    # n_values = [100, 200]
    # r_values = [.2, .3]
    # upper_cap_values = [2, 5]

    n_values = [100]
    r_values = [.3]
    upper_cap_values = [5]
    graphs = []
    for n in n_values:
        for r in r_values:
            for c in upper_cap_values:
                g = generate_graph(n, r, c,reverts=True)

                s, t, max_dist = random_source_target(g)
                while t is None:
                    print(f"Graph generated with source alone, no target possible, re-generating graph...")
                    g = generate_graph(n, r, c, reverts=True)
                    s, t, max_dist = random_source_target(g)
                g.source = s
                g.sink = t

                found_path = dijkstra_SAP(g, s, t)
                if not found_path:
                    raise RuntimeError(f"could not find path in SAP")
                path, readable = get_path_found(g, s, t)
                g.reset_path()
                g.reset_capacities()

                found_path_2 = dijkstra_DFS(g, s, t)
                if not found_path_2:
                    raise RuntimeError(f"could not find path in DFS like")
                path2, readable2 = get_path_found(g, s, t)
                g.reset_path()
                g.reset_capacities()

                found_path_3 = dijkstra_RANDOM(g, s, t)
                if not found_path_3:
                    raise RuntimeError(f"could not find path in RANDOM")
                path3, readable3 = get_path_found(g, s, t)
                g.reset_path()
                g.reset_capacities()

                found_path_4 = dijkstra_MAXCAP(g, s, t)
                if not found_path_4:
                    raise RuntimeError(f"could not find path in MAXCAP")
                path4, readable4 = get_path_found(g, s, t)  # the problem is we are trying to find max weight in
                # djikstra and now it has cycles since we put augmenting path
                g.reset_path()
                g.reset_capacities()

                graphs.append((g, s, t, max_dist, n, r, c))
    print("Test ok, starting experiments...")

    logs = []

    for method in methods.keys():
        print(f"Experiments with {method}")
        for graph_id, (graph, source, target, max_dist, n, r, c) in enumerate(graphs):
            print(f"Experiments on graph {graph_id+1}/{len(graphs)}...")
            paths = 0  # number of augmenting paths
            ML = 0  # avg of all augmenting paths
            MPL = 0  # ML / longest acyclic path from s to t(recorded in "max_dist")
            total_edges = len(graph.edges) / 2  # divide by two to remove the reverses of each edge
            while True:
                flow_incremented = methods[method](graph, source, target)
                if flow_incremented:
                    path, readable = get_path_found(graph, source, target)
                    path_size = graph.fill_backwards_path()  # updates the graph and count path's size
                    graph.reset_path()
                    ML += path_size
                    paths += 1
                else:
                    break  # its already max-flow
            ML = ML / paths
            MPL = ML / max_dist
            method_name = method
            logs.append([method_name, n, r, c, paths, ML, MPL, total_edges])
            graph.reset_path()
            graph.reset_capacities()

    print("All experiments finished.")
    pass
