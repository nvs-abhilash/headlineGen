graph = {
    "a": ["c"],
    "b": ["c", "e"],
    "c": ["a", "b", "d", "e"],
    "d": ["c"],
    "e": ["c", "b"],
    "f": []
}

def generateEdges (graph):
    edges = []
    for node in graph:
        for neighbour in graph[node]:
            edges.append ((node, neighbour))
    return edges

def findIsolatedNodes (graph):
    """ Returns a list of isolated nodes. """
    isolated = []

    for node in graph:
        if not graph[node]:
            isolated += node
    return isolated


print (generateEdges (graph))
print (findIsolatedNodes (graph))
