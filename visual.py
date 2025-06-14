import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_graph(capacity, flow, step):
    G = nx.DiGraph()
    n = len(capacity)
    for i in range(n):
        for j in range(n):
            if capacity[i][j] > 0:
                G.add_edge(i, j, capacity=capacity[i][j], flow=flow[i][j])
    pos = {
        0: (0, 0),   
        1: (1, 1),   
        2: (1, -1),  
        3: (2, 0),  
        4: (3, 1),   
        5: (3, -1)  
    }
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=16, font_weight='bold', arrows=True)
    edge_labels = {(i, j): f"{flow[i][j]}/{capacity[i][j]}" for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12)
    plt.title(f"Step {step}: Flow Visualization")
    plt.show()
def bfs(residual_graph, source, sink):
    n = len(residual_graph)
    visited = [False] * n
    parent = [-1] * n
    queue = [source]
    visited[source] = True
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if not visited[v] and residual_graph[u][v] > 0:
                queue.append(v)
                visited[v] = True
                parent[v] = u
                if v == sink:
                    return parent
    return None
def ford_fulkerson(capacity, source, sink):
    n = len(capacity)
    residual_graph = np.array(capacity)  
    flow = np.zeros((n, n), dtype=int) 
    max_flow = 0
    step = 0
    while True:
        parent = bfs(residual_graph, source, sink)
        if parent is None:
            break  
        path_flow = float('Inf')
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual_graph[u][v])
            v = parent[v]
        v = sink
        while v != source:
            u = parent[v]
            residual_graph[u][v] -= path_flow
            residual_graph[v][u] += path_flow
            flow[u][v] += path_flow
            v = parent[v]
        max_flow += path_flow
        step += 1       
        print(f"Step {step}:")
        print("Current flow matrix:")
        print(flow)
        print("Current residual graph:")
        print(residual_graph)
        print()
        draw_graph(capacity, flow, step)
    return max_flow

n = 6  
capacity = [
    [0, 16, 13, 0, 0, 0],
    [0, 0, 10, 12, 0, 0],
    [0, 4, 0, 0, 14, 0],
    [0, 0, 9, 0, 0, 20],
    [0, 0, 0, 7, 0, 4],
    [0, 0, 0, 0, 0, 0]
]
source = 0  
sink = 5   
max_flow = ford_fulkerson(capacity, source, sink)
print("Максимальный поток:", max_flow)

