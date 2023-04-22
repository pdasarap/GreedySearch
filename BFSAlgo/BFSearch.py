import numpy as np
import networkx as ntx
import matplotlib.pyplot as mplt

class BFSearch:
    def BFSearch(adj_mat):
        check = False
        visit_list = [False for i in range(18)]
        nodes_visited = []
        visited_count = 0
        for i in range(len(adj_mat)):
            for j in range(18):
                # to verify edge between any 2 vertices
                if (not(visit_list[j]) and adj_mat[i][j] == 1 and (j not in nodes_visited)):
                        nodes_visited.append(j)
                        visited_count += 1
                        visit_list[j] = True
        # if all nodes are visited
        if (visited_count == 18):
            check = True

        return check

    def display_graph(adj_mat):
        rows, cols = np.where(adj_mat == 1)
        edges = zip(rows.tolist(), cols.tolist())
        graph = ntx.Graph()
        all_rows = range(0, adj_mat.shape[0])
        for n in all_rows:
            graph.add_node(n)
        graph.add_edges_from(edges)
        ntx.draw(graph, node_size=600, with_labels=True)
        mplt.show()

