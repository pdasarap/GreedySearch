import random
import numpy as np
from graph_tools import *
from msilib.sequence import AdminExecuteSequence
from BFSearch import BFSearch
from enum import Flag
import math
from operator import itemgetter
import sys
import timeit
sys.setrecursionlimit(1_000_000)

 
nodes_list = list(range(0, 18))
adj_matrx = [[0]*18 for _ in range(18)]
#to create an adjacent matrix 18x18 and a graph of 18 nodes
global graph
graph = Graph(directed=True)
for i in nodes_list:
    graph.add_vertex(i)

def set_adj_matrix():
    nodes_list = list(range(18))
    adj_mat = [[0]*18 for _ in range(18)]
    cand_list = []

    for curr_cand in nodes_list:
        while 1:
            random_cand = random.choice(nodes_list)
            if (random_cand != curr_cand):
                if (random_cand not in cand_list):
                    cand_list.append(curr_cand)

                    #updating the adjacency matrix
                    adj_mat[curr_cand][random_cand] = 1
                    adj_mat[random_cand][curr_cand] = 1

                    #adding the edges generated randomely
                    graph.add_edge(curr_cand, random_cand)
                    graph.add_edge(random_cand, curr_cand)

            if (len(cand_list) == 3):
                cand_list = []
                break

    first_constraint_check(adj_mat)

    
def first_constraint_check(adj_mat, is_optimal=False):

    # constraint to check if graph is a complete digraph
    if not BFSearch.BFSearch(adj_mat):
        if not is_optimal:
            set_adj_matrix()
        else:
            return False

    # constraint to check if the diameter is atleast 4 
    for i in nodes_list:  
        dis, pre = graph.dijkstra(i)
        if any(4 < value for value in dis.values()):
            if not is_optimal: 
                set_adj_matrix()
            else:
                return False

    # constraint to check if degree of graph is at least 3
    for i in adj_mat:
        if i.count(1) < 3:
            if not is_optimal:
                set_adj_matrix()
            else:
                return False

    if not is_optimal:
        gen_node_coordinates(adj_mat)
    else:
        return True

def second_constraint_check(adj_mat):

    # constraint to check if graph is a complete digraph
    if not BFSearch.BFSearch(adj_mat):
        return False

    # constraint to check if the diameter is atleast 4 
    for i in nodes_list:  
        dis, pre = graph.dijkstra(i)

        if any(4 < value for value in dis.values()):
            return False

    # constraint to check if degree of graph is at least 3
    for i in adj_mat:
        if i.count(1) < 3:
            return False

    return True

# this generates the (x,y) coordinates of every node on the graph
def gen_node_coordinates(adj_mat):
    # selects coordinates randomly in range of 0-80
    coord_range = list(range(0, 80)) 
    coord_list = []
    while(len(coord_list) < 18):
        x_coord = random.choice(coord_range)
        y_coord = random.choice(coord_range)
        x_y = [x_coord, y_coord]
        if x_y not in coord_list:
            coord_list.append(x_y)

    # prints all coordinates list
    print('The coordinates of the nodes are {}'.format(coord_list))

    total_cost = total_costs(adj_mat, coord_list)


# greedy search
def greedy_search(edges_cost, opt_cost, adj_mat, coord_list):

    # the edges with max weight are removed from the graph
    edg_costs_sorted = sorted(edges_cost, key = itemgetter(2), reverse=True)
    for i in range(len(edg_costs_sorted)):
        coord_1 = edg_costs_sorted[i][0]
        coord_2 = edg_costs_sorted[i][1]
        node_1 = coord_list.index(coord_1)
        node_2 = coord_list.index(coord_2)
        
        adj_mat[node_1][node_2] = 0
        adj_mat[node_2][node_1] = 0

        # the resultant graph is checked for constraints
        if first_constraint_check(adj_mat, True):
            # if yes, calculate the total cost
            o_cost = total_costs(adj_mat, coord_list, True)

            #check if this new cost is less than the earlier cost
            if o_cost < opt_cost:
                opt_cost = o_cost 

            # Loop through all edges to remove any edges  
            for j in range(len(edg_costs_sorted)):
                coord_11 = edg_costs_sorted[j][0]
                coord_22 = edg_costs_sorted[j][1]
                node_11 = coord_list.index(coord_11)
                node_22 = coord_list.index(coord_22)
                
                adj_mat[node_11][node_22] = 0
                adj_mat[node_22][node_11] = 0
                
                # the resultant graph is checked for constraints
                if first_constraint_check(adj_mat, True):
                     # if yes, calculate the total cost
                    o_cost = total_costs(adj_mat, coord_list, True)
                    if o_cost < opt_cost:
                        opt_cost = o_cost               
                    else:
                        pass
                # if the total cost of resultant is not optimum
                # revert the removed edge
                else: 
                    adj_mat[node_11][node_22] = 1
                    adj_mat[node_22][node_11] = 1
        else:
            adj_mat[node_1][node_2] = 1
            adj_mat[node_2][node_1] = 1  
        adj_mat[node_1][node_2] = 1
        adj_mat[node_2][node_1] = 1

    # display the graph of the adjacent matrix
    mat_array = []
    mat_array = np.array(adj_mat)
    BFSearch.display_graph(mat_array)
    print('Greedy Search Optimum-cost {}' .format(opt_cost))


# heuristic            
def heuristic_search(edges_cost, final_total_cost, adj_mat, cost_mat, coord_list):

    # Let the max weight
    max_wgt = float('inf')

    # list the selected nodes to avoid redundant selections
    node_1 = [False for i in range(18)]

    res_mat = [[0]*18 for _ in range(18)]

    count = 0
    booln = False
    while(booln in node_1):
        min_wgt = max_wgt
        begin = 0
        last = 0
        for x in range(18):
            if node_1[x]:
                for y in range(18):
                    # avoid cycles in the graph
                    if (not node_1[y] and cost_mat[x][y]>0):  
                        if cost_mat[x][y] < min_wgt:
                            min_wgt = cost_mat[x][y]
                            begin, last = x, y

        node_1[last] = True

        res_mat[begin][last] = min_wgt

        if min_wgt == max_wgt:
            res_mat[begin][last] = 0

        count += 1

        # resultant matrix should have path with min cost possible
        res_mat[last][begin] = res_mat[begin][last]


    adj_mat_2 = [[0]*18 for _ in range(18)]

    for i in range(18):
        for j in range(18):
            if res_mat[i][j]!=0:
                adj_mat_2[i][j]=1

    flg = True
    # check for constraints on the resultant graph
    while not flg ==second_constraint_check(adj_mat_2):
        chk = list(map(sum, adj_mat_2))

        for i in range(18):
            for j in range(18):
                if adj_mat[i][j] != adj_mat_2[i][j]:
                    if chk[i]<3:
                        adj_mat_2[i][j]=1
                        adj_mat_2[j][i]=1
                        chk = list(map(sum, adj_mat_2))


    # now calculate the total cost of resultant graph
    o_cost = total_costs(adj_mat_2, coord_list, True)

    # display graph of resultant adjacent matrix
    mat_array = []
    mat_array = np.array(adj_mat_2)
    BFSearch.display_graph(mat_array)
    print('\n')
    print('Heuristic Search Optimum-cost {}' .format(o_cost) )


    
def total_costs(adj_mat, coord_list, is_optimal=False):

    cost_mat = [[0]*18 for _ in range(18)]

    # coordinate list
    temp_coord_list = [[x_i,y_i] for x_i, row in enumerate(adj_mat) 
                    for y_i, i in enumerate(row) if i == 1]
    distinct_coord_list = [list(i) for i in {*[tuple(sorted(i)) for i in temp_coord_list]}]


    # to calculate Euclidean distance between any two nodes
    edges = []
    for item in distinct_coord_list:
        coord = []
        for j in item:
            coord.append(coord_list[j])
        edges.append(coord)

    # to get edge weight
    edges_cost = []
    for item1 in edges:
        dis = round(np.linalg.norm(np.array(item1[0]) - 
                                    np.array(item[1])), 3)

        edges_cost.append([item1[0], item1[1], dis])

    # sum up the edge cost        
    totalcost = 0
    for i in edges_cost:
        totalcost += i[-1]
    final_totalcost = round(totalcost, 3)

    # matrix with edge costs
    x=0
    for i in range(18):
        for j in range(18):
            if i<j and adj_mat[i][j]==1:
                cost_mat[i][j] = edges_cost[x][2]
                cost_mat[j][i] = edges_cost[x][2]
                x += 1

    if not is_optimal:  
        tot_cost1 = timeit.timeit(lambda: greedy_search(edges_cost, final_totalcost, adj_mat, coord_list), number=1)
        print('Greedy Search Runtime {}ms' .format(round(tot_cost1,3)))

        tot_cost2 = timeit.timeit(lambda: heuristic_search(edges_cost, final_totalcost, adj_mat, cost_mat, coord_list), number=1)
        print('Heuristic Search Runtime {}ms' .format(round(tot_cost2,3)))

    if is_optimal:
        return final_totalcost


if __name__ == "__main__":
    
    i=0
    for i in range(5):
        set_adj_matrix()
        
    