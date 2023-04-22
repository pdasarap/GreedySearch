import numpy as np
from enum import Flag
from msilib.sequence import AdminExecuteSequence
import math
from operator import itemgetter
import sys
import timeit
sys.setrecursionlimit(10000)


#Greedy Local Search Algorithm
def greedy_search(edge_cost_list, opt_cost, adj_matr, master_coord_list):

    #Removing the maximum weighed edge from the graph
    edg_costs_sorted = sorted(edge_cost_list, key = itemgetter(2), reverse=True)
    for ec_idx1 in range(len(edg_costs_sorted)):
        point1 = edg_costs_sorted[ec_idx1][0]
        point2 = edg_costs_sorted[ec_idx1][1]
        node1 = master_coord_list.index(point1)
        node2 = master_coord_list.index(point2)
        adj_matr[node1][node2] = 0
        adj_matr[node2][node1] = 0

        #Checking if the updated graph satisfies all the 3 conditions
        if constraints_check1(adj_matr, optimize_flag=True):
            cost = total_costs(adj_matr, master_coord_list, optimize_flag=True)

            #Checking if newly calculated cost is lesser than the previous one
            if cost < opt_cost:
                opt_cost = cost 
                mat_array = []
                mat_array = np.array(adj_matr)

            #Iterating through all the edges and checking if edges can be removed    
            for ec_idx2 in range(len(edg_costs_sorted)):
                point11 = edg_costs_sorted[ec_idx2][0]
                point12 = edg_costs_sorted[ec_idx2][1]
                node11 = master_coord_list.index(point11)
                node12 = master_coord_list.index(point12)
                adj_matr[node11][node12] = 0
                adj_matr[node12][node11] = 0
                if constraints_check1(adj_matr, optimize_flag=True):
                    cost = total_costs(adj_matr, master_coord_list, optimize_flag=True)
                    if cost < opt_cost:
                        opt_cost = cost
                        mat_array = []
                        mat_array = np.array(adj_matr)               
                    else:
                        pass
                else: # Reverting the changes if the subsequent edge removal does not yeild lowest cost 
                    adj_matr[node11][node12] = 1
                    adj_matr[node12][node11] = 1
        else:
            adj_matr[node1][node2] = 1
            adj_matr[node2][node1] = 1  
        adj_matr[node1][node2] = 1
        adj_matr[node2][node1] = 1

    display_graph(mat_array)
    print('Optimum cost from Greedy Local Search Algorithm {}' .format(opt_cost))


#Original Heuristic Algorithm            
def heuristic_sSearch(edge_cost_list, final_total_cost, adj_matr, cost_matrix, master_coord_list):

    #highest weight in comparisons
    Maxx_wght = float('inf')

    # List showing which nodes are already selected so not to repeat the same node twice
    node1 = [False for node in range(18)]

    res_matr = [[0]*18 for _ in range(18)]

    count = 0

    while(False in node1):
        min_wght = Maxx_wght
        begin = 0
        end = 0
        for i in range(18):
            if node1[i]:
                for j in range(18):
                    # If the analyzed node have a path to the ending node AND its not included in resulting matrix (to avoid cycles)
                    if (not node1[j] and cost_matrix[i][j]>0):  
                        if cost_matrix[i][j] < min_weight:
                            min_wght = cost_matrix[i][j]
                            begin, end = i, j

        node1[end] = True

        res_matr[begin][end] = min_wght

        if min_wght == Maxx_wght:
            res_matr[begin][end] = 0

        count += 1

        # This matrix will have minimum cost path from source to destination
        res_matr[end][begin] = res_matr[begin][end]


    adj_matr2 = [[0]*18 for _ in range(18)]

    for i in range(18):
        for j in range(18):
            if res_matr[i][j]!=0:
                adj_matr2[i][j]=1

    check_flag = True  

    #checking if resulted graph satisfies all the conditions
    while not check_flag == constraints_check2(adj_matr2):
        chk = list(map(sum, adj_matr2))

        for i in range(0,18):
                for j in range(0,18):
                    if adj_matr[i][j] != adj_matr2[i][j] and chk[i]<3:
                        adj_matr2[i][j]=1
                        adj_matr2[j][i]=1
                        chk = list(map(sum, adj_matrix2))


    #Calculating the total cost of the end graph
    cost = total_costs(adj_matr2, master_coord_list, optimize_flag=True)

    mat_array = []
    mat_array = np.array(adj_matrix2)
    display_graph(mat_array)
    print('\n')
    print('Optimum cost from Original Heuristic Algorithm is {}' .format(cost) )


# calculate the totalcosts    
def total_costs(adj_matr, master_coord_list, optimize_flag=False):

    cost_matrx = [[0]*18 for _ in range(18)]

    #Generating the coordinate list for nodes with edges between them
    coord_list = [[x_i,y_i] for x_i, row in enumerate(adj_matr) 
                    for y_i, i in enumerate(row) if i == 1]
    unique_coord_list = [list(i) for i in {*[tuple(sorted(i)) for i in coord_list]}]


    #adding the connected x1,y1 and x2,y2 coordinates to a master list in order to calculate euclidean distance betweem the nodes
    edge_list = []
    for item in unique_coord_list:
        points = []
        for sub in item:
            points.append(master_coord_list[sub])
        edge_list.append(points)

    #Calculating euclidean distance to get the weight of each edge
    edge_cost_list = []
    for item1 in edge_list:
        dist = round(np.linalg.norm(np.array(item1[0]) - 
                                    np.array(item[1])), 3)

        edge_cost_list.append([item1[0], item1[1], dist])

    #Adding the cost of each edge        
    totalcost = 0
    for item in edge_cost_list:
        totalcost += item[-1]
    final_totalcost = round(totalcost, 3)

    #Forming a cost matrix out of adjacency matrix
    k=0
    for i in range(0,18):
        for j in range(0,18):
            if i<j and adj_matr[i][j]==1:
                cost_matrx[i][j] = edge_cost_list[k][2]
                cost_matrx[j][i] = edge_cost_list[k][2]
                k += 1

    if not optimize_flag:  
        t = timeit.timeit(lambda: greedy_Search(edge_cost_list, final_totalcost, adj_matr, master_coord_list), number=1)
        print('Run time of Greedy Local Search Algorithm is {}ms' .format(t))

        t1= timeit.timeit(lambda: heuristic_Search(edge_cost_list, final_totalcost, adj_matr, cost_matrix, master_coord_list), number=1)
        print('Run time of Original Heuristic Algorithm is {}ms' .format(t1))

    if optimize_flag:
        return final_totalcost

