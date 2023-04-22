from graph_tools import *
from BFSearch import BFSearch
from Algos import Algorithms


class Constraints:
    
    def constraints_check1(adj_matr, optimize_flag=False):
        flag = True
        nodes_list = list(range(0, 18))
        # Fully conncted
        if not BFSearch.BFSearch(adj_matr):
            if not optimize_flag:
                generate_graph()
            else:
                return False

        # Maximum 4 diameter 
        for node in nodes_list:  
            dist, prev = g.dijkstra(node)
            if any(4 < val for val in dist.values()):
                if not optimize_flag: 
                    generate_graph()
                else:
                    return False

        # Degree of every node is 3
        for item in adj_matr:
            if item.count(1) < 3:
                if not optimize_flag:
                    generate_graph()
                else:
                    return False

        if flag and not optimize_flag:
            coordinate(adj_matr)
        if flag and optimize_flag:
            return True

    def constraints_check2(adj_matr2):

        nodes_list = list(range(0, 18))
        # Fully conncted
        if not BFSearch.BFSearch(adj_matr2):
            return False

        # Maximum 4 diameter 
        for node in nodes_list:  
            dist, prev = g.dijkstra(node)

            if any(4 < val for val in dist.values()):
                    return False

        # Degree of every node is 3
        for item in adj_matr2:
            if item.count(1) < 3:
                    return False

        return True

    #Generating unique pair of x,y coordinates for each node
    def coordinate(adj_matr):
        list1 = list(range(0, 80))
        master_coord_list = []
        while(len(master_coord_list) < 18):
            coord1 = random.choice(list1)
            coord2 = random.choice(list1)
            x_y = [coord1, coord2]
            if x_y not in master_coord_list:
                master_coord_list.append(x_y)

        print('The coordinates of the nodes are {}'.format(master_coord_list))
        print('\n')

        Total_cost = Algorithms.total_costs(adj_matr, master_coord_list)


    
