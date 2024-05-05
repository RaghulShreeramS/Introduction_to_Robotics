import numpy as np
import heapq

global goal_node # tell no one of my sins

class Node:

    def __init__(self, position, neighbors, cost):
        self.position = position
        self.neighbors = neighbors 
        self.cost = cost
        self.parent = None

    def __repr__(self):
        return "position: {}\nneighbors: {}\ncost: {}".format(self.position, self.neighbors, self.cost)

    def __lt__(self, other):
        global goal_node
        return (self.cost + heuristic(self, goal_node)) < (other.cost + heuristic(other, goal_node))

def get_node_list_grid(positions, interp_len):
    node_list = []
    inf = np.finfo(np.float32).max

    num_points = positions.shape[1]
    # Assumes a square grid
    for i in range(num_points):
        position = positions[:, i]
        cost = inf
        neighbors = []

        # left
        if (i % interp_len) - 1 >= 0:
            neighbors.append((i - 1, 1))  # idx, cost
        # right
        if (i % interp_len) + 1 < interp_len:
            neighbors.append((i + 1, 1))
        # down
        if i - interp_len >= 0:
            neighbors.append((i - interp_len, 1))
        # up
        if i + interp_len < num_points:
            neighbors.append((i + interp_len, 1))
        # up-left diag
        if i + interp_len < num_points and (i % interp_len) - 1 >= 0:
            neighbors.append((i - 1 + interp_len, np.sqrt(2))) 
        # up-right
        if i + interp_len < num_points and (i % interp_len) + 1 < interp_len:
            neighbors.append((i + 1 + interp_len, np.sqrt(2)))
        # down-left
        if i - interp_len >= 0 and (i % interp_len) - 1 >= 0:
            neighbors.append((i - interp_len - 1, np.sqrt(2)))
        # down-right
        if i - interp_len >= 0 and (i % interp_len) + 1 < interp_len:
            neighbors.append((i - interp_len + 1, np.sqrt(2)))

        node_list.append(Node(position, neighbors, cost))
    return node_list

def update_nodes_obstacle(node_list, obstacle_xbounds, obstacle_ybounds, padding):
    obstacle_box_xmin = obstacle_xbounds[0] - padding
    obstacle_box_xmax = obstacle_xbounds[1] + padding
    obstacle_box_ymin = obstacle_ybounds[0] - padding
    obstacle_box_ymax = obstacle_ybounds[1] + padding

    inside_obstacle = lambda x, y: obstacle_box_xmin <= x <= obstacle_box_xmax and \
                                    obstacle_box_ymin <= y <= obstacle_box_ymax

    inf = np.finfo(np.float32).max
    for node in node_list:
        for i in range(len(node.neighbors)):
            neighbor_id, cost = node.neighbors[i]
            position = node_list[neighbor_id].position
            if inside_obstacle(*position):
                node.neighbors[i] = (neighbor_id, inf)  # update cost to inf

def heuristic(node, goal):
    return np.linalg.norm(node.position - goal.position)

def astar(node_list, start, goal):
    global goal_node
    goal_node = goal
    open_set = []
    closed_set = set()

    start.cost = 0
    start.parent = None
    heapq.heappush(open_set, start)

    while open_set:
        current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        closed_set.add(current)

        for neighbor_id, cost in current.neighbors:
            neighbor = node_list[neighbor_id]

            tentative_cost = current.cost + cost
            if tentative_cost < neighbor.cost:
                neighbor.cost = tentative_cost
                neighbor.parent = current
                # neighbor_heuristic = heuristic(neighbor, goal)
                heapq.heappush(open_set, neighbor)
    
    print("path-finding failed")
    return None
