import numpy as np
from scipy.spatial import KDTree
from astar import Node

def generate_rect_points(start, width, height, num_pts):
    startx = start[0]
    starty = start[1]
    endx = startx + width
    endy = starty + height
    num_pts = num_pts

    xspacing = np.linspace(startx, endx, num_pts).reshape(-1, 1)
    yspacing = np.linspace(starty, endy, num_pts).reshape(-1, 1)
    top = np.concatenate([xspacing, endy*np.ones((num_pts, 1))], axis=1)
    bottom = np.concatenate([xspacing, starty*np.ones((num_pts, 1))], axis=1)
    left = np.concatenate([startx*np.ones((num_pts, 1)), yspacing], axis=1)
    right = np.concatenate([endx*np.ones((num_pts, 1)), yspacing], axis=1)

    rect_points = np.concatenate([top, bottom, left, right])
    return rect_points

def fix_input(pos, scaling):
    return ((round(pos[0] * scaling) / scaling), (round(pos[1] * scaling) / scaling))

def get_node_idx_at_pos(node_list, pos, grid_side_len, interp_len):
    scaling = float(interp_len - 1) / (grid_side_len)
    fixed_pos = fix_input(pos, scaling)
    index = int(scaling*fixed_pos[0] + interp_len*scaling*fixed_pos[1])
    return index

def get_node_at_pos(node_list, pos, grid_side_len, interp_len):
    scaling = float(interp_len - 1) / (grid_side_len)
    fixed_pos = fix_input(pos, scaling)
    index = int(scaling*fixed_pos[0] + interp_len*scaling*fixed_pos[1])
    return node_list[index]

def create_node_list_grid(positions, interp_len):

    node_list = []
    inf = np.finfo(np.float32).max

    num_points = positions.shape[1]
    # Assumes square grid
    for i in range(num_points):
        position = positions[:, i]
        cost = inf
        neighbors = []

        # left
        if (i % interp_len) - 1 >= 0:
            neighbors.append((i - 1, 1)) # idx, cost
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

# Padding increases the size of the obstacle bounds on all sides
# padding = 0.1 turns obstacle of size (4, 6), (4, 6) to (3.9, 6.1), (3.9, 6.1)
def update_nodes_obstacle(node_list, obstacle_box_bounds, robot_radius):
    obstacle_box_xmin = obstacle_box_bounds[0][0]# - padding
    obstacle_box_xmax = obstacle_box_bounds[0][1]# + padding
    obstacle_box_ymin = obstacle_box_bounds[1][0]# - padding
    obstacle_box_ymax = obstacle_box_bounds[1][1]# + padding

    inside_obstacle = lambda x, y: obstacle_box_xmin <= x and x <= obstacle_box_xmax and \
                                    obstacle_box_ymin <= y and y <= obstacle_box_ymax

    # interpolate box outline for robot radius detection
    box_side_x = obstacle_box_bounds[0][1] - obstacle_box_bounds[0][0]
    box_side_y = obstacle_box_bounds[1][1] - obstacle_box_bounds[1][0]
    obstacle_box_pts = generate_rect_points((obstacle_box_bounds[0][0], obstacle_box_bounds[1][0]), box_side_x, box_side_y, num_pts=5)

    obstacle_tree = KDTree(obstacle_box_pts)
    safe_robot_radius = np.sqrt(2) * robot_radius # Since we have diagonal motions, expand radius of robot to account for this

    inf = np.finfo(np.float32).max
    for node in node_list:
        for i in range(len(node.neighbors)):
            neighbor_id, _ = node.neighbors[i]
            position = node_list[neighbor_id].position

            if inside_obstacle(*position):
                node.neighbors[i] = (neighbor_id, inf) # update cost to inf
            else: 
                # check distance from obstacle greater than robot radius
                dist, _ = obstacle_tree.query(position, k=1)
                if dist <= safe_robot_radius:
                    node.neighbors[i] = (neighbor_id, inf) # update cost to inf

def get_grid_data(x_bounds, y_bounds, obstacle_bounds, interp_len, robot_radius):
    # interp_len = 101
    x = np.linspace(x_bounds[0], x_bounds[1], interp_len)
    y = np.linspace(y_bounds[0], y_bounds[1], interp_len)

    x_vals, y_vals = np.meshgrid(x, y)
    positions = np.vstack([x_vals.ravel(), y_vals.ravel()])

    node_list = create_node_list_grid(positions, interp_len)

    # Add obstacle to graph with x and y bounds
    update_nodes_obstacle(node_list, obstacle_bounds, robot_radius)

    return node_list
