import numpy as np
import heapq
import matplotlib.pyplot as plt
from shapely.geometry import LineString


global goal_node

class Node:
    position: np.ndarray
    neighbors: list  # list of [vertex_id, cost]
    cost: float
    parent: 'Node' = None

    def __init__(self, position, neighbors, cost):
        self.position = position
        self.neighbors = neighbors 
        self.cost = cost

    def __repr__(self) -> str:
        return f"position: {self.position}\nneighbors: {self.neighbors}\ncost: {self.cost}"

    def __lt__(self, other):
        global goal_node
        return (self.cost + heuristic(self, goal_node)) < (other.cost + heuristic(other, goal_node))


def fix_input(pos, scaling):
    return ((round(pos[0] * scaling) / scaling), (round(pos[1] * scaling) / scaling))

def get_node_at_pos(node_list, pos, grid_side_len, interp_len):
    scaling = float(interp_len - 1) / (grid_side_len)
    fixed_pos = fix_input(pos, scaling)
    index = int(scaling*fixed_pos[0] + interp_len*scaling*fixed_pos[1])
    return node_list[index]

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
    inf = np.finfo(np.float32).max

    for node in node_list:
        node.cost = inf


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

            # if neighbor in closed_set:
            #     continue

            tentative_cost = current.cost + cost

            if tentative_cost < neighbor.cost:     #tentative_cost + heuristic(neighbor, goal) < neighbor.cost + heuristic(neighbor, goal): 
                neighbor.cost = tentative_cost
                neighbor.parent = current
                # neighbor_heuristic = heuristic(neighbor, goal)
                heapq.heappush(open_set, neighbor)


def visualize_map_with_obstacles(node_list, obstacle_xbounds, obstacle_ybounds, path=None):
    x_vals = [node.position[0] for node in node_list]
    y_vals = [node.position[1] for node in node_list]

    fig, ax = plt.subplots()
    ax.scatter(x_vals, y_vals, marker='.', color='black', alpha=0.0)

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6)

    # Draw obstacle
    obstacle_x = [obstacle_xbounds[0], obstacle_xbounds[1], obstacle_xbounds[1], obstacle_xbounds[0], obstacle_xbounds[0]]
    obstacle_y = [obstacle_ybounds[0], obstacle_ybounds[0], obstacle_ybounds[1], obstacle_ybounds[1], obstacle_ybounds[0]]
    ax.fill(obstacle_x, obstacle_y, color='red', alpha=0.8, label='Obstacle')

    if path is not None and len(path) > 0:  # Check if path is not empty    #if path:
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]
        ax.plot(path_x, path_y, marker='o', linestyle='-', color='blue', label='Path')

    ax.legend()
    plt.show()





def simplify_path(path, tolerance):


    # Reduce the number of point in the optimal path --------------------------------------------------------------------
    points = np.array([(point[0], point[1]) for point in path])
    line = LineString(points)
    simplified_line = line.simplify(tolerance, preserve_topology=False)
    simplified_path = np.array(simplified_line.coords)



    return simplified_path


#-----------------------------------------------------------------------HW5-----------------------------------------------------------------------------------------------------------

def generate_sweeping_waypoints(cell_origin, num_rows, num_cols, up_steps, right_left_steps):
    waypoints = []

    for row in range(num_rows):
        if row % 2 == 0:
            # Move from left to right
            for col in range(num_cols):
                x = cell_origin[0] + col * right_left_steps   # Horizontal Motion
                x = round(x, 2)
                y = cell_origin[1] + row * up_steps           # Vertical Motion
                y = round(y, 2)
                waypoints.append((x, y))
        else:
            # Move from right to left
            for col in range(num_cols - 1, -1, -1):
                x = cell_origin[0] + col * right_left_steps   # Horizontal Motion
                x = round(x, 2)
                y = cell_origin[1] + row * up_steps           # Vertical Motion
                y = round(y, 2)
                waypoints.append((x, y))

    return waypoints

def calculate_orientation(current_point, next_point):
    # Calculate orientation (theta) based on the difference in coordinates
    delta_x = next_point[0] - current_point[0]
    delta_y = next_point[1] - current_point[1]
    orientation = np.arctan2(delta_y, delta_x)
    
    return orientation

def get_waypoints_with_orientation(waypoints):

    waypoints_with_orientation = []
    # waypoints_with_orientation.append([waypoints[0][0], waypoints[0][1], 0.0])

    for i in range(len(waypoints) - 1):
        current_point = waypoints[i]
        next_point = waypoints[i + 1]
        
        # Calculate orientation for the current pair
        orientation = calculate_orientation(current_point, next_point)
        
        # Append [x, y, theta] to the new list
        waypoints_with_orientation.append([current_point[0], current_point[1], orientation])

    # Add the last waypoint with orientation
    waypoints_with_orientation.append([waypoints[-1][0], waypoints[-1][1], 0.0])  # Assuming 0.0 as default orientation

    return waypoints_with_orientation


def get_robot_waypoints(waypoints_with_orientation):
    path_robot = []
    for i in range(len(waypoints_with_orientation)-1):
        # print(i)
        path_robot.append([waypoints_with_orientation[i][0], waypoints_with_orientation[i][1], waypoints_with_orientation[i][2]])

        if(abs(waypoints_with_orientation[i][2]-waypoints_with_orientation[i+1][2]) > 0.1):
            path_robot.append([waypoints_with_orientation[i+1][0], waypoints_with_orientation[i+1][1], waypoints_with_orientation[i][2]])

    n = len(waypoints_with_orientation)-1
    path_robot.append([waypoints_with_orientation[n][0], waypoints_with_orientation[n][1], waypoints_with_orientation[n][2]])

    return path_robot


def obstacle_between_points(box_vertices, point1, point2):
    # Get the bounding box of the given vertices
    min_x = min(v[0] for v in box_vertices)
    max_x = max(v[0] for v in box_vertices)
    min_y = min(v[1] for v in box_vertices)
    max_y = max(v[1] for v in box_vertices)

    # Check if the bounding box is between the two points
    return (
        min_x >= min(point1[0], point2[0]) and
        max_x <= max(point1[0], point2[0]) and
        min_y >= min(point1[1], point2[1]) and
        max_y <= max(point1[1], point2[1])
    )

def is_point_inside_box(point, box_vertices):
    x, y = point
    
    # Extracting the vertices of the box
    x_vertices, y_vertices = zip(*box_vertices)
    
    # Checking if the point is within the bounding box
    if min(x_vertices) <= x <= max(x_vertices) and min(y_vertices) <= y <= max(y_vertices):
        return True
    else:
        return False


def get_waypoints_with_obstacle_avoidance(waypoints, obstacles_coordinates, node_list, grid_side_len):
    waypoints_with_avoidance = []
    waypoints_with_avoidance.append(waypoints[0])
    i=0
    while(i<(len(waypoints) - 1)):
        current_point = waypoints[i]
        next_point = waypoints[i + 1]
        loop_start = i+1

        # Check for obstacles between current and next point
        is_obstacle_between_points = obstacle_between_points(obstacles_coordinates,current_point,next_point)
        point_insisde_obstacle = is_point_inside_box(next_point,obstacles_coordinates)

        if is_obstacle_between_points:
            #call astar function to get the shortest path with obstacle and add the 
            i+=1

            start_node = get_node_at_pos(node_list, current_point, grid_side_len, interp_len)
            goal_node =  get_node_at_pos(node_list, next_point, grid_side_len, interp_len)
            shortest_path_obstacle = astar(node_list, start_node, goal_node)

            simplify_shortest_path = simplify_path(shortest_path_obstacle,tolerance = 0.00002)
            simplify_shortest_path = simplify_shortest_path[1:-1]

            if simplify_shortest_path != None:

                simplify_shortest_path = [tuple(arr) for arr in simplify_shortest_path]
                for arr in simplify_shortest_path:
                    temp = tuple(arr)
                    waypoints_with_avoidance.append(tuple(temp))



                
        elif point_insisde_obstacle:
            print("before", i)
            i+=1
            for j in range(loop_start,len(waypoints)):
                next_point = waypoints[j+1]
                
                if(is_point_inside_box(next_point,obstacles_coordinates)):
                    i+=1
                    print(current_point, next_point)
                    continue
                else:
                    break
            
            print("after", i)
            print(current_point, next_point)

            start_node = get_node_at_pos(node_list, current_point, grid_side_len, interp_len)
            goal_node =  get_node_at_pos(node_list, next_point, grid_side_len, interp_len)

            # print("\n\nStart and Goal node:" ,start_node)
            # print(goal_node)
            # print("\n\n")
            shortest_path_obstacle = astar(node_list, start_node, goal_node)
            
            simplify_shortest_path = simplify_path(shortest_path_obstacle,tolerance = 0.00002)
            simplify_shortest_path = simplify_shortest_path[1:-1]

            print("Here",simplify_shortest_path)

            if simplify_shortest_path.any() != None:
                
                simplify_shortest_path = [tuple(arr) for arr in simplify_shortest_path]
                for arr in simplify_shortest_path:
                    temp = tuple(arr)
                    waypoints_with_avoidance.append(tuple(temp))

            # print("Here" , simplify_shortest_path)
            if simplify_shortest_path is None:
                print("Start pos:", current_point)
                print("Goal pos:", next_point)
                print(i)


        else:
            # If no obstacle-free path found, use the original next point
            i+=1
            waypoints_with_avoidance.append(next_point)

    # Add the last waypoint
    waypoints_with_avoidance.append(waypoints[-1])

    return waypoints_with_avoidance




#----------------------------------------------------------------------------Main----------------------------------------------------------------------------------------------

interp_len = 101
x = np.linspace(0, 10, interp_len)
y = np.linspace(0, 10, interp_len)

x_vals, y_vals = np.meshgrid(x, y)
positions = np.vstack([x_vals.ravel(), y_vals.ravel()])

node_list = get_node_list_grid(positions, interp_len)

obstacle_xbounds = (2, 3)
obstacle_ybounds = (2, 3)
# Add obstacle to graph with x and y bounds
update_nodes_obstacle(node_list, obstacle_xbounds, obstacle_ybounds, padding=0.17677)


grid_side_len = 10
cell_origin = (0, 0)
grid_size = 6
up_steps = 0.2
right_left_steps = 0.5
num_rows = int((grid_size/up_steps))
num_cols = 9


v1 = (obstacle_xbounds[0],obstacle_ybounds[0])
v2 = (obstacle_xbounds[0],obstacle_ybounds[1])
v3 = (obstacle_xbounds[1],obstacle_ybounds[1])
v4 = (obstacle_xbounds[1],obstacle_ybounds[0])

obstacles_coordinates = [v1,v2,v3,v4] 

waypoints = generate_sweeping_waypoints(cell_origin, num_rows, num_cols, up_steps, right_left_steps)

waypoints_with_obstacle_avoidance = get_waypoints_with_obstacle_avoidance(waypoints,obstacles_coordinates, node_list,grid_side_len)

print(waypoints_with_obstacle_avoidance)




visualize_map_with_obstacles(node_list, obstacle_xbounds, obstacle_ybounds, waypoints_with_obstacle_avoidance)
