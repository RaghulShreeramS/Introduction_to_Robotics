import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import LineString

from astar import astar, Node
from grid_data import get_grid_data
from grid_data import get_node_at_pos as get_node_at_pos_grid
from grid_data import get_node_idx_at_pos as get_node_idx_at_pos_grid
from voronoi_data import interpolate_obstacle_points, get_voronoi_graph, create_node_list_voronoi
from voronoi_data import get_node_at_pos as get_node_at_pos_voronoi

def get_marker_locations():
    marker_locations = {}
    marker_locations[1] = [0.6,   3]
    marker_locations[2] = [  0, 0.6]
    marker_locations[3] = [  3, 2.4]
    marker_locations[4] = [2.4,   3]
    marker_locations[5] = [  3, 0.6]
    marker_locations[6] = [2.4,   0]
    marker_locations[7] = [0.6,   0]
    marker_locations[8] = [  0, 2.4]
    return marker_locations

def get_default_plot(obstacle_box_bounds=None, start=None, goal=None):
    marker_locations = get_marker_locations()
    marker_labels = sorted(marker_locations.keys())
    marker_points = np.array([marker_locations[l] for l in marker_labels])

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    # Show markers
    ax.scatter(marker_points[:, 0], marker_points[:, 1], c='b', label='AprilTag points')
    for l in marker_labels:
        # ax.text(marker_points[l-1, 0], marker_points[l-1, 1] + 0.18, str(l), ha='center')
        ax.text(marker_points[l-1, 0], marker_points[l-1, 1] + 0.10, str(l), ha='center')

    # Show obstacle
    if obstacle_box_bounds is not None:
        x_vals, y_vals =np.meshgrid(obstacle_box_bounds[0], obstacle_box_bounds[1])
        obstacle_box_corners = np.vstack([x_vals.ravel(), y_vals.ravel()]).T
        
        obstacle_box_xmin = obstacle_box_bounds[0][0]
        obstacle_box_xmax = obstacle_box_bounds[0][1]
        obstacle_box_ymin = obstacle_box_bounds[1][0]
        obstacle_box_ymax = obstacle_box_bounds[1][1]
        width = obstacle_box_xmax-obstacle_box_xmin
        height = obstacle_box_ymax-obstacle_box_ymin
        obstacle_xcenter = obstacle_box_xmin + (width / 2)
        obstacle_ycenter = obstacle_box_ymin + (height / 2)

        ax.add_patch(Rectangle((obstacle_box_xmin, obstacle_box_ymin), width, height, color='grey'))
        ax.text(obstacle_xcenter, obstacle_ycenter, "Obstacle", ha='center', va='center')
        ax.scatter(obstacle_box_corners[:, 0], obstacle_box_corners[:, 1], c='r')

    if start is not None:
        ax.scatter(start[0], start[1], c='c')
        # ax.text(start[0], start[1] - 0.4, "start")
        ax.text(start[0], start[1] - 0.2, "start")
    if goal is not None:
        ax.scatter(goal[0], goal[1], c='m')
        # ax.text(goal[0], goal[1] + 0.25, "goal")
        ax.text(goal[0], goal[1] + 0.15, "goal")

    return fig, ax

def simplify_path(path, tolerance):


    # Reduce the number of point in the optimal path --------------------------------------------------------------------
    points = np.array([(point[0], point[1]) for point in path])
    line = LineString(points)
    simplified_line = line.simplify(tolerance, preserve_topology=False)
    simplified_path = np.array(simplified_line.coords)


    # Get the optimal path with orientation after reduction-----------------------------------------------------------------------------
    path_with_orientation = []

    current_node1 = Node(np.array([path[0][0], path[0][1]]), [], 0)
    path_with_orientation.append([current_node1.position[0], current_node1.position[1], 0])

    for i in range(len(simplified_path) - 1):

        current_node = Node(np.array([simplified_path[i][0], simplified_path[i][1]]), [], 0)
        next_node = Node(np.array([simplified_path[i + 1][0], simplified_path[i + 1][1]]), [], 0)
        
        direction_vector = next_node.position - current_node.position
        orientation = np.arctan2(direction_vector[1], direction_vector[0])
        
        if not np.isclose(orientation, path_with_orientation[-1][2]):
            path_with_orientation.append([next_node.position[0], next_node.position[1], orientation])
        else:
            path_with_orientation[-1] = [next_node.position[0], next_node.position[1], orientation]
    path_with_orientation = np.array(path_with_orientation)


    #Convert the reduced waypoints to robot_waypoints---------------------------------------------------------------------
    path_robot = []
    for i in range(len(path_with_orientation)-1):
        # print(i)
        path_robot.append([path_with_orientation[i][0], path_with_orientation[i][1], path_with_orientation[i][2]])

        if(abs(path_with_orientation[i][2]-path_with_orientation[i+1][2]) > 0.1):
            path_robot.append([path_with_orientation[i][0], path_with_orientation[i][1], path_with_orientation[i+1][2]])

    n = len(path_with_orientation)-1
    path_robot.append([path_with_orientation[n][0], path_with_orientation[n][1], path_with_orientation[n][2]])

    return path_with_orientation, np.array(path_robot)



if __name__ == '__main__':
    x_bounds = [0, 3]
    y_bounds = [0, 3]
    # [xmin, xmax], [ymin, ymax] bounds
    center = 1.5
    width, length = 0.39, 0.26 # in meters
    xstart = center - (width/2)
    xend = center + (width/2)
    ystart = center - (length/2)
    yend = center + (length/2)
    obstacle_box_bounds = np.array([[xstart, xend], [ystart, yend]])
    robot_radius = 0.18

    start_pos = [2.4, 0.6]
    goal_pos = [0.6, 2.4]

    # Grid data
    interp_len = 101
    node_list = get_grid_data(x_bounds, y_bounds, obstacle_box_bounds, interp_len, robot_radius)

    grid_side_len = x_bounds[1] - x_bounds[0]
    start_node = get_node_at_pos_grid(node_list, start_pos, grid_side_len, interp_len)
    goal_node = get_node_at_pos_grid(node_list, goal_pos, grid_side_len, interp_len)
    
    # Get path with A*
    path = astar(node_list, start_node, goal_node)
    path = np.array(path)
    simplified_path, robot_waypoints = simplify_path(path, tolerance = 0.075)  # Adjust the tolerance based on your requirements
    # np.save("./shortest_path.npy", simplified_path)
    np.save("./shortest_robot_waypoints.npy", robot_waypoints)

    # print("grid")
    # print("Simplified path: \n{}".format(simplified_path))
    # print()
    # print("robot_waypoints: \n{}".format(robot_waypoints))

    # # Plot path
    # print("Number of path points: {}".format(path.shape[0]))
    # fig_g, ax_g = get_default_plot(obstacle_box_bounds, start_node.position, goal_node.position)
    # ax_g.plot(path[:, 0], path[:, 1], label='path')
    # ax_g.legend()
    # ax_g.set_title("Shortest path")

    # # Plot path
    # print("Number of path points: {}".format(simplified_path.shape[0]))
    # fig_g2, ax_g2 = get_default_plot(obstacle_box_bounds, start_node.position, goal_node.position)
    # ax_g2.plot(simplified_path[:, 0], simplified_path[:, 1], label='path')
    # ax_g2.legend()
    # ax_g2.set_title("Simplified shortest path")


    # Voronoi data
    all_obstacle_points = interpolate_obstacle_points(x_bounds, y_bounds, obstacle_box_bounds, num_pts=5)
    vertices, edges = get_voronoi_graph(all_obstacle_points, obstacle_box_bounds, start_pos, goal_pos, k=10, robot_radius=robot_radius)
    node_list =  create_node_list_voronoi(vertices, edges)

    start_node = get_node_at_pos_voronoi(node_list, start_pos)
    goal_node = get_node_at_pos_voronoi(node_list, goal_pos)

    # # Plot voronoi graph
    # fig_vg, ax_vg = get_default_plot(start=start_pos, goal=goal_pos)
    # ax_vg.set_title("Voronoi graph")
    # for i in range(len(edges)):
    #     edge_vertices = vertices[edges[i]]
    #     ax_vg.plot(edge_vertices[:, 0], edge_vertices[:, 1])
    # ax_vg.legend()
    
    # Get path with A*
    path = astar(node_list, start_node, goal_node)
    path = np.array(path)
    simplified_path, robot_waypoints = simplify_path(path, tolerance = 0.06)  # Adjust the tolerance based on your requirements
    # np.save("./safest_path.npy", simplified_path)
    np.save("./safest_robot_waypoints.npy", robot_waypoints)

    # print()
    # print("Voronoi")
    # print("Simplified path: \n{}".format(simplified_path))
    # print()
    # print("robot_waypoints: \n{}".format(robot_waypoints))

    # # Plot path
    # print("Number of path points: {}".format(path.shape[0]))
    # fig_v, ax_v = get_default_plot(obstacle_box_bounds, start_node.position, goal_node.position)
    # ax_v.plot(path[:, 0], path[:, 1], label='path')
    # ax_v.legend()
    # ax_v.set_title("Safest path")

    # # Plot path
    # print("Number of path points: {}".format(simplified_path.shape[0]))
    # fig_v2, ax_v2 = get_default_plot(obstacle_box_bounds, start_node.position, goal_node.position)
    # ax_v2.plot(simplified_path[:, 0], simplified_path[:, 1], label='path')
    # ax_v2.legend()
    # ax_v2.set_title("Simplified safest path")


    # # Grid repr plot
    # fig, ax = get_default_plot(obstacle_box_bounds, start_pos, goal_pos)
    # interp_len = 101
    # x = np.linspace(x_bounds[0], x_bounds[1], interp_len)
    # y = np.linspace(y_bounds[0], y_bounds[1], interp_len)

    # x_vals, y_vals = np.meshgrid(x, y)
    # positions = np.vstack([x_vals.ravel(), y_vals.ravel()]).T
    # ax.set_title("Grid Representation")
    # ax.scatter(positions[:, 0], positions[:, 1], c='g', s=0.5, label="Grid points", zorder=0)
    # ax.legend()


    # # Plot trajectories
    # safe_trajectory = np.load("h4_safest_path_trajectory.npy")
    # fig_1, ax_1 = get_default_plot(obstacle_box_bounds, start_pos, goal_pos)
    # ax_1.plot(safe_trajectory[:, 0], safe_trajectory[:, 1], label="Trajectory")
    # ax_1.legend()
    # ax_1.set_title("Safest path actual trajectory")

    # shortest_trajectory = np.load("h4_shortest_path_trajectory.npy")
    # fig_2, ax_2 = get_default_plot(obstacle_box_bounds, start_pos, goal_pos)
    # ax_2.plot(shortest_trajectory[:, 0], shortest_trajectory[:, 1], label="Trajectory")
    # ax_2.legend()
    # ax_2.set_title("Shortest path actual trajectory")

    # plt.show()

