import os
import pickle as pkl
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from shapely import geometry
from shapely.prepared import prep
from collections import defaultdict

def get_marker_locations():
    marker_locations = {}
    marker_locations[6]  = [    0, -0.25]
    marker_locations[10] = [  1.0, -0.25]
    marker_locations[7]  = [  2.0, -0.25]
    marker_locations[2]  = [-0.25,     0]
    marker_locations[0]  = [-0.25,   1.0]
    marker_locations[4]  = [-0.25,   2.0]
    marker_locations[3]  = [ 2.25,     0]
    marker_locations[17] = [ 2.25,   1.0]
    marker_locations[1]  = [ 2.25,   2.0]
    marker_locations[14] = [    0,  2.25]
    marker_locations[12] = [  1.0,  2.25]
    marker_locations[13] = [  2.0,  2.25]
    return marker_locations

def get_default_plot():
    marker_locations = get_marker_locations()
    marker_labels = sorted(marker_locations.keys())
    marker_points = np.array([marker_locations[l] for l in marker_labels])

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    # Show markers
    ax.scatter(marker_points[:, 0], marker_points[:, 1], c='b', label='AprilTag points')
    for l in marker_labels:
        ax.text(marker_locations[l][0], marker_locations[l][1] + 0.10, str(l), ha='center')
    return fig, ax

def get_traversed_points(robot_trajectory, grid_positions, robot_length, robot_width, spacing):
    traversed_points_set = defaultdict(set)
    for robot_location in tqdm(robot_trajectory):
        robot_center = robot_location[:2]
        theta = robot_location[2]

        # not rotated
        rect_pts = np.array([[robot_center[0] + robot_length/2, robot_center[1] - robot_width/2], [robot_center[0] + robot_length/2, robot_center[1] + robot_width/2],
                        [robot_center[0] - robot_length/2, robot_center[1] + robot_width/2], [robot_center[0] - robot_length/2, robot_center[1] - robot_width/2]])
        # rotated rectangle
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rect_pts = np.matmul(rot_matrix, (rect_pts - robot_center).T).T + robot_center

        polygon = prep(geometry.Polygon(rect_pts))
        # spacing/2 to check midpoint of cell
        position_pts = [geometry.Point(grid_positions[:, i] + spacing/2) for i in range(grid_positions.shape[1])]
        for i, pt in enumerate(position_pts):
            if polygon.covers(pt):
                traversed_points_set[grid_positions[0, i]].add(grid_positions[1, i])
                # traversed_points_set.append(grid_positions[:, i])
    return traversed_points_set

if __name__ == "__main__":
    robot_width = 0.16 # meters
    robot_length = 0.19 # meters

    # Read in robot trajectory

    rosws = "/root/rb5_ws"
    state_file_path = os.path.join(rosws, 'state_history.pkl')
    # with open(state_file_path, 'rb') as fp:
        # state_history = pkl.load(fp)
    # robot_trajectory = np.array([state[:3] for state in state_history])
    robot_trajectory = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, np.pi/2], [2, 0.2, np.pi/2], [2, 0.2, np.pi], [1, 0.2, np.pi], [0, 0.2, np.pi]]) # test trajectory

    # Create grid points
    bounds = np.array([[0, 2], [0, 2]])
    interp_len = 51
    spacing = (bounds[0, 1] - bounds[0, 0]) / float(interp_len-1)
    # x = np.linspace(bounds[0, 0], bounds[0, 1], interp_len)
    x = np.arange(bounds[0, 0], bounds[0, 1], spacing)
    # y = np.linspace(bounds[1, 0], bounds[1, 1], interp_len)
    y = np.arange(bounds[0, 0], bounds[0, 1], spacing)

    x_vals, y_vals = np.meshgrid(x, y)
    grid_positions = np.vstack([x_vals.ravel(), y_vals.ravel()])

    # Plot coverage
    fig, ax = get_default_plot()
    ax.legend()
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Color grid where robot has traversed 
    points_set = get_traversed_points(robot_trajectory, grid_positions, robot_length, robot_width, spacing)
    traversed_points = []
    for x in points_set.keys():
        for y in points_set[x]:
            traversed_points.append((x,y))
    traversed_points = np.array(traversed_points)

    total_grid_points = grid_positions.shape[1]
    print("Traversed {}/{} points of grid".format(len(traversed_points), total_grid_points))
    ax.set_title("Robot area coverage -- {:.02f}%".format((len(traversed_points)/float(total_grid_points)*100)))

    # Fill in 2x2 grid
    for point in tqdm(traversed_points):
        ax.add_patch(plt.Rectangle(point, spacing, spacing, color='green', fill=True))
    # Add 2x2 grid border
    ax.add_patch(plt.Rectangle([0, 0], 2, 2, color='black', fill=False, lw=2.0))

    # Add safe area info to legend
    legend_patch_fill = mpatches.Patch(color='green', label='Traversed area')
    legend_patch_border = mlines.Line2D([0], [0], color='black', label='Safe area border')
    h, l = ax.get_legend_handles_labels()
    h += [legend_patch_fill, legend_patch_border]
    l += [legend_patch_fill.get_label(), legend_patch_border.get_label()]
    ax.legend(h, l, loc='center left', bbox_to_anchor=(1.0, 0.5))

    # ax.scatter(traversed_points[:, 0], traversed_points[:, 1], c='r', s=10)
    plt.savefig("./coverage_test.png", bbox_inches="tight")
    # plt.show()
