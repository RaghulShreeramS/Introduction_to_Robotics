#!/usr/bin/env python
"""
Copyright 2023, UC San Diego, Contextual Robotics Institute

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
# rosrun camera_calibration cameracalibrator.py --size 8x6 --square 0.023 image:=/camera_0 camera:=/camera_0 --no-service-check
import rospy
from april_detection.msg import AprilTagDetectionArray
from geometry_msgs.msg import TransformStamped, Transform, Pose
import numpy as np
import tf 
import tf2_ros
import tf_conversions
from filterpy.kalman import KalmanFilter
import time
import math
from copy import deepcopy
import pickle as pkl
from threading import Lock

TIMESTEP = 0.05
# TIMESTEP = 0.1


""" Things to be done:

* Verify Kalman filter (state vec & covariance matrix update)
    * Fix covariance not decreasing!!!

* Compute Q matrix (Raghul)

* Make robot's dead reckoning good enough to travel 1m x 1m square 

"""

class StateVector:
    # TODO: update robot_pos, covariance when predict/update step is used in motor control
    #       Need to setup node communication first
    def __init__(self):
        self.robot_pos = np.zeros(3)
        self.markers = {}
        self.covariance = np.diag([0.0, 0.0, 0.0])
        self.covariance_history = [self.covariance]
        self.lock = Lock()
    
    # Ignore lock param for pickle
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add baz back since it doesn't exist in the pickle
        self.lock = Lock()
    
    def __len__(self):
        return len(self.robot_pos) + 3*len(self.markers)

    def get_vector(self):
        state = np.zeros(len(self))
        state[0:3] = self.robot_pos

        sorted_keys = sorted(self.markers.keys())
        for i, key in enumerate(sorted_keys):
            idx = len(self.robot_pos) + 3*i
            state[idx:idx+3] = self.markers[key]
        return state
    
    def update_state(self, new_state):
        self.robot_pos = new_state[:3]

        sorted_keys = sorted(self.markers.keys())
        for i, key in enumerate(sorted_keys):
            idx = len(self.robot_pos) + 3*i
            self.markers[key] = new_state[idx:idx+3]

    def add_firsttime_detection(self, tag_id, marker_location):
        self.lock.acquire()
        self.markers[tag_id] = marker_location
        sorted_tag_idx = 3*(sorted(self.markers.keys()).index(tag_id) + 1) # +1 bc of robot covariances
        # Expand covariance matrix
        orig_size = self.covariance.shape[0]
        new_covariance = np.insert(self.covariance, [sorted_tag_idx], np.zeros((3, orig_size)), axis=0)
        new_covariance = np.insert(new_covariance, [sorted_tag_idx], np.zeros((orig_size + 3, 3)), axis=1)
        # new_covariance = np.vstack([new_covariance, np.zeros((orig_size, 3))]) # add 3 rows
        # new_covariance = np.hstack([new_covariance, np.zeros((orig_size + 3, 3))]) # add 3 columns
        init_marker_covariance = np.array([5 * 0.5, 5 * 0.5, 4 * 25*np.pi/180]) # TODO: Randomly chose to initialize by doing scalar * Q values
        new_covariance[sorted_tag_idx + 0, sorted_tag_idx + 0] = init_marker_covariance[0]
        new_covariance[sorted_tag_idx + 1, sorted_tag_idx + 1] = init_marker_covariance[1]
        new_covariance[sorted_tag_idx + 2, sorted_tag_idx + 2] = init_marker_covariance[2]

        # insert new covariances into diagonal
        # new_diagonal = np.concatenate([original_covariances[:3*sorted_tag_idx + 3], init_marker_covariance, original_covariances[3*sorted_tag_idx + 3:]])
        # self.covariance = np.diag(new_diagonal)
        # print("original_covariances:", self.covariance)
        # print("new cov:", init_marker_covariance)
        # print("debug new covariance:", new_covariance)
        self.covariance = new_covariance
        self.covariance_history.append(self.covariance)
        self.lock.release()


class LocalizationNode:
    def __init__(self, state):
        # type: (LocalizationNode, StateVector) -> None
        self.sub_april = rospy.Subscriber("/apriltag_detection_array", AprilTagDetectionArray, self.april_detection_callback, queue_size=1)
        self.once = True
        self.state = state
        # self.control_values = control_values # u_t
        self.most_recent_detections = {} # z_t
        self.detections_lock = Lock()
    
    # Ignore lock param for pickle
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["detections_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add baz back since it doesn't exist in the pickle
        self.detections_lock = Lock()

    # For later analysis
    def save_covariances(self):
        with open('./covariances.pkl', 'wb') as f:
            pkl.dump(self.state.covariance_history, f)

    def april_detection_callback(self, april_detections):
        cur_robot_pose = self.state.robot_pos
        num_detections = len(april_detections.detections)
        if num_detections == 0:
            self.most_recent_detections = {}
            print("*** NO WAYPOINTS DETECTED ***")
            return
        # print(april_detections)

        self.detections_lock.acquire()
        detected_locations = {}
        for detection in april_detections.detections:
            # Add new detection
            if detection.id not in self.state.markers.keys():
                marker_location = getNewMarker(detection.pose, cur_robot_pose)
                self.state.add_firsttime_detection(detection.id, marker_location)
                print("NEW MARKER WORLD LOCATION", marker_location)
            
            # Get relative distance to robot
            euler = tf_conversions.transformations.euler_from_quaternion([detection.pose.orientation.w, detection.pose.orientation.x, detection.pose.orientation.y, detection.pose.orientation.z], axes='sxyz') 
            tag_angle_robot = euler[1]

            location = np.array([detection.pose.position.z, -detection.pose.position.x, tag_angle_robot]) # convert camera [z, x] to robot [x, y]
            detected_locations[detection.id] = location
            # print("Rel location to robot", location)

        print('detected_locations:', detected_locations)
        self.most_recent_detections = detected_locations
        self.detections_lock.release()

    def updateState(self, motion_ctrl):
        self.detections_lock.acquire()
        recent_detections = deepcopy(self.most_recent_detections)
        self.detections_lock.release()
        self.state.lock.acquire()
        prev_state = deepcopy(self.state)
        self.state.lock.release()
        print("updateState detections")
        print(recent_detections)
        print("prev_state")
        print(prev_state.robot_pos)
        print(prev_state.markers)
        _ , _, _, _, _, new_state, new_covariance = runKalmanFilter(prev_state, motion_ctrl, recent_detections)

        new_state = clip_rotations(new_state)
        new_covariance = clip_rotations_cov(new_covariance)
        prev_state.update_state(new_state)
        prev_state.covariance = new_covariance
        prev_state.covariance_history.append(new_covariance)
        print("updated state + detections:")
        print(prev_state.robot_pos)
        print(prev_state.markers)
        print("kalman updated covariance:")
        print(np.diagonal(new_covariance))
        self.state.lock.acquire()
        for tag_id in self.state.markers.keys():
            if tag_id not in prev_state.markers.keys():
                prev_state.add_firsttime_detection(tag_id, self.state.markers[tag_id])
        self.state.lock.release()
        self.state = prev_state
        self.most_recent_detections = {}
        # print("new state:", self.state.get_vector())
        return self.state

# Limit rotations to being in-between [-pi, pi]
# Warning: has assumption as to where rotation indices are
def clip_rotations(state_vector):
    clipped_state_vector = np.copy(state_vector)
    if len(state_vector) == 0:
        return state_vector
    for i in range(2, len(state_vector), 3):
        clipped_state_vector[i] = (clipped_state_vector[i] + np.pi) % (2 * np.pi) - np.pi
    return clipped_state_vector


def clip_rotations_cov(covariance):
    clipped_cov = np.copy(covariance)
    diagonal = np.copy(np.diagonal(clipped_cov))
    for i in range(2, len(diagonal), 3):
        diagonal[i] = (diagonal[i] + np.pi) % (2 * np.pi) - np.pi
    np.fill_diagonal(clipped_cov, diagonal)
    return clipped_cov

# prev_state: StateVector
# prev_motion_ctrl = np.array[vx, vy, omega]
# detections = {tag0: np.array[x, y, theta], tag2: np.array[x, y, theta], etc.}
# Assumes any new detected waypoints are already in prev_state
# TODO: Don't perform update step if no detections
def runKalmanFilter(prev_state, prev_motion_ctrl, detections):
    # print("prev_state", prev_state.get_vector())
    # print("prev_motion_ctrl", prev_motion_ctrl)
    # print("detections", detections)
    sorted_keys = sorted(detections.keys())
    feat_vec = np.array([detections[tag_id] for tag_id in sorted_keys]).flatten()
    # if len(feat_vec) == 0:
    #     feat_vec = np.array(prev_state.get_vector())

    F, G, H, Q, R = getKalmanMatrices(prev_state, detections)

    # print(np.matmul(H, prev_state.get_vector()))
    # print(feat_vec)

    if len(feat_vec) > 0:
        print("s")
        print(prev_state.get_vector())
        print("Hs")
        print(np.matmul(H, prev_state.get_vector()))
        filter = KalmanFilter(dim_x=len(prev_state), dim_z=len(feat_vec), dim_u=len(prev_motion_ctrl))
    else:
        filter = KalmanFilter(dim_x=len(prev_state), dim_z=1, dim_u=len(prev_motion_ctrl))
    filter.x = prev_state.get_vector()
    filter.P = prev_state.covariance
    filter.F = F
    filter.B = G
    filter.H = H # = None if no feat_vec
    filter.Q = Q
    filter.R = R # = None if no feat_vec

    filter.predict(u=prev_motion_ctrl)
    print("predict step state")
    print(filter.x)
    print("predict step covariance")
    print(np.diagonal(filter.P))
    # print(filter.P)
    if len(feat_vec) != 0:
        filter.update(feat_vec)
    
    return F, G, H, Q, R, filter.x, filter.P

def getKalmanMatrices(prev_state, detections):
    F = np.identity(len(prev_state))
    G = np.identity(3) * TIMESTEP
    G = np.vstack([G, np.zeros((3*len(prev_state.markers), 3))])

    # Create H
    if len(detections) == 0:
        # H = np.ones((1, len(prev_state)))
        # H = np.identity(len(prev_state))
        H = None
    else:
        H = np.zeros((3*len(detections), len(prev_state)))
        sorted_keys = sorted(detections.keys())
        # print('H')
        # print(H)
        # print('state_vec')
        # print(prev_state.get_vector())
        # print('detections')
        # print(detections) # TODO renable for debugging race condition
        for idx, tag_id in enumerate(sorted_keys):

            tag_vector = np.zeros(len(prev_state) - 3)
            tag_vector_x, tag_vector_y, tag_vector_theta = np.copy(tag_vector), np.copy(tag_vector), np.copy(tag_vector)
            tag_vector_x[3*idx] = 1
            tag_vector_y[3*idx + 1] = 1
            tag_vector_theta[3*idx + 2] = 1

            H[3*idx]     = np.concatenate([np.array([-1, 0, 0]), tag_vector_x])
            H[3*idx + 1] = np.concatenate([[0, -1, 0], tag_vector_y])
            H[3*idx + 2] = np.concatenate([[0, 0, -1], tag_vector_theta])

        # Transform from world to robot frame
        theta = prev_state.robot_pos[2]
        # technically should be -theta, but I switched the signs on np.sin instead (np.cos not affected)
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]) # TODO: check negative is in correct spot
        transform = np.kron(np.identity(len(detections), dtype=np.int32), rotation_matrix)
        H = np.matmul(transform, H)

    # Create Q
    robot_mean_err = TIMESTEP*(np.array([0.15, 0.05, 30*np.pi/180]))
    tag_world_mean_err = np.array([0.2, 0.15, 20*np.pi/180])
    Q_values = np.concatenate([robot_mean_err, np.tile(tag_world_mean_err, len(prev_state.markers))])**2
    Q = 0.1*np.diag(Q_values)
    # Q = np.diag(np.zeros(len(Q_values))) # testing no movement

    # Create R
    tag_detect_mean_err = np.array([0.07, 0.07, 0.174532925]) # 10 deg error (10np.pi*180)
    R_values = tag_detect_mean_err**2
    if len(detections) > 0:
        R = np.diag(np.tile(R_values, len(detections)))
    else:
        # R = np.array([[1]])
        # R = np.identity(len(prev_state))*0.05
        R = None
    return F, G, H, Q, R


def getNewMarker(tag_pose, robot_pos):
    # type: (Pose, np.ndarray) -> np.ndarray

    tag_pos_robot = np.array([tag_pose.position.z, -tag_pose.position.x])

    theta = robot_pos[2]
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    translation = np.array([robot_pos[0], robot_pos[1]])

    tag_pos_world = np.matmul(rotation_matrix, tag_pos_robot) + translation

    euler = tf_conversions.transformations.euler_from_quaternion([tag_pose.orientation.w, tag_pose.orientation.x, tag_pose.orientation.y, tag_pose.orientation.z], axes='sxyz') 
    tag_angle_robot = euler[1]
    # tag to world angle = world to robot angle + robot to tag angle
    tag_angle_world = robot_pos[2] + tag_angle_robot

    return np.array([tag_pos_world[0], tag_pos_world[1], tag_angle_world])


if __name__ == "__main__":
    rospy.init_node("localization")

    prev_state = StateVector()

    prev_state.robot_pos = np.array([1.0, 0.0, math.pi/2])
    prev_state.add_firsttime_detection(0, np.array([2, 3, math.pi]))
    prev_state.add_firsttime_detection(1, np.array([2, 1, math.pi/2]))
    prev_state.add_firsttime_detection(2, np.array([1, 0, 0]))
    motion_ctrl = np.zeros(3)

    localization_node = LocalizationNode(prev_state)

    detections = {}
    detections[0] = np.array([3, -1, math.pi/2])
    detections[1] = np.array([1, -1, 0])
    detections[2] = np.array([0, 0, -math.pi/2])


    F, G, H, Q, R, new_state, new_covariance = runKalmanFilter(prev_state, prev_motion_ctrl=motion_ctrl, detections=detections)
    print("F = \n{}".format(F))
    print("G = \n{}".format(G))
    print("H = \n{}".format(H))
    print("Q = \n{}".format(Q))
    print("R = \n{}".format(R))
    print("prev_state = \n{}".format(prev_state.get_vector()))
    print("new_state = \n{}".format(new_state))
    print("new_covariance = \n{}".format(new_covariance))


