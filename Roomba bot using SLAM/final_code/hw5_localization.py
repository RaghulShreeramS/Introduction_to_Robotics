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
from geometry_msgs.msg import TwistStamped, Pose
import numpy as np
import tf_conversions
from CustomKalmanFilter import KalmanFilter
import time
import math
import pickle as pkl
from threading import Lock

TIMESTEP = 0.05

Q_ROBOT = np.array([0.0005, 0.0005, 0.01])
Q_MARKER = np.array([0.00001, 0.00001, 0.0001])
R_MARKER = np.array([0.0025, 0.0025, 0.01])


class StateVector:
    def __init__(self, robot_pos=np.zeros(3)):
        self.robot_pos = robot_pos
        self.covariance = np.diag([0.0, 0.0, 0.0])
        self.markers = {}

        self.state_history = [self.robot_pos]
        self.covariance_history = [self.covariance]
        self.id_history = [[]]

        self.lock = Lock()
    
    # Ignore lock param for pickle
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add lock back since it doesn't exist in the pickle
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

        self.state_history.append(self.get_vector())

    def update_covariance(self, new_covariance):
        self.covariance = new_covariance
        self.covariance_history.append(self.covariance)

    def add_firsttime_detection(self, tag_id, marker_location):
        self.lock.acquire()
        self.markers[tag_id] = marker_location
        sorted_tag_idx = 3*(sorted(self.markers.keys()).index(tag_id) + 1) # +1 bc of robot covariances
        
        # Expand covariance matrix
        orig_size = self.covariance.shape[0]
        new_covariance = np.insert(self.covariance, [sorted_tag_idx], np.zeros((3, orig_size)), axis=0)
        new_covariance = np.insert(new_covariance, [sorted_tag_idx], np.zeros((orig_size + 3, 3)), axis=1)

        init_marker_covariance = 1e10 * np.ones(3)
        new_covariance[sorted_tag_idx + 0, sorted_tag_idx + 0] = init_marker_covariance[0]
        new_covariance[sorted_tag_idx + 1, sorted_tag_idx + 1] = init_marker_covariance[1]
        new_covariance[sorted_tag_idx + 2, sorted_tag_idx + 2] = init_marker_covariance[2]

        self.covariance = new_covariance
        self.lock.release()

    def filterPredict(self, control_values):
        filter = createKalmanFilter(self, {})
        filter.predict(u=control_values)
        self.update_state(filter.x)
        self.update_covariance(filter.P)
        self.id_history.append(list(self.markers.keys()))
    
    def filterUpdate(self, detections):
        sorted_keys = sorted(detections.keys())
        feat_vec = np.array([detections[tag_id] for tag_id in sorted_keys]).flatten()

        filter = createKalmanFilter(self, detections)
        filter.update(z=feat_vec)
        self.update_state(filter.x)
        self.update_covariance(filter.P)
        self.id_history.append(list(self.markers.keys()))


class LocalizationNode:
    def __init__(self, state):
        # type: (LocalizationNode, StateVector) -> None
        self.sub_april = rospy.Subscriber("/apriltag_detection_array", AprilTagDetectionArray, self.april_detection_callback, queue_size=1)
        self.sub_control = rospy.Subscriber("/robot_control", TwistStamped, self.robot_control_callback, queue_size=1)
        self.pub_pose = rospy.Publisher("/robot_pose", Pose, queue_size=1)

        self.state = state
        self.velocities = [] # u_t
        self.filter_lock = Lock()

        # Camera topic delay: Delay from lastest camera output to apriltag_detection output is about 0.25 - 0.3 seconds
        # Detection topic delay: Delay from latest detection output to when it is read in this code is 0.25 seconds max.
        # self.camera_delay = 0.5 # seconds
        self.camera_delay = 0.9 # seconds
        self.delays = []
    
    def robot_control_callback(self, twist_control_msg):
        control_time = twist_control_msg.header.stamp.to_sec() # Pretty much the same as rospy.Time.now()
        # print("Now vs. control time diff:", rospy.Time.now().to_sec() - control_time)
        # self.delays.append(rospy.Time.now().to_sec() - control_time)
        twist = twist_control_msg.twist
        velocity = np.array([twist.linear.x, twist.linear.y, twist.angular.z])
        self.velocities.append((control_time, velocity))

        # Probably want to update Kalman filter prediction (do we?) # TODO: check delay time
        self.filter_lock.acquire()
        self.update_kalman_filter_to_time(rospy.Time.now().to_sec(), delay=self.camera_delay) # rospy.Time.now().to_sec()
        self.publish_robot_position()
        self.filter_lock.release()

    
    # Should measurement time be rospy.Time.now()?
    def update_kalman_filter_to_time(self, measurement_time, delay):
        new_velocities = []
        for control_time, velocity in self.velocities:
            if control_time < measurement_time - delay:
                self.state.filterPredict(velocity)
            else:
                new_velocities.append((control_time, velocity))
        self.velocities = new_velocities


    def april_detection_callback(self, april_detections):
        image_time = april_detections.header.stamp.to_sec()
        cur_robot_pose = self.state.robot_pos
        num_detections = len(april_detections.detections)
        if num_detections == 0:
            print("*** NO WAYPOINTS DETECTED ***")
            return
        print("Detected ids: {}".format([d.id for d in april_detections.detections]))
        # print("Now vs. detection time diff:", )
        self.delays.append(rospy.Time.now().to_sec() - image_time)

        detected_locations = {}
        for detection in april_detections.detections:
            # Add new detection
            if detection.id not in self.state.markers.keys():
                marker_location = getNewMarker(detection.pose, cur_robot_pose) # in world coords
                self.filter_lock.acquire()
                self.state.add_firsttime_detection(detection.id, marker_location)
                self.filter_lock.release()
                # print("NEW MARKER WORLD LOCATION", detection.id, marker_location)
            
            # Get relative distance to robot
            euler = tf_conversions.transformations.euler_from_quaternion([detection.pose.orientation.w, detection.pose.orientation.x, detection.pose.orientation.y, detection.pose.orientation.z], axes='sxyz') 
            tag_angle_robot = clip_angle(euler[1])

            location = np.array([detection.pose.position.z, -detection.pose.position.x, tag_angle_robot]) # convert camera [z, x] to robot [x, y]
            detected_locations[detection.id] = location
        # print('detected_locations:', detected_locations)

        # Update kalman filter guess with detection
        self.filter_lock.acquire()
        self.update_kalman_filter_to_time(rospy.Time.now().to_sec(), self.camera_delay)
        self.state.filterUpdate(detected_locations)
        # self.publish_robot_position()
        self.filter_lock.release()


    def publish_robot_position(self):
        robot_position = np.array(self.state.robot_pos, copy=True)
        for _, velocity in self.velocities:
            robot_position += velocity * TIMESTEP

        robot_pose = Pose()
        robot_pose.position.x = robot_position[0]
        robot_pose.position.y = robot_position[1]
        robot_pose.orientation.z = robot_position[2] # Has angular orientation, not quaternion
        self.pub_pose.publish(robot_pose)

    def save_history(self):
        print("Saving state history...")
        if len(self.delays) != 0:
            print("Max delay time:", np.max(self.delays))
            print("Avg delay time:", np.mean(self.delays))
        else:
            print("No camera, no delay!")
        with open('./state_history.pkl', 'wb') as f:
            pkl.dump(self.state.state_history, f)
        with open('./covariance_history.pkl', 'wb') as f:
            pkl.dump(self.state.covariance_history, f)
        with open('./id_history.pkl', 'wb') as f:
            pkl.dump(self.state.id_history, f)

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


def createKalmanFilter(state, detections):
    F, G, H, Q, R = getKalmanMatrices(state, detections)
    num_detections = len(detections)
    if num_detections > 0:
        filter = KalmanFilter(dim_x=len(state), dim_z=3*num_detections, dim_u=3)
    else:
        filter = KalmanFilter(dim_x=len(state), dim_z=1, dim_u=3)
    filter.x = state.get_vector()
    filter.P = state.covariance
    filter.F = F
    filter.B = G
    filter.H = H # = None if no feat_vec
    filter.Q = Q
    filter.R = R # = None if no feat_vec
    return filter


def getKalmanMatrices(prev_state, detections):
    F = np.identity(len(prev_state))
    G = np.identity(3) * TIMESTEP
    G = np.vstack([G, np.zeros((3*len(prev_state.markers), 3))])

    # Create H
    if len(detections) == 0:
        H = None
    else:
        H = np.zeros((3*len(detections), len(prev_state)))
        sorted_marker_ids = sorted(prev_state.markers.keys())
        sorted_detection_ids = sorted(detections.keys())

        for idx, tag_id in enumerate(sorted_detection_ids):
            tag_idx = sorted_marker_ids.index(tag_id)
            tag_vector = np.zeros(len(prev_state) - 3)
            tag_vector_x, tag_vector_y, tag_vector_theta = np.copy(tag_vector), np.copy(tag_vector), np.copy(tag_vector)
            tag_vector_x[3*tag_idx] = 1
            tag_vector_y[3*tag_idx + 1] = 1
            tag_vector_theta[3*tag_idx + 2] = 1

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
    robot_mean_err = np.array(Q_ROBOT, copy=True)
    tag_world_mean_err = np.array(Q_MARKER, copy=True)
    Q_values = np.concatenate([robot_mean_err, np.tile(tag_world_mean_err, len(prev_state.markers))]) # TODO: squared
    Q = np.diag(Q_values)
    # Q = np.diag(np.zeros(len(Q_values))) # testing no movement

    # Create R
    tag_detect_mean_err = np.array(R_MARKER, copy=True)
    R_values = tag_detect_mean_err # TODO: squared
    if len(detections) > 0:
        R = np.diag(np.tile(R_values, len(detections)))
    else:
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
    tag_angle_world = clip_angle(tag_angle_world)

    return np.array([tag_pos_world[0], tag_pos_world[1], tag_angle_world])

def clip_angle(angle):
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle <= -np.pi:
        angle += 2 * np.pi
    return angle


if __name__ == "__main__":
    localization_node = LocalizationNode(StateVector())
    rospy.init_node("localization")
    rospy.on_shutdown(localization_node.save_history)
    rospy.spin()
