#!/usr/bin/env python
import sys
import roslib
import rospy
import geometry_msgs.msg
from geometry_msgs.msg import Twist
import numpy as np
import math
import tf
import tf2_ros
from tf.transformations import quaternion_matrix
import time

# Copied from hw2 solution

"""
The class of the pid controller.
"""
class PIDcontroller:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0,0.0,0.0])
        self.lastError = np.array([0.0,0.0,0.0])
        self.timestep = 0.1
        self.maximumValue = 0.02

    def setTarget(self, targetx, targety, targetw):
        """
        set the target pose.
        """
        self.I = np.array([0.0,0.0,0.0]) 
        self.lastError = np.array([0.0,0.0,0.0])
        self.target = np.array([targetx, targety, targetw])

    def setTarget(self, state):
        """
        set the target pose.
        """
        self.I = np.array([0.0,0.0,0.0]) 
        self.lastError = np.array([0.0,0.0,0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState):
        """
        return the different between two states
        """
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result 

    def setMaximumUpdate(self, mv):
        """
        set maximum velocity for stability.
        """
        self.maximumValue = mv

    def update(self, currentState):
        """
        calculate the update value on the state based on the error between current state and target state with PID.
        """
        e = self.getError(currentState, self.target)

        P = self.Kp * e
        self.I = self.I + self.Ki * e * self.timestep 
        I = self.I
        D = self.Kd * (e - self.lastError)
        result = P + I + D

        self.lastError = e

        # scale down the twist if its norm is more than the maximum value. 
        resultNorm = np.linalg.norm(result)
        if(resultNorm > self.maximumValue):
            result = (result / resultNorm) * self.maximumValue
            self.I = 0.0

        return result

def getCurrentPos(l):
    """
    Given the tf listener, we consider the camera's z-axis is the header of the car
    """
    br = tf.TransformBroadcaster()
    result = None
    foundSolution = False

    for i in range(0, 15):
        camera_name = "camera_" + str(i)
        if l.frameExists(camera_name):
            try:
                # print("Using camera", camera_name)
                now = rospy.Time()
                # wait for the transform ready from the map to the camera for 1 second.
                l.waitForTransform("world", camera_name, now, rospy.Duration(1.0))
                # extract the transform camera pose in the map coordinate.
                (trans, rot) = l.lookupTransform("world", camera_name, now)
                # convert the rotate matrix to theta angle in 2d
                matrix = quaternion_matrix(rot)
                angle = math.atan2(matrix[1][2], matrix[0][2])
                # this is not required, I just used this for debug in RVIZ
                # br.sendTransform((trans[0], trans[1], 0), tf.transformations.quaternion_from_euler(0,0,angle), rospy.Time.now(), "base_link", "world")
                result = np.array([trans[0], trans[1], angle])
                foundSolution = True
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, tf2_ros.TransformException):
                print("meet error")
    listener.clear()
    return foundSolution, result


def genTwistMsg(desired_twist):
    """
    Convert the twist to twist msg.
    """
    twist_msg = Twist()
    twist_msg.linear.x = desired_twist[0] 
    twist_msg.linear.y = desired_twist[1] 
    twist_msg.linear.z = 0
    twist_msg.angular.x = 0
    twist_msg.angular.y = 0
    twist_msg.angular.z = desired_twist[2]
    return twist_msg

def coord(twist, current_state):
    """
    Convert the twist into the car coordinate
    """
    J = np.array([[np.cos(current_state[2]), np.sin(current_state[2]), 0.0],
                  [-np.sin(current_state[2]), np.cos(current_state[2]), 0.0],
                  [0.0,0.0,1.0]])
    return np.dot(J, twist)
    


if __name__ == "__main__":
    import time
    rospy.init_node("hw2")
    pub_twist = rospy.Publisher("/twist", Twist, queue_size=1)

    listener = tf.TransformListener()
    
    # waypoint = np.load("/root/rb5_ws/src/rb5_ros/planning/src/shortest_path.npy")[1:]
    waypoint = np.load("/root/rb5_ws/src/rb5_ros/planning/src/safest_path.npy")[1:]

    # init pid controller
    pid = PIDcontroller(0.1,0.005,0.005)

    # init current state
    current_state = np.array([2.4,0.6,-np.pi/2])
    trajectory = [np.copy(current_state)]
    # calibration = np.array([0.65, 0.65, 5.1]) * 0.6 *0.9 * 1.1 * 1.1# 4.6 low battery, 5.1 good battery
    calibration = np.array([0.65, 0.65, 3.9]) * 0.6 *0.9 * 1.1 * 1.1# 4.6 low battery, 5.1 good battery

    start = time.time()
    # in this loop we will go through each way point.
    # once error between the current state and the current way point is small enough, 
    # the current way point will be updated with a new point.
    for wp in waypoint:
        print("move to way point", wp)
        # set wp as the target point
        pid.setTarget(wp)

        # calculate the current twist
        update_value = pid.update(current_state)
        # update_value = 0
        # publish the twist
        pub_twist.publish(genTwistMsg(coord(update_value, current_state)))
        # print(coord(update_value, current_state))
        time.sleep(0.05)
        # update the current state
        current_state += update_value * calibration
        found_state, estimated_state = getCurrentPos(listener)
        if found_state: # if the tag is detected, we can use it to update current state.
            current_state = estimated_state
        trajectory.append(np.copy(current_state))
        # print(current_state)
        while(np.linalg.norm(pid.getError(current_state, wp)) > 0.125): # check the error between current state and current way point
            # calculate the current twist
            update_value = pid.update(current_state)
            # update_value = 0
            # publish the twist
            pub_twist.publish(genTwistMsg(coord(update_value, current_state)))
            # print(coord(update_value, current_state))
            time.sleep(0.05)
            # update the current state
            current_state += update_value * calibration
            found_state, estimated_state = getCurrentPos(listener)
            if found_state:
                current_state = estimated_state
            trajectory.append(np.copy(current_state))
            # print(current_state)
    # stop the car and exit
    pub_twist.publish(genTwistMsg(np.array([0.0,0.0,0.0])))

    np.save("h4_trajectory.npy", trajectory)
    print("Elapsed time", time.time() - start)