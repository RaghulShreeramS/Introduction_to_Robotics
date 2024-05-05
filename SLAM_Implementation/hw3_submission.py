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
from localization import LocalizationNode, StateVector
import pickle as pkl

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

    # listener = tf.TransformListener()
    localization = LocalizationNode(state=StateVector())

    # No movement
    # waypoint = np.array([[1.0,0.0,0.0]])

    # Go forward
    # waypoint = np.array([[1.0,0.0,0.0]])

    # Rotate
    # waypoint = np.array([[0.0,0.0,0.0], 
                        #  [0.0,0.0,np.pi/2]])

    # Move then rotate
    # waypoint = np.array([[0.0,0.0,0.0], 
                        #  [1.0,0.0,0.0], 
                        #  [1.0,0.0,np.pi/2]])
    # waypoint = np.array([[0.0,0.0,0.0], 
    #                      [1.0,0.0,0.0], 
    #                      [1.0,0.0,np.pi]])

    # Rotate then forwards
    waypoint = np.array([[0.0,0.0,np.pi/2], 
                         [0.0,1.00,np.pi/2]])

    # Forward, then slide
    # waypoint = np.array([[0.0,0.0,np.pi/2], 
    #                      [1.0,0.0,np.pi/2], 
    #                      [1.0,1.0,np.pi/2]])

    # Quarter square
    # waypoint = np.array([[0.0,0.0,0.0], 
    #                      [1.0,0.0,0.0], 
    #                      [1.0,0.0,np.pi/2], 
    #                      [1.0,1.0,np.pi/2]])

    # # Half of square movement
    # waypoint = np.array([[0.0,0.0,0.0], 
    #                      [1.0,0.0,0.0], 
    #                      [1.0,0.0,np.pi/2], 
    #                      [1.0,1.0,np.pi/2], 
    #                      [1.0,1.0,np.pi]])

    # 1m x 1m square 
    # side = 1.0
    # waypoint = np.array([[0.0,0.0,0.0], 
    #                      [side,0.0,0.0], 
    #                      [side,0.0,np.pi/2], 
    #                      [side,side,np.pi/2], 
    #                      [side,side,np.pi], 
    #                      [0.0,side,np.pi],
    #                      [0.0,side,-np.pi/2],
    #                      [0.0,0.0,-np.pi/2],
    #                      [0.0,0.0,0.0]])

    # 1m x 1m square loop x2
    # side = 1.0
    # waypoint = np.array([[0.0,0.0,0.0], 
    #                      [side,0.0,0.0], 
    #                      [side,0.0,np.pi/2], 
    #                      [side,side,np.pi/2], 
    #                      [side,side,np.pi], 
    #                      [0.0,side,np.pi],
    #                      [0.0,side,-np.pi/2],
    #                      [0.0,0.0,-np.pi/2],
    #                      [0.0,0.0,0.0], 
    #                      [side,0.0,0.0], 
    #                      [side,0.0,np.pi/2], 
    #                      [side,side,np.pi/2], 
    #                      [side,side,np.pi], 
    #                      [0.0,side,np.pi],
    #                      [0.0,side,-np.pi/2],
    #                      [0.0,0.0,-np.pi/2]])

    # Octagon path
    side = 0.5
    waypoint = np.array([[0.0, 0.0, 0.0], 
                         [side, 0.0, 0.0],
                         [side, 0.0, np.pi/4], 
                         [side + side/np.sqrt(2), side/np.sqrt(2), np.pi/4], 
                         [side + side/np.sqrt(2), side/np.sqrt(2), np.pi/2], 
                         [side + side/np.sqrt(2), side + side/np.sqrt(2), np.pi/2], 
                         [side + side/np.sqrt(2), side + side/np.sqrt(2), 3*np.pi/4],
                         [side, side + side/np.sqrt(2) + side/np.sqrt(2), 3*np.pi/4], 
                         [side, side + side/np.sqrt(2) + side/np.sqrt(2), np.pi], 
                         [0, side + side/np.sqrt(2) + side/np.sqrt(2), np.pi], 
                         [0, side + side/np.sqrt(2) + side/np.sqrt(2), 5*np.pi/4], 
                         [-side/np.sqrt(2), side + side/np.sqrt(2), 5*np.pi/4], 
                         [-side/np.sqrt(2), side + side/np.sqrt(2), 3*np.pi/2], 
                         [-side/np.sqrt(2), side/np.sqrt(2), 3*np.pi/2], 
                         [-side/np.sqrt(2), side/np.sqrt(2), 7*np.pi/4], 
                         [0, 0, 7*np.pi/4], 
                         [0, 0, 2*np.pi]])


    # HW 2
    # waypoint = np.array([[0.0,0.0,0.0], 
    #                      [1.0,0.0,0.0],
    #                      [1.0,2.0,np.pi],
    #                      [0.0,0.0,0.0]])

    # init pid controller
    pid = PIDcontroller(0.1,0.005,0.005)

    # init current state
    current_state = np.array([0.0,0.0,0.0])
    trajectory = [current_state]

    # calibration = np.array([0.56, 0.56, 3.1]) #2.6 last night
    calibration = np.array([0.56, 0.56, 4.6]) * 0.6 *0.9# 4.6 low battery, 5.1 good battery

    timestep = 0.05
    # timestep = 0.1

    # in this loop we will go through each way point.
    # once error between the current state and the current way point is small enough, 
    # the current way point will be updated with a new point.
    for wp in waypoint:
        print("move to way point", wp)
        # set wp as the target point
        pid.setTarget(wp)

        # calculate the current twist
        update_value = pid.update(current_state)
        # publish the twist
        motion_ctrl = coord(update_value, current_state)
        pub_twist.publish(genTwistMsg(motion_ctrl))
        #print(coord(update_value, current_state))
        time.sleep(timestep)
        # update the current state using Kalman filter

        # From HW2
        delta_x = calibration * update_value
        expected_velocity = delta_x / 0.05
        full_state = localization.updateState(expected_velocity)
        current_state = full_state.robot_pos
        trajectory.append(full_state.get_vector())

        while(np.linalg.norm(pid.getError(current_state, wp)) > 0.05): # check the error between current state and current way point
            # calculate the current twist
            update_value = pid.update(current_state)
            # publish the twist
            motion_ctrl = coord(update_value, current_state)
            pub_twist.publish(genTwistMsg(motion_ctrl))
            time.sleep(timestep)
            # update the current state using Kalman filter
            # From HW2
            delta_x = calibration * update_value
            expected_velocity = delta_x / 0.05

            full_state = localization.updateState(expected_velocity)
            current_state = full_state.robot_pos
            trajectory.append(full_state.get_vector())
    # stop the car and exit
    pub_twist.publish(genTwistMsg(np.array([0.0,0.0,0.0])))
    
    print("Final state: ", current_state)
    with open('./trajectory.pkl', 'wb') as f:
        pkl.dump(trajectory, f)
    localization.save_covariances()
