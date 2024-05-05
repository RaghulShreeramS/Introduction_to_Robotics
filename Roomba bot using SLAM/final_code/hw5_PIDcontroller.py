#!/usr/bin/env python
import sys
import roslib
import rospy
import geometry_msgs.msg
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, Pose
import numpy as np
import math
import tf
import tf2_ros
from tf.transformations import euler_from_quaternion
import pickle as pkl

TIMESTEP = 0.05

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
        # self.maximumValue = 0.025

        # self.min_velocities = np.array([0.019, 0.0, 0.0])
        # self.ignore_threshold = np.array([0.006, 100.0, 100.0])

        self.robot_pose = None
        self.robot_pose_update = False
        self.sub_base_link = rospy.Subscriber("/robot_pose", Pose, self.pose_callback, queue_size=1)

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
        
        # print("Original update:", result)
        # mask = np.abs(result) < self.ignore_threshold
        # clipped_result = np.sign(result) * np.clip(np.abs(result), self.min_velocities, None)
        # result = mask*result + np.logical_not(mask)*clipped_result

        return result

    
    def pose_callback(self, msg):
        # position = msg.position
        # pose = np.array([position.x, position.y, position.z])
        # orientation = msg.orientation
        # orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        # euler = euler_from_quaternion(orientation_list) # (roll, pitch, yaw)
        # self.robot_pose = np.array([pose[0], pose[1], euler[2]])

        self.robot_pose = np.array([msg.position.x, msg.position.y, msg.orientation.z])
        self.robot_pose_update = True
        # self.robot_pose_t = msg.header.stamp.to_sec()
    
    def update_current_pose(self, current_pose):
        # Update the current pose
        # Given that we published the speed before sleep
        # This should give enough time for the Kalman Filter
        # to predict and publish predicted pose
        # We always use the latest pose

        if self.robot_pose_update == True:
            current_pose = self.robot_pose
            self.robot_pose_update = False
            rospy.loginfo("Get robot pose update: {}".format(current_pose))
        else:
            rospy.loginfo("No robot update, please check the Kalman Filter node.")
        return current_pose


def speed_to_twist(speed):
    twist_msg = TwistStamped()
    twist_msg.header.stamp = rospy.Time.now()
    twist_msg.twist.linear.x = speed[0]
    twist_msg.twist.linear.y = speed[1]
    twist_msg.twist.linear.z = 0
    twist_msg.twist.angular.x = 0
    twist_msg.twist.angular.y = 0
    twist_msg.twist.angular.z = speed[2]
    return twist_msg

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

def sleep(timestep=TIMESTEP):
    try: # Try to do this
        time.sleep(timestep)
    except KeyboardInterrupt: # In the case of keyboard interrupt
        rospy.signal_shutdown("Ctrl-C")
        exit()

if __name__ == "__main__":
    import time
    rospy.init_node("hw2")
    pub_twist = rospy.Publisher("/twist", Twist, queue_size=1)
    pub_speed = rospy.Publisher("/robot_control", TwistStamped, queue_size=1)

    # HW 5
    waypoint = np.load("/root/rb5_ws/src/rb5_ros/planning/roomba_path.npy") 

    # init pid controller
    pid = PIDcontroller(0.1,0.005,0.005)

    # init current state
    current_state = np.array([0.0,0.0,0.0])

    # calibration = np.array([0.56, 0.56, 3.1]) #2.6 last night
    calibration = np.array([0.56, 0.56, 4.2]) * 0.6 *0.9   *1.4# 4.6 low battery, 5.1 good battery
    # calibration_smalldist = calibration * [0.21, 1, 1]# 4.6 low battery, 5.1 good battery

    # in this loop we will go through each way point.
    # once error between the current state and the current way point is small enough, 
    # the current way point will be updated with a new point.
    for wp in waypoint:
        print("move to way point", wp)

        # set wp as the target point
        pid.setTarget(wp)

        while(np.linalg.norm(pid.getError(current_state, wp)) > 0.10): # check the error between current state and current way point
        # while(np.linalg.norm(pid.getError(current_state, wp)[:2]) > 0.07 or pid.getError(current_state, wp)[2] > 0.14): # check the error between current state and current way point
                # calculate the current twist
                update_value = pid.update(current_state)
                # print("Update value: {}".format(update_value))
                delta_x = calibration * update_value # From HW2
                # print("Calib. update value: {}".format(update_value))
                expected_velocity = delta_x / 0.05
                # print("Expected velocity: {}".format(expected_velocity))
                # publish the twist
                twist_msg = speed_to_twist(expected_velocity)
                motion_ctrl = coord(update_value, current_state)
                pub_speed.publish(twist_msg) # notify kalman filter
                pub_twist.publish(genTwistMsg(motion_ctrl)) # notify motors

                sleep()
                # update the current state using Kalman filter
                current_state = pid.update_current_pose(current_state)

    # stop the car and exit
    pub_twist.publish(genTwistMsg(np.array([0.0,0.0,0.0])))
    print("Final state: ", current_state)
