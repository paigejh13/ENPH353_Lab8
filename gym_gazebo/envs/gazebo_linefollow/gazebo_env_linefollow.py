
import cv2
import gym
import math
import rospy
import roslaunch
import time
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from time import sleep

from gym.utils import seeding


class Gazebo_Linefollow_Env(gazebo_env.GazeboEnv):

    def __init__(self):
        # Launch the simulation with the given launchfile name
        LAUNCH_FILE = '/home/fizzer/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src/linefollow_ros/launch/linefollow_world.launch'
        gazebo_env.GazeboEnv.__init__(self, LAUNCH_FILE)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world',
                                              Empty)

        self.action_space = spaces.Discrete(3)  # F,L,R
        self.reward_range = (-np.inf, np.inf)
        self.episode_history = []

        self._seed()

        self.bridge = CvBridge()
        self.timeout = 0  # Used to keep track of images with no line detected

    def process_image(self, data):
        '''
            @brief Coverts data into a opencv image and displays it
            @param data : Image data from ROS

            @retval (state, done)
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        def find_line(frame):
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #set all pixels that make up the path to 0
            threshold = 114
            _, frame2 = cv2.threshold(frame1, threshold, 255, cv2.THRESH_BINARY)

            #blur the frame
            img_blur1 = cv2.GaussianBlur(frame2,(15,15),0)

            #take the difference of the blurred and non blurred image to filter noise
            #and identify edges of the path
            img_delta10 = img_blur1.astype(np.float32) - frame2.astype(np.float32)
            img_delta10 =  cv2.normalize(img_delta10, img_delta10, 0, 1, cv2.NORM_MINMAX)


            #identify pixel locations of the path's edge
            edgeList = list()
            edgeLoc = np.zeros(2)
            k = 0
            total = 0

            for i in range(1, img_delta10.shape[1]):
                if abs(img_delta10[239,i] - img_delta10[239,i-1]) > 0.32:
                    edgeList.append(i)
                    k = k+1

            for i in range(0, len(edgeList)):
                total = total + edgeList[i]

            #centre the circle on the average of path's edge pixel locations
            if k == 1:
                xVal = total/2
            elif k != 0:
                xVal = total/k
            else:
                return -1
                
            return xVal


        NUM_BINS = 3
        bar_width = cv_image.shape[1]/10
        state = 0
        done = False

        xVal = find_line(cv_image)

        if xVal ==-1:
            done = True
        else:
            remainder = xVal%bar_width
            block = xVal-remainder
            state = int(block/bar_width)

        return state, done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        self.episode_history.append(action)

        vel_cmd = Twist()

        if action[0] == 0:  # FORWARD
            vel_cmd.linear.x = 0.5
            vel_cmd.angular.z = 0.0
            
        elif action[0] == 1:  # LEFT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.4
            
        elif action[0] == 2:  # RIGHT
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = -0.4

        self.vel_pub.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw', Image,
                                              timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        state, done = self.process_image(data)

        # Set the rewards for your action
        if not done:
            if action == 0:  # FORWARD
                reward = 4
            elif action == 1:  # LEFT
                reward = 2
            else:
                reward = 2  # RIGHT
        else:
            reward = -200

        return state, reward, done, {}

    def reset(self):

        print("Episode history: {}".format(self.episode_history))
        self.episode_history = []
        print("Resetting simulation...")
        # Resets the state of the environment and returns an initial
        # observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        # read image data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/pi_camera/image_raw',
                                              Image, timeout=5)
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.timeout = 0
        state, done = self.process_image(data)

        return state
    
