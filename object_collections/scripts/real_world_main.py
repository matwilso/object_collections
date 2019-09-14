#!/usr/bin/env python3
import itertools
import os
import subprocess
import random
import time
import warnings
warnings.filterwarnings('ignore')
import copy
import yaml
import pyautogui

import numpy as np
import tensorflow as tf
import pyperclip

from object_collections.define_flags import FLAGS
from object_collections.scripts.network_handler import NetworkHandler
import object_collections.rl.util as rl_util
from object_collections.rl.data import rollout_to_tf_record
from object_collections.envs import unmap_continuous
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import scipy
cos_dist = scipy.spatial.distance.cosine

import rospy
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from collections import deque
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Bool, Header, Float64
from geometry_msgs.msg import Vector3, PoseStamped, Pose, Point, Quaternion
from tf2_msgs.msg import TFMessage
import quaternion
from sensor_msgs.msg import Joy
from baxter_paddle_moveit_config.srv import SimpleMove, SimpleMoveRequest, SimpleMoveResponse
from baxter_paddle_moveit_config.srv import SetGoalArea, SetGoalAreaRequest, SetGoalAreaResponse

from object_collections.envs.utils.sim import R, R3D
from object_collections.envs import TABLES
import object_collections.envs.utils.sim as sim_utils

#FLAGS['bs'] = 1
assert FLAGS['use_image'] == 1, "needs to be set before"
FLAGS['use_embed'] = 0
FLAGS['shuffle'] = 0
FLAGS['shuffle_files'] = 0
FLAGS['run_rl_optim'] = 0
FLAGS['threading'] = 0
FLAGS['is_training'] = 0

resize = lambda image: cv2.resize(image, FLAGS['image_shape'][:2], interpolation=cv2.INTER_AREA) / 255.0

# TODO: fix ordering of observation.  Should be sorting because sometimes it is wrong

def sample_area(x1, x2, y1, y2, np_random=np.random):
    object_xys = []  # for checking collisions against
    sample_range = R3D(R(x1, x2), R(y1, y2), R(0,0))
    DIAMETER = 0.0254

    for i in range(FLAGS['num_objects']):
        while True:
            xy = sim_utils.sample_xyz(np_random, sample_range)[:2]
            collision = False
            for other in object_xys:
                collision = collision or np.linalg.norm(xy - other) < (DIAMETER*2.0)

            if not collision:
                object_xys.append(xy)
                break
    return np.stack(object_xys)

class RVizImageDisplay:
    def __init__(self, network_handler):
        self.trial = 0
        self.mode = 'paused'
        self.main_pub = 'goal'
        self.recording = False
        self.bag = None
        self.count = 0
        self.plan_failures = 0
        self.goal_img_msg = None
        self.mid_query = False
        self.last_start = 0
        self.last_xbox = 0
        self.last_dph = 0

        self.phi_g = None
        self.phi_s = None
        self.goal_array = None
        rospy.init_node('rviz_image_display')
        self.rate = rospy.Rate(1)
        self.network_handler = network_handler

        self.mdn_pub = rospy.Publisher('/mdn_image', Image, queue_size=10)
        self.mdn_goal_pub = rospy.Publisher('/mdn_goal_image', Image, queue_size=10, latch=True)
        self.mdn_main_pub = rospy.Publisher('/mdn_main_image', Image, queue_size=10, latch=True)
        self.action_pub = rospy.Publisher('/action_marker', Marker, queue_size=10)
        self.state_marker_pub = rospy.Publisher('/state_marker', Marker, queue_size=10, latch=True)
        self.phi_pub = rospy.Publisher('/phi_dist', Float64, queue_size=10, latch=True)
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_cb, queue_size=1)
        self.goal_image_srv = rospy.Service('/set_goal_image', Trigger, self.set_goal_image)
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_cb, queue_size=1)
        # TODO: this may need to be replaced with more
        self.query_policy_src = rospy.Service('/query_policy', Trigger, self.query_policy)
        self.goal_image = None
        self.query_image = None
        #rospy.wait_for_service('simple_move')
        self.simple_move_proxy = rospy.ServiceProxy('simple_move', SimpleMove)
        self.set_goal_service = rospy.Service('set_goal_area', SetGoalArea, self.handle_set_goal)
        rospy.sleep(rospy.Duration(3.0))
        self.handle_set_goal(SetGoalAreaRequest(-0.2, 0.2, -0.1, 0.1))

    def handle_set_goal(self, req):
        if not any([req.x1, req.x2, req.y1, req.y2]):
            bounds = (-0.2, 0.2, -0.1, 0.1)
        else:
            bounds = req.x1, req.x2, req.y1, req.y2
        array = sample_area(*bounds)
        self._set_goal_state(array, bounds)
        return SetGoalAreaResponse(True)

    def run(self):
        rospy.loginfo('running')
        while not rospy.is_shutdown():
            if self.mode == 'running':
                success = self.query_policy(None).success
                self.count += 1
                if not success:
                    self.plan_failures += 1
                print('count: {}'.format(self.count))
            else:
                start_marker = self._make_paddle_marker([0.0, 0.0, 0.0], 0.0, start_end='start', delete=True)
                end_marker = self._make_paddle_marker([0.0, 0.0, 0.0], 0.0, start_end='end', delete=True)
                self.action_pub.publish(start_marker)
                self.action_pub.publish(end_marker)
            rospy.sleep(1.5)

    def set_goal_image(self, req):
        # TODO: some check or something that these are new enough or something
        self.goal_image = self.np_img.copy()
        #if self.recording:
        #    self.bag.write('mdn_goal', self.curr_out_msg)
        self.mdn_goal_pub.publish(self.curr_out_msg)
        return TriggerResponse(success=True)

    def _make_state_marker(self, array, bounds, color='cyan', delete=False):
        x1, x2, y1, y2 = bounds

        #arr_min = np.min(array, axis=0) 
        #arr_max = np.max(array, axis=0) 
        #arr_c = (arr_min + arr_max) / 2
        #arr_s = arr_max - arr_min
        arr_c = 0.5*TABLES[FLAGS['default_table']]['wood'][:2] * np.array([(x1+x2)/2, (y1+y2)/2])
        #arr_s = 0.5*TABLES[FLAGS['default_table']]['wood'][:2] * np.array([x2-x1, y2-y1])
        arr_s = 1.15*0.5*TABLES[FLAGS['default_table']]['wood'][:2] * np.array([x2-x1, y2-y1])

        marker = Marker()
        marker.header.frame_id = 'table_center'
        marker.header.stamp = rospy.Time.now()
        marker.id = 5
        marker.type = marker.CUBE
        if delete:
            marker.action = marker.DELETE
        else:
            marker.action = marker.ADD
        marker.pose.position.x = -arr_c[0] #-0.01
        marker.pose.position.y = -arr_c[1] #+ 0.03
        marker.pose.position.z = -0.01
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(arr_s[0], arr_s[1], 0.0254)
        if color == 'cyan':
            marker.color = ColorRGBA(0, 1, 1, 0.5)
            marker.ns = 'start'
        elif color == 'red':
            marker.color = ColorRGBA(1, 0, 0, 0.5)
            marker.ns = 'goal'
        return marker

    def _set_goal_state(self, array, bounds, color='cyan'):
        self.goal_array = array
        marker = self._make_state_marker(array, bounds, color=color)
        data, ev = self.network_handler.evaluate_array(self.goal_array)
        data = data.copy()
        data = data[:,::-1,:]
        data = cv2.resize(data, (2*480, 480))
        bits = data.tobytes()

        goal_img_msg = Image(header=Header(0, rospy.Time.now(), 'table_center'))

        goal_img_msg.width = 2*480
        goal_img_msg.height = 480
        goal_img_msg.step = 3*2*480
        goal_img_msg.data = bits
        goal_img_msg.encoding = 'rgb8'
        self.mdn_goal_pub.publish(goal_img_msg)
        self.state_marker_pub.publish(marker)
        self.goal_img_msg = goal_img_msg
        #if self.recording:
        #    self.bag.write('mdn_goal', goal_img_msg)
        #    self.bag.write('state_marker', marker)

    def sample_trial(self, trial, want_bounds=[]):
        xbound = 40/60.
        ybound = 100/121.

        gsx = (trial // 5) * 10 - 5
        gex = gsx + 20
        gsy = (trial % 5) * 20
        gey = gsy + 20

        def scale_shift(sx, ex, sy, ey):
            # adjust the sample to be tighter than the drawing will be
            ssx = sx + 5
            sex = ex - 5
            ssy = sy + 5
            sey = ey - 5

            # shift to env scale
            sx = (sx - 20) / 30.0
            ex = (ex - 20) / 30.0
            ssx = (ssx - 20) / 30.0
            sex = (sex - 20) / 30.0

            sy = (sy - 50) / 60.5
            ey = (ey - 50) / 60.5
            ssy = (ssy - 50) / 60.5
            sey = (sey - 50) / 60.5
            return sx, ex, ssx, sex, sy, ey, ssy, sey

        sx, ex, ssx, sex, sy, ey, ssy, sey = scale_shift(gsx, gex, gsy, gey)
        draw_bounds = np.array([sx, ex, sy, ey])
        sample_bounds = np.array([ssx, sex, ssy, sey])

        if want_bounds:
            return draw_bounds, sample_bounds
        else:
            np_random = np.random.RandomState(trial)
            blocks = sample_area(*sample_bounds, np_random=np_random)
            self._set_goal_state(blocks, draw_bounds, color='red')

        np_random = np.random.RandomState(trial)

        while True:
            start_x = np_random.uniform(0, 20) - 5
            start_y = np_random.uniform(0, 80)

            if abs(start_y - gsy) < 20:
                continue
            else:
                break


        estart_x = start_x + 20
        estart_y = start_y + 20

        sx, ex, ssx, sex, sy, ey, ssy, sey = scale_shift(start_x, estart_x, start_y, estart_y)
        draw_bounds = np.array([sx, ex, sy, ey])
        sample_bounds = np.array([ssx, sex, ssy, sey])

        #draw_bounds = [-0.8, 0.8, -0.92, 0.92]
        #sample_bounds = [-0.8, 0.8, -0.92, 0.92]
        blocks = sample_area(*sample_bounds, np_random=np_random)
        marker = self._make_state_marker(blocks, draw_bounds, color='cyan')
        #data, ev = self.network_handler.evaluate_array(blocks)
        #data = data.copy()
        #data = data[:,::-1,:]
        #data = cv2.resize(data, (2*480, 480))
        #bits = data.tobytes()
        #goal_img_msg = Image(header=Header(0, rospy.Time.now(), 'table_center'))
        #goal_img_msg.width = 2*480
        #goal_img_msg.height = 480
        #goal_img_msg.step = 3*2*480
        #goal_img_msg.data = bits
        #goal_img_msg.encoding = 'rgb8'
        #self.mdn_goal_pub.publish(goal_img_msg)
        #self.goal_img_msg = goal_img_msg
        self.state_marker_pub.publish(marker)


    def joy_cb(self, joy_msg):
        A = 0
        B = 1
        X = 2
        Y = 3
        LB = 4
        RB = 5
        SEL = 6
        START = 7
        XBOX = 8
        JLIN = 9
        JRIN = 10
        DPH = -2
        DPV = -1

        start = joy_msg.buttons[START]
        xbox = joy_msg.buttons[XBOX]
        dph = joy_msg.axes[DPH]

        if joy_msg.buttons[Y]:
            self.sample_trial(self.trial)

        if joy_msg.buttons[A]:
            self.set_goal_image(None)

        if joy_msg.buttons[B] and joy_msg.buttons[RB]:
            import ipdb; ipdb.set_trace()
        if start and (start != self.last_start):
            if self.mode == 'paused':
                self.mode = 'running'
            else:
                self.mode = 'paused'
        if joy_msg.buttons[SEL]:
            self.count = 0
            self.plan_failures = 0
            start_marker = self._make_paddle_marker([0.0, 0.0, 0.0], 0.0, start_end='start', delete=True)
            end_marker = self._make_paddle_marker([0.0, 0.0, 0.0], 0.0, start_end='end', delete=True)
            self.action_pub.publish(start_marker)
            self.action_pub.publish(end_marker)
            goal_marker = self._make_state_marker(None, [0.0, 1.0, 0.0, 1.0], color='red', delete=True)
            start_marker = self._make_state_marker(None, [0.0, 1.0, 0.0, 1.0], color='cyan', delete=True)
            self.state_marker_pub.publish(goal_marker)
            self.state_marker_pub.publish(start_marker)

        if joy_msg.buttons[LB]:
            if self.main_pub == 'image':
                self.main_pub = 'goal'
            else:
                self.main_pub = 'image'

        if xbox and (xbox != self.last_xbox):
            ssr_name = '/home/matwilso/Desktop/trial.mp4'
            folder = os.path.join(os.getcwd(), 'bags/{}/'.format(FLAGS['suffix']))
            bags = os.path.join(folder, 'bags/')
            vids = os.path.join(folder, 'vids/')
            yamls = os.path.join(folder, 'yamls/')
            os.makedirs(bags, exist_ok=True)
            os.makedirs(vids, exist_ok=True)
            os.makedirs(yamls, exist_ok=True)
            bagname = os.path.join(bags, '{}.bag'.format(self.trial))
            vidname = os.path.join(vids, '{}.mp4'.format(self.trial))
            imgname = os.path.join(vids, '{}.png'.format(self.trial))

            if self.recording:
                pyautogui.hotkey('ctrl','shift','alt', 'r')
                self.recording = False
                #self.bag.close()
                print('{} bag closed'.format(bagname))
                #self.bag = None
                #self.p.stdin.write(b'q')
                #self.p.stdin.close()
                #time.sleep(0.1)
                #self.p.kill()
                pyautogui.click(2343, 10)
                pyautogui.click(2401, 84)
                os.rename(ssr_name, vidname)
                pyperclip.copy(vidname)
                if not FLAGS['value_goal']:
                    plt.imsave(imgname, self.goal_image)
            else:
                try:
                    os.remove(ssr_name)
                except:
                    pass
                #self.bag = rosbag.Bag(bagname, 'w')
                self.recording = True
                #cmd = 'ffmpeg -f x11grab -y -r 30 -s 2560x1440 -i :0.0 ' + vidname
                #self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
                pyautogui.hotkey('ctrl','shift','alt', 'r')
                yamlname = os.path.join(yamls, '{}.yaml'.format(self.trial))
                with open(yamlname, 'w') as f:
                    yaml.dump({'count': self.count, 'plan_failures': self.plan_failures}, f)

                rospy.sleep(2.0)
                self.mode = 'running'
            #self.count = 0
            #self.plan_failures = 0

        if dph != self.last_dph and dph > 0.0:
            self.trial = (self.trial - 1) % 15
        if dph != self.last_dph and dph < 0.0:
            self.trial = (self.trial + 1) % 15

        if joy_msg.axes[DPV] > 0.0:
            self.count -= 1
        elif joy_msg.axes[DPV] < 0.0:
            self.count += 1

        self.last_start = start
        self.last_xbox = xbox
        self.last_dph = dph

        print('mode: {}, recording: {}, trial: {} count: {}'.format(self.mode, self.recording, self.trial, self.count))

    def _make_paddle_marker(self, xyz, yaw, start_end='start', delete=False, right=True):
        if start_end == 'start':
            id = 0
            if right:
                color = ColorRGBA(0, 1, 0, 1)
            else:
                color = ColorRGBA(1, 1, 1, 1)
        else:
            id = 1
            color = ColorRGBA(1, 0, 0, 1)
        marker = Marker()
        marker.header.frame_id = 'table_center'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'action'
        marker.id = id
        marker.type = marker.CUBE
        if delete:
            marker.action = marker.DELETE
        else:
            marker.action = marker.ADD
        marker.pose.position.x = xyz[0]
        marker.pose.position.y = xyz[1]
        marker.pose.position.z = xyz[2]
        quat = quaternion.from_euler_angles(0, 0, yaw).components
        marker.pose.orientation.x = quat[1]
        marker.pose.orientation.y = quat[2]
        marker.pose.orientation.z = quat[3]
        marker.pose.orientation.w = quat[0]
        marker.scale = Vector3(0.02, 0.10, 0.2)
        marker.color = color
        return marker

    def query_policy(self, req):
        rospy.wait_for_message('/camera/rgb/image_raw', Image)
        rospy.sleep(0.5)
        query_image = self.np_img.copy()
        if FLAGS['value_goal']:
            action, info = self.network_handler.query_policy_array(query_image, self.goal_array)
        else:
            assert self.goal_image is not None, "Have to /set_goal_image before querying policy"
            goal_image = self.goal_image.copy()
            action, info = self.network_handler.query_policy(query_image, goal_image)

        act_names = FLAGS['act_names']
        x = action[act_names.index('x')]
        y = action[act_names.index('y')]
        yaw = action[act_names.index('yaw')]
        dist = action[act_names.index('dist')]

        x = unmap_continuous('x', x, FLAGS)
        y = unmap_continuous('y', y, FLAGS)
        yaw = unmap_continuous('yaw', yaw, FLAGS)
        dist = unmap_continuous('dist', dist, FLAGS)
        #print(yaw)
        #import ipdb; ipdb.set_trace()
        #sign_yaw = np.sign(yaw)
        #yaw = sign_yaw * (np.abs(yaw + np.pi/2) % np.pi)
        targetx, targety = x + np.cos(yaw)*dist,  y + np.sin(yaw)*dist

        # compute rotation matrix to rotate stuff into correct tf frame manually
        ang = -np.pi
        rotation = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        xy = rotation.dot([x,y])
        targetxy = rotation.dot([targetx, targety])
        # add angle and wrap-around
        yaw += ang
        if yaw > np.pi:
            yaw -= 2*np.pi 
        elif yaw < np.pi:
            yaw += 2*np.pi

        if (xy[1] + targetxy[1]) / 2.0 >= 0.00:
            right = False
        else:
            right = True

        # visualize
        start_marker = self._make_paddle_marker([xy[0], xy[1], 0.0], yaw, start_end='start', right=right)
        self.action_pub.publish(start_marker)
        end_marker = self._make_paddle_marker([targetxy[0], targetxy[1], 0.0], yaw, start_end='end')
        self.action_pub.publish(end_marker)
        #if self.recording:
        #    self.bag.write('start_marker', start_marker)
        #    self.bag.write('end_marker', end_marker)

        # Set pose goal
        quat = [0,0,0,1]
        start_xyz = Point(xy[0], xy[1], 0.0)
        upstart = copy.deepcopy(start_xyz)
        upstart.z += 0.1
        end_xyz = Point(targetxy[0], targetxy[1], 0.0)
        upend = copy.deepcopy(end_xyz)
        upend.z += 0.1

        pre_pose = PoseStamped(Header(0, rospy.Time.now(), 'table_center'), Pose(upstart, Quaternion(*quat)))
        start_pose = PoseStamped(Header(0, rospy.Time.now(), 'table_center'), Pose(start_xyz, Quaternion(*quat)))
        end_pose = PoseStamped(Header(0, rospy.Time.now(), 'table_center'), Pose(end_xyz, Quaternion(*quat)))
        post_pose = PoseStamped(Header(0, rospy.Time.now(), 'table_center'), Pose(upend, Quaternion(*quat)))
        dry_run = False

        self.mid_query = True
        ok = self.simple_move_proxy(SimpleMoveRequest(start_pose, end_pose, yaw, dry_run)).success
        self.mid_query = False

        return TriggerResponse(success=ok)

    def image_cb(self, img_msg):
        #if self.recording:
        #    self.bag.write('/camera/rgb/image_raw', img_msg)
        out_msg = copy.deepcopy(img_msg)
        np_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
        self.np_img = resize(np_img)
        data, ev = self.network_handler.evaluate([self.np_img], self.goal_array)
        data = data.copy()
        #self.phi_s = ev['phi_s']
        #self.phi_g = ev['value_phi_g']
        #dist = cos_dist(self.phi_s[0], self.phi_g[0])
        #self.phi_pub.publish(dist)
        data = data[:,::-1,:]
        data = cv2.resize(data, (2*480, 480))
        bits = data.tobytes()
        out_msg.data = bits
        out_msg.width = 2*480
        out_msg.height = 480
        out_msg.step = 3*2*480
        self.curr_out_msg = out_msg

        if not self.mid_query:
            if 'dyn' not in FLAGS['suffix']:
                self.mdn_pub.publish(out_msg)
                if self.main_pub == 'image':
                    self.mdn_main_pub.publish(out_msg)
                else:
                    if self.goal_img_msg is not None:
                        self.mdn_main_pub.publish(self.goal_img_msg)
            else:
                data[:] = 255
                out_msg.data = data.tobytes()
                self.mdn_pub.publish(out_msg)
                if self.main_pub == 'image':
                    self.mdn_main_pub.publish(out_msg)
                else:
                    if self.goal_img_msg is not None:
                        self.mdn_main_pub.publish(self.goal_img_msg)
                
        #rospy.signal_shutdown(0)

if __name__ == "__main__":
    network_handler = NetworkHandler(FLAGS)
    node = RVizImageDisplay(network_handler)
    #while True:
    #  x1, x2, y1, y2 = map(float, input('x1 x2 y1 y2: ').split())
    #  array = sample_area(x1, x2, y1, y2)
    #  node._set_goal_state(array)
    node.run()


