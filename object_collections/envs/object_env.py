import re
import itertools
import os
import copy
import time
import yaml
import numpy as np
import tensorflow as tf
import quaternion
from collections import deque
import pickle
import cv2

import gym

import mujoco_py
import gym
from gym import error, spaces
from gym.utils import seeding

import matplotlib.pyplot as plt

from .utils import sim as sim_utils
from .utils.sim import R, R3D, rto3d, random_quat, jitter_quat, sample_quat, look_at, quat_from_euler
from .utils.table import set_table, TABLES
from .utils.pid import PID
from .base import BaseEnv, goal_distance

from mujoco_py.modder import BaseModder, CameraModder, LightModder, MaterialModder
from .utils.modder import TextureModder
from .utils.common import discretize, undiscretize, map_continuous, unmap_continuous


# check if two box regions overlap
def overlap1D(box1, box2):
    return box1[1] >= box2[0] and box2[1] >= box1[0]

# check if two 3D volumes overlap
def overlap3D(box1, box2):
    return overlap1D(box1[0], box2[0]) and overlap1D(box1[1], box2[1]) and overlap1D(box1[2], box2[2])

# take a subset of a dictionary based on keys
def subdict(dict, keys):
    return {key: dict[key] for key in keys}


class ObjectCollectionEnv(BaseEnv):
    """Custom muluti-object environment"""

    def __init__(self, FLAGS):
        super().__init__(FLAGS)

        if self.FLAGS['baxter']:
            self.robot_offset = lambda: self.model.body_pos[self.name2bid('base_ground')]
        else:
            self.robot_offset = lambda: self.model.body_pos[self.name2bid('robot_table_floor')]

        self.table_center = lambda: TABLES[self.FLAGS['default_table']]['pos'] + self.robot_offset() + [0, 0, TABLES[self.FLAGS['default_table']]['wood'][2]]

        self.LIGHT = {'ambient': self.model.light_ambient, 'active': self.model.light_active, 'castshadow': self.model.light_castshadow, 'diffuse': self.model.light_diffuse, 'dir': self.model.light_dir, 'pos': self.model.light_pos, 'specular': self.model.light_specular}
        self.START_BODY_POS = self.model.body_pos.copy()
        self.START_LIGHT = {}
        for key in self.LIGHT:
            self.START_LIGHT[key] = self.LIGHT[key].copy()

        self.tex_modder = TextureModder(self.sim)
        self.cam_modder = CameraModder(self.sim)
        self.light_modder = LightModder(self.sim)
        if self.FLAGS['domrand']:
            self.tex_modder.whiten_materials()

        self.goal = {}

        # mean camera position defined relative to table center
        if FLAGS['baxter']:
            self.CAMERA_POS_MEAN = np.array([-0.64, 0.025, 1.0]) 
        else:
            self.CAMERA_POS_MEAN = np.array([-0.65, 0.0, 0.68])

        # establish values for discretizing
        self.X = self.FLAGS['ACTS']['x']
        self.Y = self.FLAGS['ACTS']['y']
        self.YAW = self.FLAGS['ACTS']['yaw']
        self.DIST = self.FLAGS['ACTS']['dist']

        self.action_space = self.FLAGS['action_space']
        self.observation_space = self.FLAGS['observation_space']

        if self.FLAGS['goal_conditioned']:
            self.goal['array'] = np.zeros(self.FLAGS['state_shape'])
            if self.FLAGS['use_image']:
                self.goal['image'] = np.zeros(self.FLAGS['image_shape'])

        self.paddle_bid = self.name2bid('paddle')
        self.paddle_gid = self.name2gid('paddle')
        self.N = int(0.5 / self.dt)  # N should correspond to about 0.5 seconds of timestep

        self.consecutive_fences = 0
        self.escape_count = 0
        self._set_invisiwall()
        self._reset_sim()

        #self.viewer.add_marker(pos=cam_pos, label="CAM: {}".format(cam_pos))

    def _get_object_poses(self):
        objs = []
        for i in range(self.FLAGS['num_objects']):
            obj = 'object{}:joint'.format(i)
            pose = self.sim.data.get_joint_qpos(obj).copy()
            pose[:3] -= self.table_center()
            if not self.FLAGS['use_quat']:
                pose = pose[:3]
            objs.append(pose)
        objs = np.stack(objs)
        return objs

    def _get_obs(self):
        self.sim.step()

        array = self._get_object_poses()

        obs = {'array': array, 'single': np.zeros(4)}
        if self.FLAGS['use_image']:
            image = self._get_cam_frame().astype(np.float32)
            image = cv2.resize(image, self.FLAGS['image_shape'][:2], interpolation=cv2.INTER_AREA)
            image /= 255.0

            obs.update({'image': image})

        if self.FLAGS['goal_conditioned']:
            obs.update({'goal_array': self.goal['array']})
            if self.FLAGS['use_image']:
                obs.update({'goal_image': self.goal['image']})
        return obs


    def _geofence_escape(self):
        """check if any of the cubes have escaped the table.  this should not be possible
        but still happens due to some glitch with mujoco

        return True if any cube escapes
        
        """
        cubes = self._get_object_poses()

        cube_size = self.sim.model.geom_size[self.name2gid('object0')]

        table_range = 0.5*TABLES[self.FLAGS['default_table']]['wood']
        table_range[2] = 0.1 # be forgiving on the height, since we already have object_fell to check for drops
        table_box = (1 + 0.01) * np.stack([-table_range, table_range], axis=-1)

        for c in cubes:
            cube_box = c[:3,None] + np.stack([-cube_size, cube_size], axis=-1)
            if not overlap3D(cube_box, table_box):
                return True

        return False

    def _object_fell(self):
        """check if any of the cubes have fallen off the table"""
        cubes = self._get_object_poses()
        for c in cubes:
            if c[2] < -0.01:
                return True
        return False

    def step(self, action):
        # convert from tuple to dictionary. this is pretty risky and bad practice, but should throw errors for shape mismatch in most cases *crosses fingers*
        if isinstance(action, tuple) or isinstance(action, np.ndarray):
            names = ['x', 'y', 'yaw']
            names += ['dist'] if self.FLAGS['use_dist'] else []
            sorted_names = list(sorted(names))
            action = {sorted_names[i]: action[i] for i in range(len(sorted_names))}
        action = copy.deepcopy(action)
        safe_action = self._set_action(action)

        reward = 0.0

        if self._geofence_escape() or self._object_fell():
            reward += -10.0
            done = True
            print('AHHH')
        else:
            done = False

        obs = self._get_obs()

        if self.FLAGS['penalize_invisiwall'] and safe_action != '':
            if 'wall' in safe_action:
                reward += -1.0

        info = {'seed': self.current_seed}
        out_obs = self.get_out_obs()

        return out_obs, reward, done, info

    def _set_action(self, action):
        """Apply the action to the env
        
        Return True is safe action, else false
        """
        self._randomize()
        contact_invisiwall = False
        if self.FLAGS['discrete']:
            x = self.X[action['x']]
            y = self.Y[action['y']]
            yaw = self.YAW[action['yaw']]
            if self.FLAGS['use_dist']:
                dist = self.DIST[1+action['dist']]
            else:
                dist = self.FLAGS['max_dx']
        else:
            spaces = self.action_space.spaces
            def clip_act(key): 
                """clip and then map back to semantic action scale"""
                clipped = np.clip(action[key], -1.0, 1.0)
                return unmap_continuous(key, clipped, self.FLAGS)
            x = clip_act('x')
            y = clip_act('y')
            yaw = clip_act('yaw')
            if self.FLAGS['use_dist']:
                dist = clip_act('dist')
            else:
                dist = self.FLAGS['max_dx']
        
        # Parse action
        s_xyz = np.array([x, y, 0.5])
        s_xyz += self.table_center()
        self.model.geom_pos[self.paddle_gid] = s_xyz
        self.sim.data.geom_xpos[self.paddle_gid] = s_xyz

        quat = quaternion.from_euler_angles(0, 0, yaw).normalized().components
        self.sim.model.geom_quat[self.paddle_gid] = quat

        xp_aid = self.model.actuator_name2id('xp')
        xv_aid = self.model.actuator_name2id('xv')
        yp_aid = self.model.actuator_name2id('yp')
        yv_aid = self.model.actuator_name2id('yv')
        zp_aid = self.model.actuator_name2id('zp')
        zv_aid = self.model.actuator_name2id('zv')

        self.sim.data.set_joint_qpos('paddle:slidex', np.array([0]))
        self.sim.data.set_joint_qpos('paddle:slidey', np.array([0]))
        self.sim.data.set_joint_qpos('paddle:slidez', np.array([0]))
        self.sim.data.set_joint_qpos('paddle:hingez', np.array([0]))

        self.sim.data.ctrl[:] = 0.0

        # Set goal positions of paddle (lower to targetz, then push to targetx and targety based on action)
        if self.FLAGS['baxter']:
            targetz = -0.925 + 0.83
        else:
            targetz = 0.69
        targetx, targety = s_xyz[0]+np.cos(yaw)*dist,  s_xyz[1]+np.sin(yaw)*dist
        # reverse push at the end so that we are not in contact with block when we lift.
        btargetx, btargety = s_xyz[0]+np.cos(yaw)*(0.5*dist),  s_xyz[1]+np.sin(yaw)*(0.5*dist)

        # Draw indicator of paddle target pos
        target = np.array([targetx, targety, targetz])
        site_id = self.name2sid('target0')
        self.sim.model.site_rgba[site_id] = np.zeros(4)
        if self.FLAGS['render']:
            self.sim.model.site_pos[site_id] = target
            self.sim.model.site_quat[site_id] = quat

        # Vertical PID controller to lower and raise 
        P = 400.0
        D = 40.0
        v_pid = PID(P, D, curr_fn=lambda: self.sim.data.geom_xpos[self.paddle_gid][2], goal=targetz)

        # Horizontal (xy) PID controllers to push
        HP = 600.0
        HD = 60.0
        hx_pid = PID(HP, HD, curr_fn=lambda: self.sim.data.geom_xpos[self.paddle_gid][0], goal=targetx)
        hy_pid = PID(HP, HD, curr_fn=lambda: self.sim.data.geom_xpos[self.paddle_gid][1], goal=targety)

        # Lower the paddle
        for i in range(self.N):
            self.sim.data.ctrl[zp_aid], self.sim.data.ctrl[zv_aid]  = v_pid.step()
            self.sim.step()
            contact_invisiwall = contact_invisiwall or self._collide_wall()
            if self.FLAGS['render']:
                self.viewer.render()

        # Push forward to the target position
        for i in range(int(1.5*self.N)):
            #self.sim.data.ctrl[zp_aid], self.sim.data.ctrl[zv_aid]  = v_pid.step()
            self.sim.data.ctrl[xp_aid], self.sim.data.ctrl[xv_aid] = hx_pid.step()
            self.sim.data.ctrl[yp_aid], self.sim.data.ctrl[yv_aid] = hy_pid.step()
            contact_invisiwall = contact_invisiwall or self._collide_wall()
            self.sim.step()
            if self.FLAGS['render']:
                self.viewer.render()

        # Reverse a bit
        hx_pid.reset_goal(btargetx)
        hy_pid.reset_goal(btargety)
        for i in range(int(0.1*self.N)):
            #self.sim.data.ctrl[zp_aid], self.sim.data.ctrl[zv_aid]  = v_pid.step()
            self.sim.data.ctrl[xp_aid], self.sim.data.ctrl[xv_aid] = hx_pid.step()
            self.sim.data.ctrl[yp_aid], self.sim.data.ctrl[yv_aid] = hy_pid.step()
            contact_invisiwall = contact_invisiwall or self._collide_wall()
            self.sim.step()
            if self.FLAGS['render']:
                self.viewer.render()

        # Raise the paddle back up after pushing
        v_pid.reset_goal(1.5)
        for i in range(self.N*3):
            self.sim.data.ctrl[zp_aid], self.sim.data.ctrl[zv_aid]  = v_pid.step()
            self.sim.data.ctrl[xp_aid], self.sim.data.ctrl[xv_aid]  = hx_pid.step()
            self.sim.data.ctrl[yp_aid], self.sim.data.ctrl[yv_aid]  = hy_pid.step()
            contact_invisiwall = contact_invisiwall or self._collide_wall()
            self.sim.step()
            if self.FLAGS['render']:
                self.viewer.render()


        if self.FLAGS['penalize_invisiwall'] and contact_invisiwall:
            safe = 'wall'
        else:
            safe = ''
        return safe

    def reset(self):
        self.goal = self._sample_goal()

        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim(reset_mode=self.np_random.choice(self.FLAGS['reset_mode']))

        curr = self._get_obs()['array']
        if self.FLAGS['render'] and self.FLAGS['goal_conditioned']:
            goal_array = self.goal['array'] * 0.5*TABLES[self.FLAGS['default_table']]['wood'][:2]
            site_dests = np.c_[goal_array, curr[:,2]] + self.table_center() 
            for i in range(self.FLAGS['num_objects']):
                site = self.name2sid('object{}'.format(i)) 
                if self.FLAGS['render']:
                    self.model.site_pos[site] = site_dests[i] - [0.0, 0.0, 0.1]
                    self.model.site_rgba[site] = np.ones(4)
        return self.get_out_obs() 

    def _collide_wall(self):
        wall_gids = [self.name2gid('invisiwall_'+str(i)) for i in range(4)]
        obj_gids = [self.name2gid('object'+str(i)) for i in range(self.FLAGS['num_objects'])]

        bad_contacts = [1 for cc in self.sim.data.contact if (cc.geom1 in wall_gids and cc.geom2 in obj_gids) or (cc.geom1 in obj_gids and cc.geom2 in wall_gids)]
        return len(bad_contacts) > 0

    def _set_invisiwall(self):
        # NOTE: Generally changing the shape of objects does not do well for contacts
        X = self.FLAGS['ACTS']['x']
        Y = self.FLAGS['ACTS']['y']
        DEPTH, Z = self.model.geom_size[self.name2gid('invisiwall_0')][1:]
        Z -= (Z/2)

        idx = 0
        for dir in [0, -1]:
            for xy in ['x', 'y']:
                gid = self.name2gid('invisiwall_'+str(idx))
                if self.FLAGS['view_invisiwall']:
                    pass
                else:
                    self.model.geom_rgba[gid] = np.zeros((4,))
                if xy == 'x':
                    yval = 1.05*np.sign(Y[dir]) * (np.abs(Y[dir]) + DEPTH)
                    self.model.geom_pos[gid] = self.table_center() + np.array([0, yval, Z])
                    #self.model.geom_size[gid] = np.array([X, DEPTH, Z])
                else:
                    xval = 1.05*np.sign(X[dir]) * (np.abs(X[dir]) + DEPTH)
                    self.model.geom_pos[gid] = self.table_center() + np.array([xval, 0, Z])
                    #self.model.geom_size[gid] = np.array([DEPTH, Y, Z])

                idx += 1
            
        self.sim.forward()
        self.sim.step()

    def _sample_xyzs(self, reset_mode=None):
        reset_mode = reset_mode or 'cluster'#self.FLAGS['reset_mode']
        object_xyzs = []  # for checking collisions against
        sample_range = R3D(self.FLAGS['obj_rangex'], self.FLAGS['obj_rangey'], R(0,0))

        if reset_mode == 'uniform':
            for i in range(self.FLAGS['num_objects']):
                while True:
                    object_xyz = sim_utils.sample_xyz(self.np_random, sample_range)
                    collision = False
                    # TODO: check if this is necessary. could get speedup
                    for other in object_xyzs:
                        collision = collision or np.linalg.norm(object_xyz[:2] - other[:2]) < 0.0127

                    if not collision and sim_utils.in_range(object_xyz[0], self.FLAGS['obj_rangex']) and sim_utils.in_range(object_xyz[1], self.FLAGS['obj_rangey']):
                        object_xyzs.append(object_xyz)
                        break
        elif reset_mode == 'cluster':
            diameter = self.model.geom_size[self.name2gid('object0')][0]
            dirs = list(itertools.product([-diameter*2,0.0,diameter*2], [-diameter*2,0.0,diameter*2]))

            first_object_xyz = sim_utils.sample_xyz(self.np_random, sample_range)
            object_xyzs.append(first_object_xyz)

            for i in range(1, self.FLAGS['num_objects']):
                while True:
                    reference = object_xyzs[self.np_random.randint(len(object_xyzs))]
                    dir = np.array(dirs[self.np_random.randint(len(dirs))])
                    dir *= self.np_random.uniform(1.0,2.0)
                    xyz = reference.copy()
                    xyz[:2] = reference[:2] + dir

                    collision = False
                    for other in object_xyzs:
                        collision = collision or np.linalg.norm(xyz[:2] - other[:2]) < 0.0127

                    if not collision and sim_utils.in_range(xyz[0], self.FLAGS['obj_rangex']) and sim_utils.in_range(xyz[1], self.FLAGS['obj_rangey']):
                        object_xyzs.append(xyz)
                        break

        return object_xyzs

    def _sample_object_poses(self, reset_mode=None):
        #rstate = np.random.RandomState(seed=0)  # use this to always reset to deterministic state
        xyz = self._sample_xyzs(reset_mode=reset_mode)
        for i in range(self.FLAGS['num_objects']):
            gid = self.name2gid('object{}'.format(i))
            #self.model.geom_size[gid] = self.FLAGS['cube_size']
            obj = 'object{}:joint'.format(i)
            object_qpos = self.sim.data.get_joint_qpos(obj)
            assert object_qpos.shape == (7,)
            object_xpos = self.table_center() + [0.0, 0.0, 0.2] +  xyz[i]
            object_quat = sim_utils.sample_quat(self.np_random, R3D(R(0,180), R(0,0), R(0,0)))
            object_qpos = np.concatenate([object_xpos, object_quat])
            self.sim.data.set_joint_qpos(obj, object_qpos)

    def _reset_sim(self, reset_mode=None):
        self.object_has_fallen = False
        self.object_fallen_stuff = None
        self.sim.set_state(self.initial_state)
        if self.FLAGS['baxter']:
            set_table(self.model, 'object_table', 'folding', self.FLAGS)
        else:
            set_table(self.model, 'robot_table', 'robot', self.FLAGS)
            set_table(self.model, 'object_table', 'small', self.FLAGS)

        self.model.site_size[self.name2sid('target0')] = self.model.geom_size[self.paddle_gid]
        self.cam_pos = self.model.cam_pos[0] = self.table_center() + np.array([-0.75, 0.0, 0.75])
        target_pos = self.sim.data.body_xpos[self.name2bid('object_table')] 
        self.model.cam_quat[0] = look_at(self.cam_pos, target_pos)

        # reset objects
        self._sample_object_poses(reset_mode=reset_mode)

        for i in range(5*self.N):
            self.sim.step()
            if self.FLAGS['render']:
                self.viewer.render()

        return not self._object_fell() and not self._geofence_escape()

    def _sample_goal(self):
        """NOTE: must be called before reset, as this can mess with state"""
        goal = {}
        if self.FLAGS['goal_conditioned']:
            # Same mechanism for resetting sim, in order to get to a random state
            did_reset_sim = False
            while not did_reset_sim:
                did_reset_sim = self._reset_sim(reset_mode='cluster')

            obs = self._get_obs()
            obs['array'][:,:3] /= (0.5*TABLES[self.FLAGS['default_table']]['wood'])
            if not self.FLAGS['use_quat']:
                obs['array'] = obs['array'][...,:2]
            goal['array'] = copy.deepcopy(obs['array'])
            if self.FLAGS['use_image']:
                goal['image'] = copy.deepcopy(obs['image'])
            return goal
        else:
            sample_range = R3D(self.FLAGS['obj_rangex'], self.FLAGS['obj_rangey'], R(0,0))
            goal['array'] = sim_utils.sample_xyz(self.np_random, sample_range)
            return goal

    def get_out_obs(self):
        """This matches the observation space.  The other one is just for internal use"""
        obs = self._get_obs()

        obs['array'][:,:3] /= (0.5*TABLES[self.FLAGS['default_table']]['wood'])
        if not self.FLAGS['use_quat']:
            obs['array'] = obs['array'][...,:2]
        
        obs_names = ['array', 'single']
        if self.FLAGS['goal_conditioned']:
            assert not np.all(np.equal(obs['goal_array'], 0.0)), "goal was not set before out_obs"
            obs_names.append('goal_array')
            if self.FLAGS['use_image']:
                obs_names.append('goal_image')
        if self.FLAGS['use_image']:
            obs_names.append('image')
            if self.FLAGS['use_canonical']:
                obs_names.append('canonical')
                self._canonicalize()
                self.sim.step()
                canon_obs = self._get_obs()
                obs['canonical'] = canon_obs['image']

        for key in obs: obs[key] = obs[key].astype(np.float32)
        out_obs = copy.deepcopy(tuple([obs[key] for key in sorted(obs_names)]))
        #print(obs['array'])
        #print()
        if np.max(np.abs(obs['array'])) > 1.01:
            print('AHHH out obs', obs['array'])
        return out_obs

    def _canonicalize(self):
        # Canonicalize textures
        for name in self.model.geom_names + ('skybox',):
            if re.match('object\d', name) is not None:
                self.tex_modder.set_rgb(name, np.array([255, 0, 255], dtype=np.uint8))
            else:
                self.tex_modder.set_rgb(name, np.array([255]*3, dtype=np.uint8))
        # light
        for key in self.LIGHT:
            self.LIGHT[key][:] = self.START_LIGHT[key].copy()

        # camera
        if self.FLAGS['mild_canonical']:
            self.cam_pos = self.CAMERA_POS_MEAN + self.table_center()
        else:
            self.cam_pos = self.CAMERA_POS_MEAN + self.table_center() + [0.639, -0.025, 0.125]
        target_id = self.model.body_name2id('object_table')
        target_pos = self.sim.data.body_xpos[target_id] #+ sim_utils.sample_xyz(self.np_random, R3D(R(-0.05, 0.05), R(-0.05, 0.05), R(0.1,0.1)))
        quat = look_at(self.cam_pos, target_pos)
        self.cam_modder.set_quat('camera1', quat)
        self.cam_modder.set_pos('camera1', self.cam_pos)
        self.cam_modder.set_fovy('camera1', 45)

        # robot
        if self.FLAGS['baxter']:
            for name in self.model.joint_names:
                if 'paddle' not in name and 'object' not in name:
                    id = self.sim.model.joint_name2id(name)
                    self.sim.data.set_joint_qpos(name, 0.0)
            #self.model.body_pos[self.name2bid('base')] = self.START_BODY_POS[self.name2bid('base')] 
        else:
            for joint in ['lbr4_j{}'.format(i) for i in range(7)]:
                id = self.sim.model.joint_name2id(joint)
                self.sim.data.set_joint_qpos(joint, 0.0)

        PREFIX = 'distract'
        geom_names = [name for name in self.model.geom_names if name.startswith(PREFIX)]
        for name in geom_names: 
            gid = self.model.geom_name2id(name)
            self.model.geom_pos[gid] = np.array([0,0,-2])
            self.model.geom_rgba[gid][-1] = 0

    def _randomize(self):
        if self.FLAGS['domrand']:
            self._rand_textures()
            self._rand_camera()
            self._rand_lights()
            #self._rand_robot()
            self._rand_distract()
        else:
            self._canonicalize()

    def _rand_textures(self):
        """Randomize all the textures in the scene, including the skybox"""
        bright = self.np_random.binomial(1, 0.8)
        for name in self.sim.model.geom_names + ('skybox',):
            self.tex_modder.rand_all(name)
            if bright: 
                if name == 'object_table':
                    self.tex_modder.brighten(name, self.np_random.randint(150,255))
                else:
                    self.tex_modder.brighten(name, self.np_random.randint(0,150))

    def _rand_camera(self):
        """Randomize pos, orientation, and fov of camera
        FOVY:
        Kinect2 is 53.8
        ASUS is 45 
        https://www.asus.com/us/3D-Sensor/Xtion_PRO_LIVE/specifications/
        http://smeenk.com/kinect-field-of-view-comparison/
        """
        dx = 0.05
        self.cam_pos = self.CAMERA_POS_MEAN + self.table_center()
        C_R3D = R3D(R(-dx,dx), R(-dx, dx), R(-2*dx, 2*dx))
        self.cam_pos += sim_utils.sample_xyz(self.np_random, C_R3D)
        self._rand_camera_angle()
        self.cam_modder.set_pos('camera1', self.cam_pos)
        self.cam_modder.set_fovy('camera1', sim_utils.sample(self.np_random, R(44, 46)))

    def _rand_camera_angle(self):
        # Look approximately at the robot
        target_id = self.model.body_name2id('object_table')
        target_pos = self.sim.data.body_xpos[target_id] + sim_utils.sample_xyz(self.np_random, R3D(R(-0.02, 0.02), R(-0.02, 0.02), R(0.0,0.2)))
        quat = look_at(self.cam_pos, target_pos)
        self.cam_modder.set_quat('camera1', quat)
    
    def _rand_lights(self):
        """Randomize pos, direction, and lights"""
        # light stuff
        X = R(-1.0, 1.0) 
        Y = R(-0.6, 0.6)
        Z = R(0.1, 1.5)
        LIGHT_R3D = self.table_center()[:,None] + R3D(X, Y, Z)
        LIGHT_UNIF = R3D(R(0,1), R(0,1), R(0,1))

        for i, name in enumerate(self.model.light_names):
            lid = self.model.light_name2id(name)
            # random sample 80% of any given light being on 
            if lid != 0:
                self.light_modder.set_active(name, sim_utils.sample(self.np_random, [0,1]) < 0.8)
                self.light_modder.set_dir(name, sim_utils.sample_light_dir(self.np_random))

            self.light_modder.set_pos(name, sim_utils.sample_xyz(self.np_random, LIGHT_R3D))


            spec =    np.array([sim_utils.sample(self.np_random, R(0.5,1))]*3)
            diffuse = np.array([sim_utils.sample(self.np_random, R(0.5,1))]*3)
            ambient = np.array([sim_utils.sample(self.np_random, R(0.5,1))]*3)

            self.light_modder.set_specular(name, spec)
            self.light_modder.set_diffuse(name,  diffuse)
            self.light_modder.set_ambient(name,  ambient)
            self.model.light_castshadow[lid] = sim_utils.sample(self.np_random, [0,1]) < 0.5

    def _rand_robot(self):
        """Randomize joint angles"""
        if self.FLAGS['baxter']:
            for name in self.model.joint_names:
                if 'paddle' not in name and 'object' not in name:
                    id = self.sim.model.joint_name2id(name)
                    self.sim.data.set_joint_qpos(name, sim_utils.sample(self.np_random, self.model.jnt_range[id]))

            self.model.body_pos[self.name2bid('base')] = self.START_BODY_POS[self.name2bid('base')] + sim_utils.sample_xyz(self.np_random, R3D(R(0,0), R(0,0), R(-0.05, 0.05)))

        else:
            for name in ['lbr4_j{}'.format(i) for i in range(7)]:
                id = self.sim.model.joint_name2id(name)
                self.sim.data.set_joint_qpos(name, sim_utils.sample(self.np_random, self.model.jnt_range[id]))

    def _rand_distract(self):
        """Randomize the position and size of the distractor objects"""
        PREFIX = 'distract'
        geom_names = [name for name in self.model.geom_names if name.startswith(PREFIX)]

        # Size range
        SX = R(0.01, 0.3)
        SY = R(0.01, 0.3)
        SZ = R(0.01, 0.3)
        S3D = R3D(SX, SY, SZ)

        # Back range
        B_PX = R(0.5, 1.0)
        B_PY = R(-2, 2)
        B_PZ = R(0.1, 0.5)
        B_P3D = R3D(B_PX, B_PY, B_PZ)

        # Front range
        F_PX = R(-0.5, 0.5)
        F_PY = R(-2, 2)
        F_PZ = R(-0.1, 0.3)
        F_P3D = R3D(F_PX, F_PY, F_PZ)

        for name in geom_names: 
            gid = self.model.geom_name2id(name)
            range = B_P3D if np.random.binomial(1, 0.5) else F_P3D

            mid = self.table_center().copy()
            mid[2] = -0.925

            self.model.geom_pos[gid] = mid + sim_utils.sample_xyz(self.np_random, range) 
            self.model.geom_quat[gid] = sim_utils.random_quat(self.np_random) 
            self.model.geom_size[gid] = sim_utils.sample_xyz(self.np_random, S3D)
            self.model.geom_type[gid] = sim_utils.sample_geom_type(self.np_random)
            self.model.geom_rgba[gid][-1] = np.random.binomial(1, 0.5)