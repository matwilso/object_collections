import os
import copy
import time
import yaml
import numpy as np
import tensorflow as tf
import quaternion

import gym
import cv2

import mujoco_py
import gym
from gym import error, spaces
from gym.utils import seeding

import matplotlib.pyplot as plt

from .utils.sim import R, R3D, rto3d, random_quat, jitter_quat, sample_quat, look_at, quat_from_euler
from .utils.table import TABLES
from .utils.pid import PID

# Based on multi-goal robotics environments from OpenAI Gym, and then hacked to hell.
# https://github.com/openai/gym/tree/master/gym/envs/robotics

class BaseEnv(gym.Env, gym.utils.EzPickle):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.model = mujoco_py.load_model_from_path(self.FLAGS['filepath'])
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=self.FLAGS['nsubsteps'])
        self.viewer = None
        self.viewer = self._get_viewer() 

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.seed()
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.cam_aspect, self.cam_fovy = 1.3333333333, 45  # ASUS

        self.name2bid = self.model.body_name2id
        self.name2gid = self.model.geom_name2id
        self.name2sid = self.model.site_name2id

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def _get_cam_frame(self):
        """Grab an RGB image from the camera"""
        height = 480
        width = np.ceil(height * self.cam_aspect)
        cam_img = self.sim.render(width, height, camera_name='camera1')[::-1, :, :] # Rendered images are upside-down.

        if self.FLAGS['display_data']:
            display_image(cam_img, FLAGS=self.FLAGS)
        return cam_img

    # Env methods
    # ----------------------------
    def seed(self, seed=None):
        self.current_seed = seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def close(self):
        if self.viewer is not None:
            #self.viewer.finish()
            self.viewer = None

    def render(self, mode='human'):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim) if self.FLAGS['mj_gpu'] else DummyRenderer()
            self._viewer_setup()
        return self.viewer

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position, for example."""
        body_id = self.sim.model.body_name2id('object_table')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 0.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def display_image(cam_img, FLAGS):
    """matplotlib show image"""
    cam_img = cv2.resize(cam_img.astype(np.float32) / 255.0, (FLAGS['image_width'], FLAGS['image_height']), interpolation=cv2.INTER_AREA)
    plt.imshow(cam_img)
    plt.show()


# dummy objects to swap in for object when we don't want to render
class DummyCam:
    pass
class DummyRenderer(object):
    def __init__(self):
        self.cam = DummyCam()
        self.cam.lookat = {}
    def render(self):
        pass
    def finish(self):
        pass
