import os
import time
import numpy as np
import quaternion

normalize = lambda x: x / np.linalg.norm(x)

# MATH UTILS
def look_at(from_pos, to_pos):
    """Compute quaternion to point from `from_pos` to `to_pos`

    We define this ourselves, rather than using Mujoco's body tracker,
    because it makes it easier to randomize without relying on calling forward() 
    
    Reference: https://stackoverflow.com/questions/10635947/what-exactly-is-the-up-vector-in-opengls-lookat-function
    """
    from mujoco_py import functions
    up = np.array([0, 0, 1]) # I guess this is who Mujoco does it 
    n = normalize(from_pos - to_pos)
    u = normalize(np.cross(up, n))
    v = np.cross(n, u)
    mat = np.stack([u, v, n], axis=1).flatten()
    quat = np.zeros(4)
    functions.mju_mat2Quat(quat, mat) # this can be replaced with np.quaternion something if we need
    return quat


# OBJECT TYPE THINGS
def Range(min, max):
    """Return 1d numpy array of with min and max"""
    if min <= max:
        return np.array([min, max])
    else:
        print("WARNING: min {} was greater than max {}".format(min, max))
        return np.array([max, min])
R = Range

def Range3D(x, y, z):
    """Return numpy 1d array of with min and max"""
    return np.array([x,y,z])
R3D = Range3D

def rto3d(r):
    return Range3D(r, r, r)

def in_range(val, r):
    return r[0] <= val and val <= r[1]

def in_range3d(val3d, r3d):
    return in_range(val3d[0], r3d[0]) and in_range(val3d[1], r3d[1]) and in_range(val3d[2], r3d[2])

# UTIL FUNCTIONS FOR RANDOMIZATION
def sample(np_random, num_range, mode='standard', as_int=False):
    """Sample a float in the num_range

    mode: logspace means the range 0.1-0.3 won't be sample way less frequently then the range 1-3, because of different scales (i think)
    """
    if mode == 'standard':
        samp = np_random.uniform(num_range[0], num_range[1])
    elif mode == 'logspace':
        num_range = np.log(num_range)
        samp = np.exp(np_random.uniform(num_range[0], num_range[1]))

    if as_int:
        return int(samp)
    else:
        return samp


def sample_geom_type(np_random, types=["sphere", "capsule", "ellipsoid", "cylinder", "box"], p=[0.05, 0.05, 0.1, 0.2, 0.6]):
    """Sample a mujoco geom type (range 3-6 capsule-box)"""
    ALL_TYPES = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"]

    shape = np_random.choice(types, p=p)
    return ALL_TYPES.index(shape)

def sample_xyz(np_random, range3d, mode='standard'):
    """Sample 3 floats in the 3 num_ranges"""
    x = sample(np_random, range3d[0], mode=mode)
    y = sample(np_random, range3d[1], mode=mode)
    z = sample(np_random, range3d[2], mode=mode)
    return np.array([x, y, z])

def _in_cube(arr3, range3d):
    """accepts array of xyz, and Range3D"""
    return np.all(arr3 > range3d[:,0]) and np.all(arr3 < range3d[:,1])

def sample_xyz_restrict(np_random, range3d, restrict):
    """Like sample_xyz, but if it lands in any of the restricted ranges, then resample"""
    while True:
        x, y, z = sample_xyz(np_random, range3d)
        if not _in_cube(np.array([x,y,z]), restrict):
            break
    return (x, y, z)


def sample_joints(np_random, jnt_range, jnt_shape):
    """samples joints"""
    return (jnt_range[:,1] - jnt_range[:,0]) * np_random.sample(jnt_shape) + jnt_range[:,0]

def sample_light_dir(np_random):
    """Sample a random direction for a light. I don't quite understand light dirs so
    this might be wrong"""
    # Pretty sure light_dir is just the xyz of a quat with w = 0.
    # I random sample -1 to 1 for xyz, normalize the quat, and then set the tuple (xyz) as the dir
    LIGHT_DIR = Range3D(Range(-1,1), Range(-1,1), Range(-1,1))
    return np.quaternion(0, *sample_xyz(np_random, LIGHT_DIR)).normalized().components.tolist()[1:]

def quat_from_euler(angle3):
    roll = angle3[0] * np.pi / 180
    pitch = angle3[1] * np.pi / 180
    yaw = angle3[2] * np.pi / 180
    quat = quaternion.from_euler_angles(roll, pitch, yaw)
    return quat.normalized().components

def sample_quat(np_random, angle3):
    """Sample a quaterion from a range of euler angles in degrees"""
    roll = sample(np_random, angle3[0]) * np.pi / 180
    pitch = sample(np_random, angle3[1]) * np.pi / 180
    yaw = sample(np_random, angle3[2]) * np.pi / 180

    quat = quaternion.from_euler_angles(roll, pitch, yaw)
    return quat.normalized().components

def jitter_angle(np_random, quat, angle3):
    """Jitter quat with an angle range"""
    if len(angle3) == 2:
        angle3 = rto3d(angle3)

    sampled = sample_quat(np_random, angle3)
    return (np.quaternion(*quat) * np.quaternion(*sampled)).normalized().components

def random_quat(np_random):
    """Sample a completely random quaternion"""
    quat_random = np.quaternion(*(np_random.randn(4))).normalized()
    return quat_random.components

def jitter_quat(np_random, quat, amount):
    """Jitter a given quaternion by amount"""
    jitter = amount * np_random.randn(4)
    quat_jittered = np.quaternion(*(quat + jitter)).normalized()
    return quat_jittered.components