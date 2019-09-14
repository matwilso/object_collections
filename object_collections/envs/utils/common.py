import numpy as np

# TODO: add map and unmap table here 

def discretize(act, val, FLAGS):
    """
    Convert a value into a discrete value for the discrete env

    Args:
        act (string): xy, yaw, or dist
        val (float): values closest to env vals
        FLAGS (dict): FLAGS needed for parameterizing env vals
    """
    env_val = FLAGS['ACTS'][act]
    return np.argmin(np.abs(env_val - val))

def undiscretize(act, val, FLAGS):
    # TODO: 
    pass

def map_continuous(act, val, FLAGS):
    low = FLAGS['ACTS'][act][0]
    high = FLAGS['ACTS'][act][-1]
    val = np.clip(val, low, high)  # clip here because this is used 

    mean = (low + high) / 2
    new_val = 2 * (val - mean) / (high - low)
    assert new_val >= -1 and new_val <= 1, 'new_val {}'.format(new_val)
    return new_val

def unmap_continuous(act, val, FLAGS):
    low = FLAGS['ACTS'][act][0]
    high = FLAGS['ACTS'][act][-1]
    assert val >= -1 and val <= 1, 'val {}'.format(val)

    mean = (low + high) / 2
    old_val = (val * (high - low)/2) + mean
    assert old_val >= low and old_val <= high, 'old_val {} low {} high {}'.format(old_val, low, high)
    return old_val
