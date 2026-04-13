import math
import numpy as np

from vsystem.config import (
    DEFAULT_COLLISION_RETRY_COUNT,
    DEFAULT_TORTUOSITY_ANGLE_BASE,
    DEFAULT_TORTUOSITY_ANGLE_CAP,
    DEFAULT_TORTUOSITY_ANGLE_SLOPE,
    DEFAULT_TORTUOSITY_ENABLED,
    DEFAULT_TORTUOSITY_MIN_DEPTH,
    DEFAULT_TORTUOSITY_PROB_BASE,
    DEFAULT_TORTUOSITY_PROB_SLOPE,
    DEFAULT_VOXEL_OVERLAP_TOLERANCE,
    VSystemConfig,
    config_to_properties,
    resolve_config,
)

DEFAULT_PROPERTIES = config_to_properties(VSystemConfig())
active_properties = DEFAULT_PROPERTIES.copy()


def _sync_module_globals(properties):
    """
    Mirrors validated properties into module globals for the generation modules.

    Args:
        properties (dict): Flattened, validated property dictionary.

    Returns:
        None
    """
    global k, epsilon, randmarg, sigma, stochparams
    global collision_clearance, collision_retry_count, voxel_overlap_tolerance
    global tortuosity_enabled, tortuosity_min_depth, tortuosity_prob_base
    global tortuosity_prob_slope, tortuosity_angle_base, tortuosity_angle_slope
    global tortuosity_angle_cap

    k = properties["k"]
    epsilon = properties["epsilon"]
    randmarg = properties["randmarg"]
    sigma = properties["sigma"]
    stochparams = properties["stochparams"]
    collision_clearance = properties["collision_clearance"]
    collision_retry_count = properties["collision_retry_count"]
    voxel_overlap_tolerance = properties["voxel_overlap_tolerance"]
    tortuosity_enabled = properties["tortuosity_enabled"]
    tortuosity_min_depth = properties["tortuosity_min_depth"]
    tortuosity_prob_base = properties["tortuosity_prob_base"]
    tortuosity_prob_slope = properties["tortuosity_prob_slope"]
    tortuosity_angle_base = properties["tortuosity_angle_base"]
    tortuosity_angle_slope = properties["tortuosity_angle_slope"]
    tortuosity_angle_cap = properties["tortuosity_angle_cap"]


def setProperties(properties):
    """
    Validates and stores generation properties for the generation modules.

    Args:
        properties (dict | VSystemConfig | None): Property dictionary or
            typed config object. `None` restores canonical defaults.

    Returns:
        None
    """
    global active_properties

    config = resolve_config(properties)
    active_properties = config_to_properties(config)
    _sync_module_globals(active_properties)


def getProperties():
    """
    Returns a copy of the active generation properties.

    Returns:
        dict: The currently active property dictionary.
    """
    return active_properties.copy()


def calParam(text, params):
    """
    Calculates the value within the parentheses before analyzing the text.

    Args:
        text (str): Mathematical expression to evaluate.
        params (dict): Parameter values used to resolve symbolic names.

    Returns:
        str: String representation of the evaluated expression.
    """
    txt = text[:]
    for key in params:
        txt = txt.replace(key, str(params[key]))

    return str(params["co"] / eval(txt))


def calBifurcation(d0):
    """
    Calculates daughter branch diameters and bifurcation angles.

    Args:
        d0 (float): Input parent diameter.

    Returns:
        dict: Bifurcation parameters for the current recursion step.
    """
    resp = {}

    dOpti = d0 / 2 ** (1.0 / k)
    if stochparams:
        d1 = abs(np.random.normal(dOpti, dOpti / sigma))
    else:
        d1 = dOpti

    if d1 >= d0:
        d1 = dOpti

    d2 = (d0 ** k - d1 ** k) ** (1.0 / k)
    alpha = d2 / d1

    xtmp = (1 + alpha * alpha * alpha) ** (4.0 / 3) + 1 - alpha ** 4
    xtmpb = 2 * ((1 + alpha * alpha * alpha) ** (2.0 / 3))
    a1 = math.acos(xtmp / xtmpb)

    xtmp = (1 + alpha * alpha * alpha) ** (4.0 / 3) + (alpha ** 4) - 1
    xtmpb = 2 * alpha * alpha * ((1 + alpha * alpha * alpha) ** (2.0 / 3))
    a2 = math.acos(xtmp / xtmpb)

    resp["d1"] = d1
    resp["d2"] = d2
    resp["d0"] = d0
    resp["th1"] = a1 * 180 / math.pi
    resp["th2"] = a2 * 180 / math.pi
    resp["co"] = getLength(d0)

    return resp


def getLength(d0):
    """
    Returns the branch length based on the current diameter.

    Args:
        d0 (float): Parent diameter.

    Returns:
        float: Branch length.
    """
    c0 = d0 * epsilon
    return np.random.uniform(c0 - randmarg, c0 + randmarg)


setProperties(None)
