import numpy as np
import random
import math

DEFAULT_COLLISION_RETRY_COUNT = -1
DEFAULT_VOXEL_OVERLAP_TOLERANCE = -1
DEFAULT_TORTUOSITY_ENABLED = True
DEFAULT_TORTUOSITY_MIN_DEPTH = 2
DEFAULT_TORTUOSITY_PROB_BASE = 40.0
DEFAULT_TORTUOSITY_PROB_SLOPE = 5.0
DEFAULT_TORTUOSITY_ANGLE_BASE = 15.0
DEFAULT_TORTUOSITY_ANGLE_SLOPE = 2.0
DEFAULT_TORTUOSITY_ANGLE_CAP = 30.0

default={"k": 3,
          "epsilon": 10 , # Proportion between length & diameter
          "randmarg": 3 , # Randomness margin between length & diameter
          "sigma": 5, # Determines type deviation for Gaussian distributions
          "stochparams": True, # Whether the generated parameters will also be stochastic
          "collision_clearance": 0.0,
          "collision_retry_count": DEFAULT_COLLISION_RETRY_COUNT,
          "voxel_overlap_tolerance": DEFAULT_VOXEL_OVERLAP_TOLERANCE,
          "tortuosity_enabled": DEFAULT_TORTUOSITY_ENABLED,
          "tortuosity_min_depth": DEFAULT_TORTUOSITY_MIN_DEPTH,
          "tortuosity_prob_base": DEFAULT_TORTUOSITY_PROB_BASE,
          "tortuosity_prob_slope": DEFAULT_TORTUOSITY_PROB_SLOPE,
          "tortuosity_angle_base": DEFAULT_TORTUOSITY_ANGLE_BASE,
          "tortuosity_angle_slope": DEFAULT_TORTUOSITY_ANGLE_SLOPE,
          "tortuosity_angle_cap": DEFAULT_TORTUOSITY_ANGLE_CAP}

active_properties = default.copy()

def setProperties(properties):
    """
    Sets global property values based on the input dictionary.

    Args:
        properties (dict): A dictionary containing the properties and their values.

    Returns:
        None

    """
    if properties == None:
        properties = default

    global k, epsilon, randmarg, sigma, stochparams
    global collision_clearance, collision_retry_count, voxel_overlap_tolerance
    global tortuosity_enabled, tortuosity_min_depth, tortuosity_prob_base
    global tortuosity_prob_slope, tortuosity_angle_base, tortuosity_angle_slope
    global tortuosity_angle_cap
    global active_properties

    active_properties = default.copy()
    active_properties.update(properties)

    k = active_properties['k']
    epsilon = active_properties['epsilon']
    randmarg = active_properties['randmarg']
    sigma = active_properties['sigma']
    stochparams = active_properties['stochparams']
    collision_clearance = active_properties['collision_clearance']
    collision_retry_count = active_properties['collision_retry_count']
    voxel_overlap_tolerance = active_properties['voxel_overlap_tolerance']
    tortuosity_enabled = active_properties['tortuosity_enabled']
    tortuosity_min_depth = active_properties['tortuosity_min_depth']
    tortuosity_prob_base = active_properties['tortuosity_prob_base']
    tortuosity_prob_slope = active_properties['tortuosity_prob_slope']
    tortuosity_angle_base = active_properties['tortuosity_angle_base']
    tortuosity_angle_slope = active_properties['tortuosity_angle_slope']
    tortuosity_angle_cap = active_properties['tortuosity_angle_cap']

def getProperties():
    """
    Returns a copy of the active generation properties.

    Returns:
        dict: The currently active property dictionary.

    """
    return active_properties.copy()

def calParam(text, params):
    '''
    Calculates the value within the parentheses before analyzing the text. For example, f('co / 4'), f requires 
    the value of 'co'. 
    
    Args:
    text (str): A string containing the mathematical expression to be evaluated
    params (dict): A dictionary containing parameter names (as keys) and their values (as values)
    
    Returns:
    str: A string representation of the evaluated expression 
    '''

    txt = text[:]
    for i in params: txt = txt.replace(i, str(params[i]))
        
    return str(params['co'] / eval(txt))

def calBifurcation(d0):
    '''
    Calculates the diameters and angles of bifurcation given an input diameter
    
    Args:
    d0 (float): input diameter
    
    Returns:
    resp (dict): a dictionary containing the calculated values for d1, d2, d0, th1, th2, and co
    '''

    resp = {}

    dOpti = d0 / 2 ** (1.0 / k)
    if stochparams: d1 = abs(np.random.normal(dOpti, dOpti / sigma))
    else: d1 = dOpti # Optimal diameter

    if d1 >= d0: d1 = dOpti # Elimate possibility of d1 being greater than d0

    d2 = (d0 ** k - d1 ** k) ** (1.0 / k) # Calculate second diameter
    # alpha = abs(np.random.uniform(1., 0.25)) * (d2 / d1) # Rate of symmetry of daughters (=1 symmetrical ?)
    alpha = d2 / d1

    '''
    Equations which mimic bifurcation angles in the human body
    Liu et al. (2010) and Zamir et al. (1988)
    '''
    xtmp = (1 + alpha * alpha * alpha) ** (4.0 / 3) + 1 - alpha ** 4
    xtmpb = 2 * ((1 + alpha * alpha * alpha ) ** (2.0 / 3))
    a1 = math.acos(xtmp / xtmpb)

    xtmp = (1 + alpha * alpha * alpha) ** (4.0 / 3) + (alpha ** 4) - 1
    xtmpb = 2 * alpha * alpha * ((1 + alpha * alpha * alpha) ** (2.0/3))
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
    Returns the length of the branch based on the diameter of the parent branch.

    Parameters:
    d0 (float): The diameter of the parent branch.

    Returns:
    float: The length of the branch.
    """
    c0 = d0 * epsilon
    # abs(np.random.normal(50,10))
    return np.random.uniform(c0 - randmarg, c0 + randmarg)
