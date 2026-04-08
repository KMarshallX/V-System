import numpy as np
import random
import math

DEFAULT_COLLISION_RETRY_COUNT = 3
DEFAULT_VOXEL_OVERLAP_TOLERANCE = 0

default={"k": 3,
          "epsilon": 10 , # Proportion between length & diameter
          "randmarg": 3 , # Randomness margin between length & diameter
          "sigma": 5, # Determines type deviation for Gaussian distributions
          "stochparams": True, # Whether the generated parameters will also be stochastic
          "collision_clearance": 0.0,
          "collision_retry_count": DEFAULT_COLLISION_RETRY_COUNT,
          "voxel_overlap_tolerance": DEFAULT_VOXEL_OVERLAP_TOLERANCE}

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
