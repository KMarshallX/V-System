import random
import numpy as np
from math import floor
from libGenerator import calBifurcation, calParam, getProperties
from analyseGrammar import posneg

STEM_SEGMENT_COUNT = 5
STEM_TORTUOSITY_TURNS = STEM_SEGMENT_COUNT - 1
TORTUOSITY_VARIATION_MARGIN = 0.15

def clip_to_unit_interval(value):
    """
    Clips a numeric value to the inclusive range [0, 1].

    Args:
    value (float): Value to clip.

    Returns:
    float: Clipped value.
    """
    return max(0.0, min(1.0, value))

def get_tortuosity_plan(depth, step_count=STEM_TORTUOSITY_TURNS, properties=None):
    """
    Builds the stochastic tortuosity schedule for a child stem at a given depth.

    Args:
    depth (int): Recursion depth of the stem being generated.
    step_count (int, optional): Number of steering increments to distribute. Defaults to 4.
    properties (dict, optional): Explicit property dictionary. Defaults to active properties.

    Returns:
    dict: Tortuosity plan with probability, selected angle, and per-step increments.
    """
    if properties is None:
        properties = getProperties()

    empty_plan = {
        "apply": False,
        "probability": 0.0,
        "angle": 0.0,
        "increments": [],
    }

    if (not properties.get("tortuosity_enabled", False)) or step_count <= 0:
        return empty_plan

    min_depth = properties.get("tortuosity_min_depth", 0)
    if depth < min_depth:
        return empty_plan

    depth_offset = depth - min_depth
    probability = clip_to_unit_interval(
        properties.get("tortuosity_prob_base", 0.0)
        + properties.get("tortuosity_prob_slope", 0.0) * depth_offset
    )
    if probability <= 0.0:
        return empty_plan

    angle = min(
        properties.get("tortuosity_angle_base", 0.0)
        + properties.get("tortuosity_angle_slope", 0.0) * depth_offset,
        properties.get("tortuosity_angle_cap", 0.0),
    )
    angle = max(0.0, angle)
    if angle == 0.0:
        return {
            "apply": False,
            "probability": probability,
            "angle": 0.0,
            "increments": [],
        }

    if probability < 1.0 and random.random() >= probability:
        empty_plan["probability"] = probability
        return empty_plan

    weights = np.ones(step_count)
    for idx in range(step_count):
        weights[idx] *= random.uniform(
            1.0 - TORTUOSITY_VARIATION_MARGIN,
            1.0 + TORTUOSITY_VARIATION_MARGIN,
        )

    weights = weights / np.sum(weights)
    direction = random.choice((-1.0, 1.0))
    increments = [direction * angle * weight for weight in weights]

    return {
        "apply": True,
        "probability": probability,
        "angle": angle,
        "increments": increments,
    }

def format_turn(operator, angle):
    """
    Serializes a turn payload while preserving the original grammar operator.

    Args:
    operator (str): Grammar turn operator, either '+' or '-'.
    angle (float): Signed steering angle in degrees.

    Returns:
    str: Grammar token for the turn.
    """
    return operator + '(' + str(angle) + ')'

def build_stem_description(forward_tokens, base_turns, depth):
    """
    Combines stem forward tokens and turn tokens, optionally adding tortuous steering.

    Args:
    forward_tokens (list): Ordered forward grammar tokens for the stem.
    base_turns (list): Ordered `(operator, payload)` pairs between successive forward tokens.
    depth (int): Recursion depth of the stem being generated.

    Returns:
    str: Combined grammar description for the stem.
    """
    tortuosity_plan = get_tortuosity_plan(depth, step_count=len(base_turns))
    steering_offsets = tortuosity_plan["increments"] if tortuosity_plan["apply"] else [0.0] * len(base_turns)

    description = []
    for index, forward_token in enumerate(forward_tokens):
        description.append(forward_token)
        if index < len(base_turns):
            operator, payload = base_turns[index]
            if operator == '+':
                adjusted_payload = payload + steering_offsets[index]
            else:
                adjusted_payload = payload - steering_offsets[index]
            description.append(format_turn(operator, adjusted_payload))

    return ''.join(description)

def I(n, d0, val=3):
    '''
    A recursive function to generate an "I" fractal pattern.
    
    Args:
    n (int): the level of recursion. n = 0 returns "I".
    d0 (float): the initial length of the line segment.
    val (int, optional): a constant used in the function. Default is 3.
    
    Returns:
    str: the resulting string representing the fractal pattern.
    '''
    if n > 0:
        params = calBifurcation(d0)
        p1 = calParam(str.join('co/',str(int(val))), params)
        rotate = np.random.uniform(22.5, 27.5)
        return 'f(' + p1 + ',' + str(params['d0']) + ')' + '+(' + str(rotate) + ')' +\
            '[' + R(n-1, params['d0']) + ']'
    else: return 'I'


def R(n, d0):
    '''
    A recursive function to generate an "R" fractal pattern.
    
    Parameters:
    - n (int): The level of recursion. Must be a non-negative integer.
    - d0 (float): The diameter of the parent branch. Must be a positive float.
    
    Returns:
    - A string description of a bifurcating tree.
    '''
    if n > 0:
        params = calBifurcation(d0)
        p1 = calParam(str.join('co/',str(int(3))), params)
        p2 = calParam(str.join('co/',str(int(2))), params)
        descrip = 'f(' + p1 + ')' + D(n-1, d0) +\
            D(n-1, d0) + D(n-1, d0) + '[' + B(n-1, params['d1']) +\
                ']' + 'f(' + p2 + ',' + str(params['d2']) + ')' + B(n-1, params['d2'])
        return descrip
    else: return 'R'
                    

def B(n, d0):
    """
    A recursive function to generate an "B" fractal pattern.

    Args:
    - n (int): The current bifurcation level.
    - d0 (float): The current diameter of the parent branch.

    Returns:
    - A string describing the bifurcation structure.

    Raises:
    - N/A
    """
    if n > 0:
        rotate = np.random.uniform(22.5, 27.5)
        return D(n-1, d0) + D(n-1, d0) + D(n-1, d0) +\
            '/(' +str(rotate) + ')' + A(n-1, d0)
    else: return 'B'


def F(n, d0, depth=0):
    """
    A recursive function to generate an "F" fractal pattern.
    
    Args:
    - n (int): number of iterations to perform
    - d0 (float): the initial distance
    
    Returns:
    - str: a string of characters representing the fractal generated by the given parameters.
    """
    if n > 0:
        params = calBifurcation(d0)
        randInt = random.randint(-1,1)
        theta1 = params['th1']
        theta2 = params['th2']
        tilt = np.random.uniform(22.5, 27.5)*random.randint(-1,1)
        return S(n-1, d0, depth=depth)+'['+'+('+str(theta1)+')'+'/('+str(np.random.uniform(22.5, 27.5)*random.randint(-1,1))+')'+\
            F(n-1, params['d1'], depth=depth+1)+']'+'['+'-('+str(theta2)+')'+'/('+str(np.random.uniform(22.5, 27.5)*random.randint(-1,1))+')'+\
                F(n-1, params['d2'], depth=depth+1)+']'
    else: return 'F'
    
    
def S(n, d0, val=5, margin=0.5, depth=0):
    """
    S function generates the shape of a tree branch.

    Args:
    n (int): The depth of the recursive algorithm.
    d0 (float): The trunk diameter of the tree.
    val (int, optional): Default is 5. The angle in degrees between a branch and the trunk.
    margin (float, optional): Default is 0.5. The chance of using one of two sub-functions (S1, S2) to generate the stem.

    Returns:
    str: A string representing the shape of a tree's stem.
    """
    r = random.random()
    if r>= 0.0 and r < margin: return '{' + S1(n, d0, val, depth=depth) + '}'
    if r >= margin and r < 1.0: return '{' + S2(n, d0, val, depth=depth) + '}'


def S1(n, d0, val=5, depth=0):
    """
    Recursive function that generates a string representation of a tree structure.
    Includes shrinkage and expansion of vessel diameters.
    
    Args:
    - n (int): the recursion level, i.e. the number of times the function is called recursively.
    - d0 (float): the initial diameter of the tree.
    - val (int): value to be passed to function D(). Default value is 5.
    
    Returns:
    - str: string representation of a tree structure.
    """
    if n > 0:
        # "Fanning" of trees
        # rotate = np.random.uniform(22.5, 27.5)*random.randint(-1,1)
        params = calBifurcation(d0) # calculates the branching parameters based on the starting thickness of the trunk
        randInt = random.randint(-1, 1) # randomly generates an integer of either -1, 0, or 1
        theta1 = params['th1'] * randInt # calculates the angle of the first branch based on the random integer
        theta2 = params['th2'] * abs(randInt) # calculates the angle of the second branch based on the absolute value of the random integer
        if random.random() < 0.1: # randomly generates growth and decay parameters
            growth = np.sort(np.random.uniform(1., 1.5, 3)) # generates 3 growth factors between 1 and 1.5 and sorts them in ascending order
            decay = -np.sort(-np.random.uniform(1., 1.5, 2)) # generates 2 decay factors between 1 and 1.5, negates them, and sorts them in ascending order
        else:
            growth = np.ones(3) # sets growth factors to 1 if the random number is greater than or equal to 0.1
            decay = np.ones(2) # sets decay factors to 1 if the random number is greater than or equal to 0.1
        forward_tokens = [
            D(n-1, growth[0]*params['d0'], val),
            D(n-1, growth[1]*params['d0'], val),
            D(n-1, growth[2]*params['d0'], val),
            D(n-1, decay[0]*params['d0'], val),
            D(n-1, decay[1]*params['d0'], val),
        ]
        base_turns = [('+', theta1), ('-', theta2), ('-', theta1), ('+', theta2)]
        return build_stem_description(forward_tokens, base_turns, depth)
    else:
        return 'S' # returns the string 'S' if n is 0


def S2(n, d0, val=5, depth=0):
    """
    Recursive function that generates a string representation of a tree structure.
    Includes shrinkage and expansion of vessel diameters.
    
    Args:
    - n (int): the recursion level, i.e. the number of times the function is called recursively.
    - d0 (float): the initial diameter of the tree.
    - val (int): value to be passed to function D(). Default value is 5.
    
    Returns:
    - str: string representation of a tree structure.
    """
    if n>0:
        # "Fanning" of trees
        #rotate = np.random.uniform(22.5, 27.5)*random.randint(-1,1)
        params = calBifurcation(d0)
        randInt = random.randint(-1,1)
        theta1 = params['th1']*randInt #+ np.random.uniform(-2.5, 2.5)
        theta2 = params['th2']*abs(randInt) #+ np.random.uniform(-2.5, 2.5)
        if random.random() < 0.1:
            growth = np.sort(np.random.uniform(1., 1.5, 3))
            decay = -np.sort(-np.random.uniform(1., 1.5, 2))
        else:
            growth = np.ones(3)
            decay = np.ones(2)
        forward_tokens = [
            D(n-1, growth[0]*params['d0'], val),
            D(n-1, growth[1]*params['d0'], val),
            D(n-1, growth[2]*params['d0'], val),
            D(n-1, decay[0]*params['d0'], val),
            D(n-1, decay[1]*params['d0'], val),
        ]
        base_turns = [('-', theta1), ('+', theta2), ('+', theta1), ('-', theta2)]
        return build_stem_description(forward_tokens, base_turns, depth)
    else: return 'S'


def D(n, d0, val=5):
    """
    Generate a string of L-system symbols for generating a fractal tree using a
    deterministic context-free Lindenmayer system (D0L-system) with production rules.

    Parameters:
    n (int): Number of iterations.
    d0 (float): Starting diameter of the tree.
    val (int): Value to be passed to `calParam` function (default is 5).

    Returns:
    str: The generated L-system symbols string for the fractal tree.

    """
    if n>0:
        params = calBifurcation(d0)
        p1 = calParam(str.join('co/',str(int(val))), params)
        return 'f(' + p1 + ',' + str(params['d0']) + ')'
    else: return 'D'
    

def G(n, d0, val=5):
    """
    Generates a string representing the result of the function f with parameters p1 and d0.

    Args:
    - n (int): number of iterations of the function f.
    - d0 (float): parameter d0 of the bifurcation function.
    - val (int): value to use for calculating the parameter p1. Default is 5.
    - shift (float): value to add to the result of f. Default is 18.0.

    Returns:
    - str: a string representing the result of the function f with parameters p1 and d0.
    """
    if n>0:
        params = calBifurcation(d0)
        p1 = calParam(str.join('co/',str(int(val))), params)
        return 'f(' + p1 + ',' + str(params['d0']) + ')' 
    else: return 'G'
    
    
def A(n, d0):
    """
    Generates string description of fractal tree using recursive approach
    Args:
        n (int): Depth of the recursion
        d0 (float): Initial branch length
    Returns:
        str: String description of the fractal tree
    """
    if n > 0:
        params = calBifurcation(d0)
        return S(n-1, d0) + '[+(' + str(params['th1']) + ')' + A(n-1, params['d1']) + ']' +\
            '[+(' + str(params['th2']) + ')' + A(n-1, params['d2'])
    else: return 'A'





# 2D grammars - don't currently work with framework
def simplest_gramma(n=None, theta1=20., theta2=20., params=None):
    return 'f' + '[' + '+' + F(n-1, params['d0']) + ']' + '-' + F(n-1, params['d0'])

def simple_grammar(n=None, theta1=20., theta2=20., params=None):
    return 'f'+'['+'+('+str(theta1)+')'+F(n-1,params['d1'])+']'+'['+'-('+str(theta2)+')'+F(n-1,params['d2'])+']'
