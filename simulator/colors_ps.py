"""
Coloring functions used to create a 3-channel matrix representing an RGB image
from a layer represented by a bitmask. The output matrix should support 8 bit
color depth (i.e. its data type is uint8).

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
import numpy as np
import random

def color_w_constant_color(fmask, color):
    """
    Color whole layer with a constant color.

    Input:
    fmask -- binary mask indicating the shape that is to be coloured
    color -- (r,g,b) tuple

    Output:
    matrix of dimensions (x,y,3) representing 3-channel image
    """

    # Initialize empty matrix
    
    # Set each channel to the value given by the input color tuple

    return img


def color_w_random_color(fmask, mean, range):
    """
    Colors layer with constant color, then draws random integer uniformly from
    [mean-range;mean+range] and adds it to the image.

    Input:
    fmask -- binary mask indicating the shape that is to be coloured
    mean  -- mean color, (r,g,b) tuple
    range -- range within which to vary mean color

    Output:
    matrix of dimensions (x,y,3) representing 3-channel image
    """

    # Generate an image coloured with the 'mean' color
    
    # Cast image to a data type supporting negative values and values greater 
    # than 255 to avoid overflows

    # Produce random integer noise uniformly drawn from [-range;range] covering
    # the whole image and add it to the image

    # Cut off values exceeding the uint8 data type and cast the image back

    return np.array(img_copy, dtype=np.uint8)


def color_w_constant_color_random_mean(fmask, mean, lb, ub):
    """
    Picks a random color from ([mean[0]-lb;mean[0]+ub],...) and colors
    layer with this color.

    Input:
    fmask -- binary mask indicating the shape that is to be coloured
    mean  -- mean color, (r,g,b) tuple
    lb    -- lower bound for the interval to draw the random mean color from
    ub    -- upper bound for the interval to draw the random mean color from

    Output:
    matrix of dimensions (x,y,3) representing 3-channel image
    """

    # Draw a random color from [r/g/b-lb;r/g/b+ub] and use it to colour
    # the image. Make sure the generated color is supported on [0;255]x[0;255]x[0;255]

    return img


# Public API
# Exporting a registry instead of the functions allows us to change the
# implementation whenever we want.
COLOR_FCT_REGISTRY = {
    'constant'            : color_w_constant_color,
    'random'              : color_w_random_color,
    'constant_random_mean': color_w_constant_color_random_mean
}

