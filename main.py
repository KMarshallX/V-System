import random
import nibabel as nib
import numpy as np
from skimage import io
from preprocessing import resize_stacks, resize_volume
from libGenerator import (
    setProperties,
    DEFAULT_COLLISION_RETRY_COUNT,
    DEFAULT_VOXEL_OVERLAP_TOLERANCE,
    DEFAULT_TORTUOSITY_ENABLED,
    DEFAULT_TORTUOSITY_MIN_DEPTH,
    DEFAULT_TORTUOSITY_PROB_BASE,
    DEFAULT_TORTUOSITY_PROB_SLOPE,
    DEFAULT_TORTUOSITY_ANGLE_BASE,
    DEFAULT_TORTUOSITY_ANGLE_SLOPE,
    DEFAULT_TORTUOSITY_ANGLE_CAP,
)
from vSystem import F, I, R, B
from analyseGrammar import branching_turtle_to_coords
from visuals import plot_coords, print_coords
from computeVoxel import process_network
from utils import bezier_interpolation

# Default Parameters
n = 6  # No. of networks to be created
d0_mean = 20.0  # Diameter of base of tree
d0_std = 10.0  # Standard deviation of base diameter
tissue_volume = (250, 250, 150)  # Specify number of pixels in the tissue volume
outpath = "./output_syn/"  # Output path
iterations = np.linspace(2, 12, n).astype(int)  # Number of iterations for each network 

for file in range(n):
    # Lindenmayer System Parameters
    properties = {
        "k": 3, 
        "epsilon": random.uniform(4, 10),  # Proportion between length & diameter
        "randmarg": random.uniform(2, 4),  # Randomness margin between length & diameter
        "sigma": 5,  # Determines type deviation for Gaussian distributions
        "stochparams": True,  # Whether the generated parameters will also be stochastic
        "collision_clearance": 2.0,  # Extra radius margin for collision checks
        "collision_retry_count": DEFAULT_COLLISION_RETRY_COUNT,  # < 0 keeps collision avoidance disabled
        "voxel_overlap_tolerance": DEFAULT_VOXEL_OVERLAP_TOLERANCE,  # < 0 keeps voxel overlap rejection disabled
        "tortuosity_enabled": DEFAULT_TORTUOSITY_ENABLED,
        "tortuosity_min_depth": DEFAULT_TORTUOSITY_MIN_DEPTH,
        "tortuosity_prob_base": DEFAULT_TORTUOSITY_PROB_BASE,
        "tortuosity_prob_slope": DEFAULT_TORTUOSITY_PROB_SLOPE,
        "tortuosity_angle_base": DEFAULT_TORTUOSITY_ANGLE_BASE,
        "tortuosity_angle_slope": DEFAULT_TORTUOSITY_ANGLE_SLOPE,
        "tortuosity_angle_cap": DEFAULT_TORTUOSITY_ANGLE_CAP,
    }

    d0 = np.random.normal(d0_mean, d0_std)  # Randomly assign base diameter (no dimension)
    # niter = random.randint(6, 14)  # Randomly assign number of V-System recursive loops
    niter = iterations[file]
    setProperties(properties)  # Setting V-Sytem properties
    print(f"Creating image ... {file} with {niter} iterations")

    # Run V-System grammar for n iterations
    turtle_program = F(niter, d0)

    # Convert grammar into coordinates
    coords = branching_turtle_to_coords(turtle_program, d0)

    # Analyse / sort coordinate data
    update = bezier_interpolation(coords)

    # If you fancy, plot a 2D image of the network!
    # plot_coords(newdata, array=True, bare_plot=False) # bare_plot removes the axis labels

    # Run 3D voxel traversal to generate binary mask of V-System network
    image = process_network(update, tVol=tissue_volume).astype("int8")

    # Save image volume as nifti file
    if not DEFAULT_TORTUOSITY_ENABLED:
        nib.save(nib.Nifti1Image(image, affine=np.eye(4)), outpath + f"Lnet_i{niter}_{file}.nii.gz")
    else:
        nib.save(nib.Nifti1Image(image, affine=np.eye(4)), outpath + f"Lnet_i{niter}_{file}_tort.nii.gz")
    
    
    
    # io.imsave(
    #     outpath + "Lnet_i{}_{}.tiff".format(niter, file),
    #     np.transpose(image, (2, 0, 1)),
    #     bigtiff=False,
    # )
