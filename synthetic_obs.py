import sys, os
import numpy as np
sys.path.append('/usr/lib/python2.7/dist-packages/')
import vtktools
import seaborn as sns
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# run PIV on all files in a directory and make gif
def synthetic_obs(directory_obs, field_name, mean, variance):

    vx_obs = []
    vy_obs = []
    vm_obs = []

    # part 1: create list of all image files in the directory

    count = 0
    for filename in os.listdir(directory_obs):
        if filename.endswith('.vtu'):
            # part 2: Convert .vtu files to numpy arrays
            temp = read_vtk(directory_obs + filename, field_name, filetype = 'vtu')
            temp = interpolate_field(temp, ranges = [[0.5, 1.5], [0.0, 0.6]], x_points = 300)
            # part 3: Save data into numpy arrays
            vx_obs.append(temp[field_name][0] + np.random.normal(mean, variance, temp[field_name][0].shape))
            vy_obs.append(temp[field_name][1] + np.random.normal(mean, variance, temp[field_name][0].shape))
            vm_obs.append(temp[field_name][2]+ np.random.normal(mean, variance, temp[field_name][0].shape))
            print(count)
            count = count + 1
            continue
        else:
            continue
    np.save(directory_obs + "/vx_obs", np.array(vx_obs))
    np.save(directory_obs + "/vy_obs", np.array(vy_obs))
    np.save(directory_obs + "/vm_obs", np.array(vm_obs))




