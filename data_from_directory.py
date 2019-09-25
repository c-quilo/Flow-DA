import sys, os
import numpy as np
sys.path.append('/usr/lib/python2.7/dist-packages/')
import vtktools
import seaborn as sns
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# run PIV on all files in a directory and make gif
def data_from_directory(directory, data_directory, print_directory, field_name):

    vx_field = []
    vy_field = []
    vm_field = []

    # part 1: create list of all image files in the directory

    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.vtu'):
            # part 2: Convert .vtu files to numpy arrays
            temp = read_vtk(directory + filename, field_name, filetype = 'vtu')
            temp = interpolate_field(temp, ranges = [[0.5, 1.5], [0.0, 0.6]], x_points = 300)

            # part 3: Save data into numpy arrays
            vx_field.append(temp[field_name][0])
            vy_field.append(temp[field_name][1])
            vm_field.append(temp[field_name][2])
            #plt.figure(figsize = (30,20))
            #plt.imshow(temp[field_name][0], cmap = 'RdBu_r')
            #plt.colorbar()
            #plt.clim(0, 0.04)
            #Prints figure into another directory
            #plt.savefig(print_directory + str(count) + '.png')
            print(count)
            count  = count + 1
            continue
        else:
            continue
    np.save(data_directory + "/vx_field", np.array(vx_field))
    np.save(data_directory + "/vy_field", np.array(vy_field))
    np.save(data_directory + "/vm_field", np.array(vm_field))




