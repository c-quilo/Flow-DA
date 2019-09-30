import sys, os
import numpy as np
import pandas as pd
sys.path.append('/usr/lib/python2.7/dist-packages/')
sys.path.append('/data/PycharmProjects/DA_imaging/')
import vtktools
import seaborn as sns
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import griddata
from eofs.standard import Eof
from threeDvarFunctions import *
from scipy.optimize import minimize
from preprocessing import *
import preprocessing
from sklearn.metrics import mean_squared_error

from scipy.interpolate import griddata

def read_vtk(file_name, field_name, filetype="vtu"):
    """Read the data in vtk/vtu/vtp files
    Return a dictionary containing 'mesh' and other physics field data in numpy form
    """

    reader = vtk.vtkXMLUnstructuredGridReader()
    if filetype == "vtp":
        reader = vtk.vtkXMLPolyDataReader()

    reader.SetFileName(file_name)
    reader.Update()  # Needed because of GetScalarRange
    output = reader.GetOutput()

    # Get the field data
    vtk_array_dict = dict()
    numpy_array_dict = dict()

    #field_names = [item for item in field_name]

    #while isinstance(field_names[0], list):
    #    field_names = [item for item in field_names[0]]

    # Get the mesh data
    nodes_vtk_array = reader.GetOutput().GetPoints().GetData()
    numpy_array_dict['mesh'] = vtk_to_numpy(nodes_vtk_array)

    vtk_array_dict[field_name] = output.GetPointData().GetArray(field_name)
    numpy_array_dict[field_name] = vtk_to_numpy(output.GetPointData().GetArray(field_name))

    return numpy_array_dict

def interpolate_field(data, field_name, ranges=None, x_points=100, dim=2):
    """Do interpolation in the field using the mesh point data
    to make the a complete data matrix
    2D interpolation is developed now, there are three components:
    vx, vy, v_magnitude
    """

    #field_names = [item for item in data.keys()].pop()
    #while not isinstance(field_names, list):
    #    field_names = [field_names]

    if ranges is None:
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
    else:
        xmin, xmax = ranges[0][0], ranges[0][1]
        ymin, ymax = ranges[1][0], ranges[1][1]

    npts_x = x_points
    npts_y = np.floor(npts_x * (ymax - ymin) / (xmax - xmin))
    # define grid
    xi = np.linspace(xmin, xmax, npts_x)
    yi = np.linspace(ymin, ymax, npts_y)

    x, y, z = data['mesh'][:, 0], data['mesh'][:, 1], data['mesh'][:, 2]

    field_value_dict = dict()
    if dim == 2:
        # 2D interpolate
        name = field_name
        size = data[field_name].shape
        data_components = []
        for i in range(size[1] - 1):
            data_components.append(griddata((x, y), data[name][:, i], (xi[None, :], yi[:, None]), method='cubic'))

        magnitude = np.sqrt(np.power(data[name][:, 0], 2) + np.power(data[name][:, 1], 2))
        data_components.append(griddata((x, y), magnitude, (xi[None, :], yi[:, None]), method='cubic'))
        field_value_dict[name] = data_components
    else:
        print("Under construction")
    return field_value_dict

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
            temp_interp = interpolate_field(temp, field_name, ranges = [[0.5, 1.5], [0.0, 0.6]], x_points = 300)

            # part 3: Save data into numpy arrays
            vx_field.append(temp_interp[field_name][0])
            vy_field.append(temp_interp[field_name][1])
            vm_field.append(temp_interp[field_name][2])
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
    np.save(data_directory + "vx_field", np.array(vx_field))
    np.save(data_directory + "vy_field", np.array(vy_field))
    np.save(data_directory + "vm_field", np.array(vm_field))

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
            temp = interpolate_field(temp, field_name, ranges = [[0.5, 1.5], [0.0, 0.6]], x_points = 300)
            # part 3: Save data into numpy arrays
            vx_obs.append(temp[field_name][0] + np.random.normal(mean, variance, temp[field_name][0].shape))
            vy_obs.append(temp[field_name][1] + np.random.normal(mean, variance, temp[field_name][0].shape))
            vm_obs.append(temp[field_name][2] + np.random.normal(mean, variance, temp[field_name][0].shape))
            print(count)
            count = count + 1
            continue
        else:
            continue
    np.save(directory_obs + "vx_obs", np.array(vx_obs))
    np.save(directory_obs + "vy_obs", np.array(vy_obs))
    np.save(directory_obs + "vm_obs", np.array(vm_obs))

# Definition of gradient of J
def gradJ(v):
    import numpy as np
    Vv = np.dot(V, v)
    Jmis = np.subtract(Vv, d)
    g1 = Jmis.copy()
    VT = np.transpose(V)
    g2 = np.dot(VT, g1)
    gg2 = np.multiply(invR, g2)
    ggJ = v + gg2
    return ggJ

def PrepareForDA(ug, V):
    import numpy as np

    ##ug.GetFieldNames()
    ##uvwVec = ug.GetScalarField('Tracer')
    uvwVec = ug
    n = len(uvwVec)
    #m = trnc

    xB = uvwVec.copy()
    x0 = np.ones(n)

    Vin = np.linalg.pinv(V)
    v0 = np.dot(Vin, x0)
    HxB = np.copy(xB)
    d = np.subtract(y, HxB)

    return v0, n, xB, d, uvwVec
# Definition of cost function J
def J(v):
    import numpy as np
    vT = np.transpose(v)
    vTv = np.dot(vT, v)
    Vv = np.dot(V, v)
    Jmis = np.subtract(Vv, d)
    JmisT = np.transpose(Jmis)
    RJmis = JmisT.copy()
    J1 = invR * np.dot(Jmis, RJmis)
    Jv = (vTv + J1) / 2
    return Jv
# Compute the minimum of cost function J

def ThreeD_VAR(V, v0, n, xB, d):
    import numpy as np

    res = minimize(J, v0, method='L-BFGS-B', jac=gradJ, options={'disp': False})

    vDA = np.array([])
    vDA = res.x
    deltaXDA = np.dot(V, vDA)
    xDA = xB + deltaXDA

    return xDA

directory = '/data/Re3000_DH2_outputs2019_08_21_13_58_08/Velocity2d/'
data_directory = directory
print_directory = '/data/Re3000/'
directory_obs = '/data/Re5000_DH2_outputs2019_08_21_12_03_38/Velocity2d/'

field_name = 'Depth averaged velocity'

data_from_directory(directory, data_directory, print_directory, field_name)
synthetic_obs(directory_obs, field_name, 0, 0.001)

#Covariance

file_name = 'vm_field.npy'

modelThetis = np.load(data_directory+file_name)
# Projection to 2D
modelThetis2D = []
for i in range(len(modelThetis)):
    temp = modelThetis[i,:,:].flatten()
    modelThetis2D.append(temp)

modelThetis2D = np.array(modelThetis2D)

#PCA
targetVariance = 0.99
solver = Eof(modelThetis2D)
varianceCumulative = np.cumsum(solver.varianceFraction())
#reducedCovariance = solver.pcs(npcs = np.where(varianceCumulative > 0.99)[0][0], pcscaling = 2)
reducedCovariance = solver.eofs(neofs = np.where(varianceCumulative > 0.99)[0][0], eofscaling = 2)
np.save(data_directory + 'reducedCV', reducedCovariance)
# Load
#3Dvar

# Load truncated SVD matrix
#V = np.load(data_directory + 'reducedCV.npy')
V = np.transpose(reducedCovariance)
lam = 0.1e-10

synthObs = (modelThetis2D + np.random.normal(0, 0.01, modelThetis2D.shape))[500, :]

R = lam * 0.9
invR = 1 / R

y = synthObs
#for ts in 10:
ts = 10
start = time.time()
ug = modelThetis2D[10, :]
v0, n, xB, d, uvwVec = PrepareForDA(ug)
xDA = ThreeD_VAR(V, v0, n, xB, d)
print(time.time() - start)

plt.clf()

fig = plt.figure()
plt.subplot(221)
plt.imshow(np.reshape(ug, (modelThetis.shape[1], modelThetis.shape[2])), cmap = 'Blues')
rmseM = np.sqrt(mean_squared_error(synthObs, ug))
plt.title('Model (RMSE = '+ '%7.6f' % rmseM + ')')
plt.colorbar()
plt.clim(0, 0.05)

plt.subplot(222)
plt.imshow(np.reshape(synthObs, (modelThetis.shape[1], modelThetis.shape[2])), cmap = 'Blues')
plt.title('Synthetic observations')
plt.colorbar()
plt.clim(0, 0.05)

plt.subplot(223)
plt.imshow(np.reshape(xDA, (modelThetis.shape[1], modelThetis.shape[2])), cmap = 'Blues')
rmseDA = np.sqrt(mean_squared_error(synthObs, xDA))
plt.title('3D-var (RMSE = '+ '%7.6f' % rmseDA + ')')
plt.colorbar()
plt.clim(0, 0.05)

plt.subplot(224)
plt.imshow(np.reshape(xDA, (modelThetis.shape[1], modelThetis.shape[2])) - np.reshape(synthObs, (modelThetis.shape[1], modelThetis.shape[2])), cmap = 'RdBu_r')
plt.title('Residual error')
plt.colorbar()
plt.clim(-0.05, 0.05)

fig.suptitle('Error Obs = ' + str(lam)) # or plt.suptitle('Main title')
