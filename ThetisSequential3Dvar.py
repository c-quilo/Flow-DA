"""
Forecast and data assimilation using Thetis
"""import thetis as th
from thetis import *
import firedrake as fire

import glob
import pylab as plt
import time
import datetime
import numpy as np
import math
import os
import sys
import matplotlib
sys.path.append('/usr/lib/python2.7/dist-packages/')
import vtktools
matplotlib.use('pdf')
from scipy.optimize import minimize
import shutil
import time
import matplotlib.pyplot as plt

#Number of iterations
timesteps = 5
#Length of forecast
lengthForecast = 10
#Origin folder path
filepath = '/data/Re3000_DH2_outputs2019_08_21_13_58_08/'
#Save destination
src = '/data/ThetisTest/Generator/'
destFolder = '/data/ThetisTest/Output/DA/CFD/lf_' + str(lengthForecast)
#Time-step of simulation (seconds)
dt = 0.01

#Load initial velocity fields
filename_vel = 'Velocity2d/Velocity2d_1.vtu'
vx_field = []
vy_field = []
ugg = vtktools.vtu(filepath + filename_vel)
test = ugg.GetVectorField('Depth averaged velocity')
vx_field.append(test[:, 0])
vy_field.append(test[:, 1])
vx = np.array(vx_field)
vy = np.array(vy_field)

#Load initial elevation fields
filename_elev = 'Elevation2d/Elevation2d_1.vtu'
elev_field = []
ugg = vtktools.vtu(filepath + filename_elev)
test = ugg.GetScalarField('Elevation')
elev_field.append(test)
elev = np.array(elev_field)

filePath = '/data/ThetisTest/Re3000_DH2_outputs/'
print_directory_obs = '/data/Re5000_DH2_outputs2019_08_21_12_03_38/'
directory_model = '/data/Re3000_DH2_outputs2019_08_21_13_58_08/Velocity2d/'
directory_obs = '/data/Re5000_DH2_outputs2019_08_21_12_03_38/Velocity2d/'
print_directory_model = '/data/Re3000_DH2_outputs2019_08_21_13_58_08/Vel_vtu_numpy/'

#Load historical velocities and elevations
file_name = 'vx_field.npy'
modelThetis2Dvx = np.load(print_directory_model + file_name)
file_name = 'vy_field.npy'
modelThetis2Dvy = np.load(print_directory_model + file_name)
file_name = 'elev_field.npy'
modelThetis2Delev = np.load(print_directory_model + file_name)

#Truncated covariance matrices of velocities and elevation (used in the DA step)
Vvx = np.load(print_directory_model + 'reducedCV_vx.npy')
Vvy = np.load(print_directory_model + 'reducedCV_vy.npy')
Velev = np.load(print_directory_model + 'reducedCV_elev.npy')
Vvx = np.transpose(Vvx)
Vvy = np.transpose(Vvy)
Velev = np.transpose(Velev)

# Load initial velocity fields
filename_vel = 'Velocity2d/Velocity2d_1.vtu'
vx_field = []
vy_field = []
ugg = vtktools.vtu(filepath + filename_vel)
test = ugg.GetVectorField('Depth averaged velocity')
vx_field.append(test[:, 0])
vy_field.append(test[:, 1])
new_vx = np.array(vx_field)
new_vy = np.array(vy_field)

# Load initial elevation fields
filename_elev = 'Elevation2d/Elevation2d_1.vtu'
elev_field = []
ugg = vtktools.vtu(filepath + filename_elev)
test = ugg.GetScalarField('Elevation')
elev_field.append(test)
new_elev = np.array(elev_field)

#Execution time starts
start = time.time()

for iteration in range(timesteps):
    print('Iteration: ' + str(iteration + 1) + ' of ' + str(timesteps) + '.' )
    startLoop = time.time()
    #Forecast by the Thetis model in forecastThetis
    bg_vx, bg_vy, bg_elev = forecastThetis(new_vx, new_vy, new_elev, src, lengthForecast, dt)
    bg_vx = np.squeeze(bg_vx)
    bg_vy = np.squeeze(bg_vy)
    bg_elev = np.squeeze(bg_elev)

    #Saves files in specified folder and restarts the simulation once done
    modifyAndCopyFiles(src, 'Velocity2d', iteration, lengthForecast, destFolder)
    modifyAndCopyFiles(src, 'Elevation2d', iteration, lengthForecast, destFolder)

    #Load observations
    lam = 0.1e-10
    R = lam * 0.9
    invR = 1 / R

    obs_vel = print_directory_obs + 'Velocity2d/Velocity2d_500.vtu'
    vx_field = []
    vy_field = []
    ugg = vtktools.vtu(obs_vel)
    test = ugg.GetVectorField('Depth averaged velocity')
    vx_field.append(test[:, 0])
    vy_field.append(test[:, 1])
    yVx = np.squeeze(np.array(vx_field))
    yVy = np.squeeze(np.array(vy_field))

    obs_elev = print_directory_obs + 'Elevation2d/Elevation2d_500.vtu'
    elev_field = []
    ugg = vtktools.vtu(obs_elev)
    test = ugg.GetScalarField('Elevation')
    elev_field.append(test)
    yElev = np.squeeze(np.array(elev_field))

    #Data assimilation 3Dvar
    new_vx = DAvar(bg_vx, yVx, Vvy, lam)
    new_vy = DAvar(bg_vy, yVy, Vvy, lam)
    new_elev = DAvar(bg_elev, yElev, Velev, lam)
    #v0, n, xB, d, uvwVec = PrepareForDA(bgVx, Vvx, yVx)
    #new_vx = ThreeD_VAR(Vvx, v0, n, xB, d)
    #v0, n, xB, d, uvwVec = PrepareForDA(bgVy, Vvy, yVy)
    #new_vy = ThreeD_VAR(V, v0, n, xB, d)
    #v0, n, xB, d, uvwVec = PrepareForDA(bgElev, Velev, yElev)
    #new_elev = ThreeD_VAR(V, v0, n, xB, d)
    endLoop = startLoop - time.time()
    print('Elapsed time loop: ' + str(- endLoop) + ' s')

end = start - time.time()
print('Elapsed time total: ' + str(- end) + ' s')

