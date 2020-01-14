"""
Restarts Thetis from a new set of initial conditions
"""
import thetis as th
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
import matplotlib.pyplot as plt
import shutil
import utils


timesteps = 3
lengthForecast = 10
filepath = '/data/Re3000_DH2_outputs2019_08_21_13_58_08/'
src = '/data/ThetisTest/Generator/'
destFolder = '/data/ThetisTest/Output/lf_' + str(lengthForecast)
dt = 0.01

filename_vel = 'Velocity2d/Velocity2d_1.vtu'
vx_field = []
vy_field = []
ugg = vtktools.vtu(filepath + filename_vel)
test = ugg.GetVectorField('Depth averaged velocity')
vx_field.append(test[:, 0])
vy_field.append(test[:, 1])
vx = np.array(vx_field)
vy = np.array(vy_field)

# New elevation
filename_elev = 'Elevation2d/Elevation2d_1.vtu'
elev_field = []
ugg = vtktools.vtu(filepath + filename_elev)
test = ugg.GetScalarField('Elevation')
elev_field.append(test)
elev = np.array(elev_field)

for iteration in range(timesteps):
    print('Iteration: ' + str(iteration + 1) + ' of ' + str(timesteps) + '.' )
    new_vx, new_vy, new_elev = forecastThetis(vx, vy, elev, src, lengthForecast, dt)
    modifyAndCopyFiles(src, 'Velocity2d', iteration, lengthForecast, destFolder)
    modifyAndCopyFiles(src, 'Elevation2d', iteration, lengthForecast, destFolder)
