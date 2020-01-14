"""
Forecast and data assimilation using ROMS and 3Dvar
"""#import thetis as th
#from thetis import *
#import firedrake as fire

import glob
import pylab as plt
import time
import datetime
import numpy as np
import math
import os
import sys
from tensorflow import keras
import matplotlib
sys.path.append('/usr/lib/python2.7/dist-packages/')
import vtktools
import utils
import EnKFfunctions
#matplotlib.use('pdf')
from scipy.optimize import minimize

import time
import matplotlib.pyplot as plt

#Forecast by th LSTM NN
def forecastLSTM(init, model, look_back, forecastLength, eofs):
    forecast = shortRangeForecast(init, model, look_back, forecastLength)
    forecast = inverseScalerThetis(forecast, xmin, xmax, 0, 1)
    forecast = np.matmul(forecast, eofs)
    forecast = VTKVectorMaker(forecast, sectionIndexes)
    new_vx = forecast[:, 1] * sigmaVx + meanVx
    new_vy = forecast[:, 2] * sigmaVy + meanVy
    new_elev = forecast[:, 0] * sigmaElev + meanElev

    return new_vx, new_vy, new_elev

#Length of the short-range forecast
lengthForecast = 10
look_back = 1

#Paths to folders
print_directory = '/data/Flow-DA/data/'
filePath = '/data/ThetisTest/Re3000_DH2_outputs/'
print_directory_obs = '/data/Re5000_DH2_outputs2019_08_21_12_03_38/'
directory_model = '/data/Re3000_DH2_outputs2019_08_21_13_58_08/'
directory_obs = '/data/Re5000_DH2_outputs2019_08_21_12_03_38/Velocity2d/'
print_directory_model = '/data/Re3000_DH2_outputs2019_08_21_13_58_08/Vel_vtu_numpy/'
destFolder = '/data/ThetisTest/Output/DA/NN/lf_' + str(lengthForecast) + '/'

#Load model
model = keras.models.load_model(print_directory + 'lstmThetis_hvxvy.h5')

#Load historical velocity and elevation field for the covariance matrix
file_name = 'vx_field.npy'
modelThetis2Dvx = np.load(print_directory_model + file_name)
file_name = 'vy_field.npy'
modelThetis2Dvy = np.load(print_directory_model + file_name)
file_name = 'elev_field.npy'
modelThetis2Delev = np.load(print_directory_model + file_name)
meanVx = np.mean(modelThetis2Dvx, axis = 0)
meanVy = np.mean(modelThetis2Dvy, axis = 0)
meanElev = np.mean(modelThetis2Delev, axis = 0)
mean_hvxvy = np.hstack((meanElev, meanVx, meanVy))
pcs = np.load(print_directory + 'reduced_hvxvy_pcs.npy')
eofs = np.load(print_directory + 'reduced_hvxvy_eofs.npy')

xmin = pcs[:-1].min(axis=0)
xmax = pcs[:-1].max(axis=0)

#Load covariance matrices
Vvx = np.load(print_directory_model + 'reducedCV_vx.npy')
Vvy = np.load(print_directory_model + 'reducedCV_vy.npy')
Velev = np.load(print_directory_model + 'reducedCV_elev.npy')

Vvx = np.transpose(Vvx)
Vvy = np.transpose(Vvy)
Velev = np.transpose(Velev)

#Load initial velocity fields
filename_vel = 'Velocity2d/Velocity2d_1.vtu'
vx_field = []
vy_field = []
ugg = vtktools.vtu(filePath + filename_vel)
test = ugg.GetVectorField('Depth averaged velocity')
vx_field.append(test[:, 0])
vy_field.append(test[:, 1])
new_vx = np.array(vx_field)
new_vy = np.array(vy_field)

#Load initial elevation fields
filename_elev = 'Elevation2d/Elevation2d_1.vtu'
elev_field = []
ugg = vtktools.vtu(filePath + filename_elev)
test = ugg.GetScalarField('Elevation')
elev_field.append(test)
new_elev = np.array(elev_field)

#Load standard deviations
sigmaVx = np.load(print_directory + 'sigmaVx.npy')
sigmaVy = np.load(print_directory + 'sigmaVy.npy')
sigmaElev = np.load(print_directory + 'sigmaElev.npy')

#Standardisation of the fields
init = np.hstack(((new_elev - meanElev)/sigmaElev, (new_vx - meanVx)/sigmaVx, (new_vy - meanVy)/sigmaVy))
init = np.matmul(init, np.transpose(eofs))
init = scalerThetis(init, xmin, xmax, 0, 1)

sectionIndexes = [modelThetis2Delev.shape[1], modelThetis2Dvx.shape[1], modelThetis2Dvy.shape[1]]

start = time.time()

for iteration in range(5):
    print('Iteration: ' + str(iteration))
    startLoop = time.time()
    #Forecast with LSTM
    bgVx, bgVy, bgElev = forecastLSTM(init, model, look_back, lengthForecast, eofs)
    # Save these backgrounds
    phyState = np.hstack((bgVx, bgVy))

    vel_field = VTKVectorMaker(phyState, sectionIndexes[1:])

    #Save new forecasted fields
    filename = 'Velocity2d_' + str(lengthForecast*(iteration)) + '.vtu'
    ugg = vtktools.vtu(directory_model + 'Velocity2d/' + filename)
    ugg.AddVectorField('Depth averaged velocity', vel_field)
    ugg.Write(destFolder + 'Velocity2d_' + str(iteration) + '.vtu')

    filename = 'Elevation2d_' + str(lengthForecast*(iteration)) + '.vtu'
    ugg = vtktools.vtu(directory_model + 'Elevation2d/' + filename)
    ugg.AddScalarField('Elevation', bgElev)
    ugg.Write(destFolder + 'Elevation2d_' + str(iteration) + '.vtu')

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

    #3Dvar Data assimilation
    new_vx = DAvar(bgVx, yVx, Vvy, lam)
    new_vy = DAvar(bgVy, yVy, Vvy, lam)
    new_elev = DAvar(bgElev, yElev, Velev, lam)
    #v0, n, xB, d, uvwVec = PrepareForDA(bgVx, Vvx, yVx)
    #new_vx = ThreeD_VAR(Vvx, v0, n, xB, d)
    #v0, n, xB, d, uvwVec = PrepareForDA(bgVy, Vvy, yVy)
    #new_vy = ThreeD_VAR(V, v0, n, xB, d)
    #v0, n, xB, d, uvwVec = PrepareForDA(bgElev, Velev, yElev)
    #new_elev = ThreeD_VAR(V, v0, n, xB, d)
    phyState = np.hstack((new_vx, new_vy))

    vel_field = VTKVectorMaker(phyState, sectionIndexes[1:])

    filename = 'Velocity2d_' + str(lengthForecast*(iteration)) + '.vtu'
    ugg = vtktools.vtu(directory_model + 'Velocity2d/' + filename)
    ugg.AddVectorField('Depth averaged velocity', vel_field)
    ugg.Write(destFolder + 'Velocity2d_DA_' + str(iteration) + '.vtu')

    filename = 'Elevation2d_' + str(lengthForecast*(iteration)) + '.vtu'
    ugg = vtktools.vtu(directory_model + 'Elevation2d/' + filename)
    ugg.AddScalarField('Elevation', new_elev)
    ugg.Write(destFolder + 'Elevation2d_DA_' + str(iteration) + '.vtu')

    # Add the initialisation for the next loop
    init = np.hstack(((new_elev - meanElev) / sigmaElev, (new_vx - meanVx) / sigmaVx, (new_vy - meanVy) / sigmaVy))
    init = np.expand_dims(init, 0)
    init = np.matmul(init, np.transpose(eofs))
    init = scalerThetis(init, xmin, xmax, 0, 1)
    endLoop = startLoop - time.time()
    print('Elapsed time loop: ' + str(- endLoop) + ' s')
end = start - time.time()
print('Elapsed time total: ' + str(- end) + ' s')

