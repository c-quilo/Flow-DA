"""
Ensemble Kalman Filter
Date: 16/09/2019
"""
import numpy as np
from tensorflow import keras
# Initial parameters
lengthForecast = 5
ensembleNumber = 50
modelStepNumber = 55
meanEnsemblePerturbation = 0
varianceEnsemblePerturbation = 0.05
observationStepNumber = 100
R = 1e-6
look_back = 5
directory_model = '/data/Re3000_DH2_outputs2019_08_21_13_58_08/Velocity2d/'
directory_obs = '/data/Re5000_DH2_outputs2019_08_21_12_03_38/Velocity2d/'
print_directory_model = '/data/Re3000_DH2_outputs2019_08_21_13_58_08/Vel_vtu_numpy/'
print_directory_obs = '/data/Re5000_DH2_outputs2019_08_21_12_03_38/Vel_vtu_numpy/'

# Load model background
file_name = 'vm_field.npy'
modelBackground = np.load(print_directory_model + file_name)
meanModel = np.mean(modelBackground, axis = 0)

pcs = np.load(print_directory_model + 'reduced_pcs.npy')
eofs = np.load(print_directory_model + 'reduced_eofs.npy')
scaler.fit(pcs)
xmin = pcs[:-1].min(axis=0)
xmax = pcs[:-1].max(axis=0)

# Load observations
observations = np.load(print_directory_obs + file_name)

# Load model
model = keras.models.load_model('/data/lstmThetis.h5')

# Perturbation of model background (Choose number of ensembles)
scaler = MinMaxScaler(feature_range=(0, 1))
initialState = []
for i in range(len(pcs)):
    #initialState.append(scaler.fit_transform(np.reshape(pcs[i, :], (-1, 1))))
    temp, u, v = scalerThetis(pcs[i, :], 0, 1)
    initialState.append(temp)
initialState = np.squeeze(np.array(initialState))

modelWithLags = lookBack(initialState, look_back)
initialState = modelWithLags[modelStepNumber, :, :]

perturbedState = []
for i in range(ensembleNumber):
    perturbedState.append(initialState + np.random.normal(meanEnsemblePerturbation, varianceEnsemblePerturbation, (initialState.shape[0], initialState.shape[1])))
perturbedState = np.array(perturbedState)

#initialStep = lookBack(pcs, look_back)
#ensembles = []
#initialState = initialStep[modelStepNumber, :, :]
#for i in range(ensembleNumber):
    temp, u, v = scalerThetis(initialState + np.random.normal(meanEnsemblePerturbation, varianceEnsemblePerturbation, (initialState.shape[0], initialState.shape[1])), 0, 1)
    ensembles.append(temp)
#perturbedState = np.array(ensembles)

# Short-range forecast
forecastEnsemble = []
for nEns in range(ensembleNumber):
    forecastEnsemble.append(shortRangeForecast(perturbedState[nEns, :, :], model, look_back, lengthForecast))
forecastEnsemble = np.array(forecastEnsemble)
forecastEnsemble = np.transpose(forecastEnsemble)

def inverseScalerThetis(xscaled, xmin, xmax, min, max):
    scale = (max - min) / (xmax - xmin)
    xInv = (xscaled/scale) - (min/scale) + xmin
    return xInv

# Inverse of the forecast ensemble
forecastEnsembleInv = []
for i in range(ensembleNumber):
    #forecastEnsembleInv.append(scaler.inverse_transform(np.reshape(forecastEnsemble[:, i], (-1, 1))))
    forecastEnsembleInv.append(inverseScalerThetis(forecastEnsemble[:, i], xmax, xmin, 0, 1))
forecastEnsembleInv = np.transpose(np.array(forecastEnsembleInv))

forecast = np.matmul(pcs[modelStepNumber + lengthForecast, :], eofs) + meanModel
modelForecast = np.matmul(np.mean(forecastEnsembleInv, axis = 1), eofs) + meanModel

# Covariance matrix
ensembleMean = np.mean(forecastEnsembleInv, axis = 1)
anomalyMatrix = forecastEnsembleInv - np.transpose(np.tile(ensembleMean, (ensembleNumber, 1)))
B = (np.matmul(anomalyMatrix, np.transpose(anomalyMatrix)))/(ensembleNumber)# - 1)

# Perturbation of observations
observationState = observations[observationStepNumber, :]

perturbedObservationState = []
for i in range(ensembleNumber):
    perturbedObservationState.append(observationState + np.random.normal(0, R, observationState.shape))
perturbedObservationState = np.array(perturbedObservationState)
perturbedObservationState = np.transpose(perturbedObservationState)

# Kalman gain

#H = np.transpose(eofs)
H = np.identity(eofs.shape[1])
Htilde = np.matmul(H, np.transpose(eofs))
D = perturbedObservationState
X = forecastEnsembleInv
K1 = np.matmul(B, np.transpose(Htilde))
K2 = np.matmul(np.matmul(Htilde, B), np.transpose(Htilde)) + R*np.identity(eofs.shape[1])

# Misfits
misfit = D - (np.matmul(Htilde, forecastEnsembleInv) + np.transpose(np.tile(meanModel, (ensembleNumber, 1))))

# Update of model state
kalmanGain = np.matmul(K1, np.linalg.inv(K2))
updateKF = np.matmul(kalmanGain, misfit)
update = forecastEnsembleInv + updateKF
meanUpdate = np.mean(update, axis=1)

#Back to Physical space
#phyState = np.matmul(np.squeeze(scaler.inverse_transform(np.reshape(meanUpdate, (-1, 1)))), np.transpose(H)) + meanModel
phyState = np.matmul(meanUpdate, eofs) + meanModel

# Save to .vtu
filename = 'Velocity2d_'+str(modelStepNumber + lengthForecast)+'.vtu'

ugg = vtktools.vtu(directory_model + filename)
ugg.AddScalarField('DA', phyState)
ugg.AddScalarField('Obs', observationState)
ugg.AddScalarField('Forecast', forecast)
ugg.AddScalarField('ForecastModel', modelForecast)

ugg.Write(print_directory_model + 'DA_results.vtu')

