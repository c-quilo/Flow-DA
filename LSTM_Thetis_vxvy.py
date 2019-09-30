import numpy as np
import sys, os
sys.path.append('/usr/lib/python2.7/dist-packages/')
import vtktools
import eofs
from eofs.standard import Eof
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import tensorflow.keras as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RNN
from keras.layers import GRU
from keras.layers import Dropout
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from keras.models import save_model
from keras.models import load_model
import dateutil
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from tensorflow.keras.callbacks import TensorBoard
import pickle

print_directory_model = '/data/Flow-DA/data/'
file_name = 'vx_field.npy'
modelThetis2Dvx = np.load(print_directory_model + file_name)
file_name = 'vy_field.npy'
modelThetis2Dvy = np.load(print_directory_model + file_name)
modelData = np.hstack((modelThetis2Dvx, modelThetis2Dvy))

#PCA
targetVariance = 0.99
solver = Eof(modelData)
varianceCumulative = np.cumsum(solver.varianceFraction())
pcs = solver.pcs(npcs = np.where(varianceCumulative > 0.99)[0][0], pcscaling = 0)
eofs = solver.eofs(neofs = np.where(varianceCumulative > 0.99)[0][0], eofscaling = 0)
np.save(print_directory_model + 'reduced_vxvy_pcs', pcs)
np.save(print_directory_model + 'reduced_vxvy_eofs', eofs)

meanThetis_vxvy = np.mean(modelData, axis = 0)
np.save(print_directory_model, meanThetis_vxvy)

X = pcs[:-1]
y = pcs[1::]

scaler = MinMaxScaler(feature_range=(0, 1))

scaler.fit(pcs)
xmin = pcs[:-1].min(axis=0)
xmax = pcs[:-1].max(axis=0)

def scalerThetis(x, xmin, xmax, min, max):
    scale = (max - min)/(xmax - xmin)
    xScaled = scale*x + min - xmin*scale
    return xScaled

X = scaler.transform(X)
y = scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

def lookBack(X, look_back = 1):
    #look_back = 10
    X_lb = np.empty((X.shape[0] - look_back + 1, look_back, X.shape[1]))
    #X_test = np.empty((look_back, X_test.shape[1], X_train.shape[0] - look_back + 1))
    ini = 0
    fin = look_back
    for i in range(X.shape[0] - look_back + 1):
        X_lb[i,:,:] = (X[ini:fin,:])
        ini = ini + 1
        fin = fin + 1
    return X_lb

#Do something similar for testing
look_back = 5
y_train = y_train[look_back - 1:, :]
y_test = y_test[look_back - 1:, :]

X_train = lookBack(X_train, look_back)
X_test = lookBack(X_test, look_back)
X_total = lookBack(X, look_back)

#Create and fit the LSTM network
np.random.seed(42)
model = keras.models.Sequential([
    keras.layers.LSTM(units = 12, input_shape = (X_train.shape[1], X_train.shape[2]), return_sequences = True),
    keras.layers.Dropout(0.1),
#model.add(LSTM(50, return_sequences = True))
    keras.layers.LSTM(200, return_sequences = False),
    keras.layers.Dropout(0.1),
#model.add(LSTM(100, return_sequences = True))
#model.add(LSTM(100, return_sequences = False))
#model.add(Dropout(0.9))
#model.add(Dropout(0.4))
#model.add(Dropout(0.2))
    keras.layers.Dense(12, activation = 'relu')])
tensorboard = TensorBoard(log_dir = 'data/logs/training/{}'.format(time()))
#logdir="logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 10)
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
#adam = optimizers.Adam(lr=0.0085, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
rmsprop = optimizers.RMSprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
#adadelta = optimizers.Adadelta(lr=1.0, rho=0.8, epsilon=None, decay=0.0)
nadam = optimizers.Nadam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, schedule_decay = 0.004)
model.compile(loss='mean_squared_error', metrics=['mae'], optimizer='nadam')
model.fit(X_train, y_train, epochs=1000, batch_size=256, verbose=2, callbacks=[tensorboard], validation_data = (X_test, y_test), shuffle = True)

#os.system('scp lstm.h5 caq13@taizi.doc.ic.ac.uk:/data/lstm.h5')
#model = load_model('/data/lstm.h5')

predictions_test = model.predict(X_test)
predictions_train = model.predict(X_train)
pred_train = scaler.inverse_transform(predictions_train)
pred_test = scaler.inverse_transform(predictions_test)

y_train_inv = scaler.inverse_transform(y_train)
y_test_inv = scaler.inverse_transform(y_test)###
#y_test_inv_plot = np.concatenate((np.nan * np.zeros([X_train.shape[0], X_train.shape[1]]), y_test_inv))
#pred_test_plot = np.concatenate((np.nan * np.zeros([X_train.shape[0], X_train.shape[1]]), pred_test))

plt.clf()
plt.figure(1)
for dimension in range(22):
    plt.subplot(5, 5, dimension + 1)
    sns.lineplot(range(X_train.shape[0]), y_train_inv[:, dimension])
    sns.lineplot(range(X_train.shape[0]), pred_train[:, dimension])##
    sns.lineplot(range(X_train.shape[0], X_train.shape[0] + pred_test.shape[0]), y_test_inv[:,dimension])
    sns.lineplot(range(X_train.shape[0], X_train.shape[0] + pred_test.shape[0]), pred_test[:,dimension])

"""
Recursive forecast using the output of the LSTM as input for the next time-step
"""

#X_test = np.reshape(X_test, (X_test.shape[2], X_test.shape[1], X_test.shape[0]))
ini_input = 0
ini_array = np.expand_dims(X_total[ini_input, :, :], axis = 0)
forecast = []
for i in range(1500):
    temp_0 = model.predict(ini_array)
    forecast.append(temp_0)
    temp_1 = np.expand_dims(temp_0, axis = 0)
    temp_2 = ini_array[:, 1:look_back, :]
    temp_3 = np.hstack((temp_2, temp_1))
    ini_array = temp_3
forecast = np.array(forecast)
forecast = np.squeeze(forecast)
forecast = scaler.inverse_transform(forecast)
forecast = np.matmul(forecast, eofs) + meanThetis
forecast = np.vstack((modelThetis2D[:look_back,:], forecast))

#Convert to physical space
#forecast_Phys = np.matmul(forecast, eofs) + meanThetis



rom = np.matmul(pcs, eofs) + meanThetis
prediction = model.predict(X_total)
prediction = scaler.inverse_transform(prediction)
prediction = np.matmul(prediction, eofs) + meanThetis
prediction = np.vstack((modelThetis2D[:look_back,:], prediction))

plt.figure(2)
for dimension in range(25):
    plt.subplot(5, 5, dimension + 1)
    sns.lineplot(range(X_train.shape[0]), y_train_inv[:, dimension])
    sns.lineplot(range(X_train.shape[0]), pred_train[:, dimension])
    sns.lineplot(range(X_train.shape[0]), forecast[:,dimension])##
    sns.lineplot(range(X_train.shape[0], X_train.shape[0] + pred_test.shape[0]), y_test_inv[:,dimension])
    sns.lineplot(range(X_train.shape[0], X_train.shape[0] + pred_test.shape[0]), pred_test[:,dimension])
    plt.xlim(0,200)

"""
Creating the .vtu files to open with Paraview
"""
contador = 0
handicap = 60
for i in range(1501):
    ug = vtktools.vtu(directory_model + 'Velocity2d_' + str(i)+'.vtu')
    ug.AddScalarField('ROM', rom[i, :])
    ug.AddScalarField('LSTM_Tracer', prediction[i, :])
    ug.AddScalarField('LSTM_FreeForecast', forecast[i, :])
    ug.AddScalarField('Depth averaged velocity', modelThetis2D[i, :])
    #ug.RemoveField('ROM')
    #ug.AddField('ROM', rom[contador + handicap, :])
    ug.Write(directory_model + 'Velocity2d_' + str(i)+'.vtu')
    contador += 1
    print(contador)

# Back to physical space
r_pred_train = np.matmul(pred_train, eofs)
r_pred_test = np.matmul(pred_test, eofs)

r_y_train = np.matmul(y_train_inv, eofs)
r_y_test = np.matmul(y_test_inv, eofs)
y_test_inv_plot = np.concatenate((np.nan * np.zeros([X_train.shape[0], r_pred_train.shape[1]]), r_y_test))
pred_test_plot = np.concatenate((np.nan * np.zeros([X_train.shape[0], r_pred_train.shape[1]]), r_pred_test))

plt.figure(3)
for dimension in range(49):
    plt.subplot(7, 7, dimension + 1)
    sns.lineplot(range(X_train.shape[0]), r_y_train[:, dimension])
    sns.lineplot(range(X_train.shape[0]), r_pred_train[:, dimension])##
    #plt.figure(1)
    sns.lineplot(range(X_train.shape[0], X_train.shape[0] + pred_test.shape[0]), r_y_test[:,dimension])
    sns.lineplot(range(X_train.shape[0], X_train.shape[0] + pred_test.shape[0]), r_pred_test[:,dimension])



model.save(print_directory_model + 'lstmThetis_vxvy.h5')
