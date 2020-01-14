"""
Functions used by the Ensemble Kalman filter
"""
#Iterative forecast by the LSTM NN
def shortRangeForecast(perturbedState, model, look_back, lengthForecast):
#    ini_input = 0
    ini_array = np.expand_dims(perturbedState, axis = 0)
    forecast = []
    for i in range(lengthForecast):
        temp_0 = model.predict(ini_array)
        forecast.append(temp_0)
        temp_1 = np.expand_dims(temp_0, axis = 0)
        temp_2 = ini_array[:, 1:look_back, :]
        temp_3 = np.hstack((temp_2, temp_1))
        ini_array = temp_3
    forecast = np.array(forecast)
    forecast = np.squeeze(forecast)
    return forecast[lengthForecast - 1, :]

#Creates a 3D array with lag n
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

#Inverse scaler
def inverseScalerThetis(xscaled, xmin, xmax, min, max):
    scale = (max - min) / (xmax - xmin)
    xInv = (xscaled/scale) - (min/scale) + xmin
    return xInv

#Min max scaler
def scalerThetis(x, xmin, xmax, min, max):
    scale = (max - min)/(xmax - xmin)
    xScaled = scale*x + min - xmin*scale
    return xScaled

#Makes a vector to be saved in .vtu format
def VTKVectorMaker(data, sectionIndexes):
    vector = []
    for i in sectionIndexes:
        temp = data[:i]
        vector.append(temp)
        data = data[i::]
    vector.append(np.zeros(i))
    vector = np.transpose(vector)
    return vector
