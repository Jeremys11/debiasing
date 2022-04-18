import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

###     HELPER FUNCTIONS    ###
###     HELPER FUNCTIONS    ###
###     HELPER FUNCTIONS    ###
###     HELPER FUNCTIONS    ###

def shuffle(inputs, labels):
    perm = np.random.default_rng().permutation(inputs.shape[0])
    return inputs[perm], labels[perm]

def timeslice(data_X, data_Y=[], input_width=24, shuffle=False):

    X = np.array(data_X)
    Y = np.array(data_Y)

    inputs = []
    labels = []
    for i in range(len(X)-input_width-1):
        if(Y[i+input_width]==-999 or (-999 in X[i:i+input_width][0])):
            continue
        inputs.append(X[i:i+input_width])
        if(np.size(data_Y)==0):
            labels.append(X[i+input_width])
        else:
            labels.append(Y[i+input_width])

    inputs = np.array(inputs)
    labels = np.array(labels)

    if(shuffle):
        inputs, labels = shuffle(inputs, labels)

    return np.array(inputs), np.array(labels)


##              ##
##  PROCESSING  ##
##              ##

def processing(selected_location,airnow_data,forecast_data):


    # Sort both data sources into a single array so they can be normalized together
    data = pd.DataFrame()
    data["airnow"] = airnow_data[selected_location] # *1000 here if not preprocessing
    data["forecast"] = forecast_data["o3"][selected_location] # *1000 here if not preprocessing
    data["temp"] = forecast_data["temp"][selected_location]
    data["windspeed"] = forecast_data["windspeed"][selected_location]
    data["pbl"] = forecast_data["pbl"][selected_location]

    data = np.array(data)

    # Save original data shapes for later since the normalization process destroys array shapes
    original_shape = {}
    original_shape["o3"] = data[:,:2].shape
    original_shape["temp"] = data[:,2].shape
    original_shape["windspeed"] = data[:,3].shape
    original_shape["pbl"] = data[:,4].shape

    # Normalize the data
    scaler = StandardScaler().fit(data[:,:2].reshape((-1,1)))
    o3_scaled = scaler.transform(data[:,:2].reshape((-1,1))).reshape(original_shape["o3"])
    temp_scaled = StandardScaler().fit_transform(data[:,2].reshape((-1,1))).reshape(original_shape["temp"])
    windspeed_scaled = StandardScaler().fit_transform(data[:,3].reshape((-1,1))).reshape(original_shape["windspeed"])
    pbl_scaled = StandardScaler().fit_transform(data[:,4].reshape((-1,1))).reshape(original_shape["pbl"])
    hours = [time.hour for time in list(pd.to_datetime(airnow_data.index))]
    data_scaled = np.c_[o3_scaled,temp_scaled,hours,windspeed_scaled,pbl_scaled]

    # Zero out the mean since we'll be using the scalar to de-scale error values later
    oldmean = scaler.mean_
    scaler.mean_ = 0

    # Separate normalized data sources
    airnow = data_scaled[:,0]
    forecast = data_scaled[:,1:]

    # Generate timesteps
    base_X, base_Y = timeslice(forecast, airnow)

    # Format training data for the LSTM
    features = 1
    if(len(base_X.shape)>2):
        features = base_X.shape[2]
    base_X = base_X.reshape((base_X.shape[0], base_X.shape[1], features))

    # Separate data into different sets
    test_X = base_X[int(len(base_X)*0.9)+4:]
    base_X = base_X[:int(len(base_X)*0.9)+4]
    test_Y = base_Y[int(len(base_Y)*0.9)+4:]
    base_Y = base_Y[:int(len(base_Y)*0.9)+4]

    base_X, base_Y = shuffle(base_X, base_Y)

    X = base_X[:int(len(base_X)*0.8)]
    val_X = base_X[int(len(base_X)*0.8):]

    Y = base_Y[:int(len(base_X)*0.8)]
    val_Y = base_Y[int(len(base_Y)*0.8):]

    scaler.mean_ = oldmean

    airnow_obs = scaler.inverse_transform([test_Y]).flatten()
    forecast_obs = np.array(forecast_data["o3"][selected_location])[-len(airnow_obs):]

    return(X,Y,val_X,val_Y,test_X,test_Y,scaler,oldmean,forecast_obs,airnow_obs)
