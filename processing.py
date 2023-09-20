import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import copy
import bz2

def shuffle(inputs, labels):
    rng = np.random.default_rng()
    perm = rng.permutation(inputs.shape[0])
    return inputs[perm], labels[perm]

def timeslice(data_X, data_Y=[], input_width=24, shuffle=False):

    X = np.array(data_X)
    Y = np.array(data_Y)

    inputs = []
    labels = []
    for i in range(len(X)-input_width-1):
        if(Y[i+input_width]==None or (None in X[i:i+input_width][0])):
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


def processing(selected_location):

    data = bz2.BZ2File('bz2_test.pbz2', 'rb')
    airnow_data, forecast_data = pickle.load(data)
    #airnow_data, forecast_data = pickle.load(open("data/my_data_o3.p", "rb"))
    airnow_data.index.name = "date_time"

    # Sort both data sources into a single array so they can be normalized together
    data = pd.DataFrame()
    data["airnow"] = airnow_data[selected_location]
    data["forecast"] = forecast_data["O3"][selected_location,:]
    data["temp"] = forecast_data["TEMP2"][selected_location,:]
    data["windspeed"] = forecast_data["WSPD10"][selected_location,:]
    data["pbl"] = forecast_data["PBL2"][selected_location,:]

    #Drop rows with NAN values
    #data.dropna(inplace=True)
    copy_data = copy.deepcopy(data)
    copy_data.index.name = "date_time"

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
    #o3_scaled = data[:,:2]
    #temp_scaled = data[:,2]
    #windspeed_scaled = data[:,3]
    #pbl_scaled = data[:,4]
    hours = [time.hour for time in list(pd.to_datetime(copy_data.index))]
    data_scaled = np.c_[o3_scaled,temp_scaled,hours,windspeed_scaled,pbl_scaled]

    # Zero out the mean since we'll be using the scalar to de-scale error values later
    oldmean = scaler.mean_
    scaler.mean_ = 0

    # Separate normalized data sources
    airnow = data_scaled[:,0]
    forecast = data_scaled[:,1:]

    # Generate timesteps
    base_X, base_Y = timeslice(forecast, airnow)

    # Format training data for the Resnet
    features = base_X.shape[2]
    base_X = base_X.reshape((base_X.shape[0], base_X.shape[1], features))

    # Separate data into different sets
    #For some reason, changing the *0.8 here changes the output in a constant way for any model
    test_X = base_X[int(len(base_X)*0.8):]
    base_X = base_X[:int(len(base_X)*0.8)]
    test_Y = base_Y[int(len(base_Y)*0.8):]
    base_Y = base_Y[:int(len(base_Y)*0.8)]

    base_X, base_Y = shuffle(base_X, base_Y)

    X = base_X[:int(len(base_X)*0.8)]
    val_X = base_X[int(len(base_X)*0.8):]

    Y = base_Y[:int(len(base_X)*0.8)]
    val_Y = base_Y[int(len(base_Y)*0.8):]

    return(X,Y,val_X,val_Y,test_X,test_Y,scaler,oldmean)