from keras.models import Sequential
from keras.layers import Dense, LSTM
from processing import processing
from keras.backend import clear_session


def lstm_main(my_location):

    X,Y,val_X,val_Y,test_X,test_Y,scaler,oldmean = processing(my_location)

    model= Sequential()
    model.add(LSTM(10, input_shape=(X.shape[1], X.shape[2]), dropout=0.0, stateful=False, return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    model.summary()

    history = model.fit(X, Y, epochs=100, batch_size=32, validation_data=(val_X, val_Y), verbose=2, shuffle=False)
    
    scaler.mean_ = oldmean

    eval_model = scaler.inverse_transform([model.predict(test_X).flatten()]).flatten()

    clear_session()

    return(eval_model)