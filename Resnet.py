from keras.models import Model
from keras.layers import Input, Dense, Activation, Conv1D, BatchNormalization, Add
from processing import processing


def resnet_model(input_layer,input_shape):
	x1 = Conv1D(filters=10, kernel_size=3, padding="same", input_shape=input_shape)(input_layer)
	x2 = Conv1D(filters=10,kernel_size=1,padding="same",input_shape=input_shape)(input_layer)
	x = BatchNormalization()(x1)
	x = Activation("relu")(x)
	x = Conv1D(filters=10, kernel_size=3, padding="same", input_shape=input_shape)(x)
	x = BatchNormalization()(x)
	y = Add()([x, x2])
	output_layer = Activation("relu")(y)
	
	return output_layer

def resnet_main():

	my_location = 250010002
	X,Y,val_X,val_Y,test_X,test_Y,scaler,oldmean = processing(my_location)

	input_shape = (X.shape[1],X.shape[2])

	inputs = Input(shape=input_shape)

	outputs = resnet_model(inputs,input_shape=input_shape)
	for i in range(0):
		outputs = resnet_model(outputs,input_shape=input_shape)

	predictions = Dense(1)(outputs)

	model = Model(inputs=inputs, outputs=predictions)
	model.compile(optimizer='adam',
				loss='mse')

	model.summary()
	history = model.fit(X, Y, epochs=1, batch_size=32, validation_data=(val_X, val_Y), verbose=2, shuffle=False)

	scaler.mean_ = oldmean

	#Generates output predictions for the input samples.
	eval_model = scaler.inverse_transform([model.predict(test_X).flatten()]).flatten()
	eval_obs = scaler.inverse_transform([test_Y]).flatten()
	eval_obs_clean = list(map(lambda x: None if(x==0) else x,eval_obs))

	return(eval_obs_clean)


