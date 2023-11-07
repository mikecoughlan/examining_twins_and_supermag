def full_model(input_shape, loss='mse', early_stop_patience=20, initial_filters=32, learning_rate=1e-05):
	'''
	Concatenating the CNN models together with the MLT input

	Args:
		loss (str, optional): loss function to be uesd for training. Defaults to 'categorical_crossentropy'.
		early_stop_patience (int, optional): number of epochs the model will continue training once there
												is no longer val loss improvements. Defaults to 3.

	Returns:
		object: model configuration ready for training
		object: early stopping conditions
	'''

	# CNN model
	inputs = Input(shape=(input_shape[1], input_shape[2], 1))
	conv1 = Conv2D(initial_filters, 5, padding='same', activation='relu')(inputs)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(2)(conv1)
	conv2 = Conv2D(initial_filters*2, 3, padding='same', activation='relu')(pool1)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(2)(conv2)
	conv3 = Conv2D(initial_filters*4, 2, padding='same', activation='relu')(pool2)
	conv3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(2)(conv3)
	conv4 = Conv2D(initial_filters*4, 2, padding='same', activation='relu')(pool3)
	conv4 = BatchNormalization()(conv4)
	pool4 = MaxPooling2D(2)(conv4)
	flat = Flatten()(pool4)

	# MLT input
	mlt_input = Input(shape=(2,))
	mlt_dense = Dense(32, activation='relu')(mlt_input)

	# combining the two
	combined = concatenate([flat, mlt_dense])
	dense1 = Dense(initial_filters*4, activation='relu')(combined)
	dense1 = BatchNormalization()(dense1)
	drop1 = Dropout(0.2)(dense1)
	dense2 = Dense(initial_filters*2, activation='relu')(drop1)
	dense2 = BatchNormalization()(dense2)
	drop2 = Dropout(0.2)(dense2)
	dense3 = Dense(initial_filters, activation='relu')(drop2)
	dense3 = BatchNormalization()(dense3)
	drop3 = Dropout(0.2)(dense3)
	output = Dense(1, activation='linear')(drop3)

	model = Model(inputs=[inputs, mlt_input], outputs=output)

	opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)		# learning rate that actually started producing good results
	model.compile(optimizer=opt, loss=loss)					# Ive read that cross entropy is good for this type of model
	early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stop_patience)		# early stop process prevents overfitting

	return model, early_stop



def fit_full_model(model, xtrain, xval, ytrain, yval, train_mlt, val_mlt, early_stop, CV, delay, first_time=True):
	'''
	Performs the actual fitting of the model.

	Args:
		model (keras model): model as defined in the create_model function.

		xtrain (3D np.array): training data inputs
		xval (3D np.array): validation inputs
		ytrain (2D np.array): training target vectors
		yval (2D np.array): validation target vectors
		early_stop (keras early stopping dict): predefined early stopping function
		split (int): split being trained. Used for saving model.
		station (str): station being trained.
		first_time (bool, optional): if True model will be trainined, False model will be loaded. Defaults to True.

	Returns:
		model: fit model ready for making predictions.
	'''

	if first_time:

		# reshaping the model input vectors for a single channel
		Xtrain = xtrain.reshape((xtrain.shape[0], xtrain.shape[1], xtrain.shape[2], 1))
		Xval = xval.reshape((xval.shape[0], xval.shape[1], xval.shape[2], 1))

		print(f'XTrain: {np.isnan(Xtrain).sum()}')
		print(f'XVal: {np.isnan(Xval).sum()}')
		print(f'yTrain: {np.isnan(ytrain).sum()}')
		print(f'yVal: {np.isnan(yval).sum()}')
		print(f'train_mlt: {np.isnan(train_mlt).sum()}')
		print(f'val_mlt: {np.isnan(val_mlt).sum()}')

		model.fit(x=[Xtrain, train_mlt], y=ytrain,
					validation_data=([Xval, val_mlt], yval),
					verbose=1, shuffle=True, epochs=500,
					callbacks=[early_stop], batch_size=64)			# doing the training! Yay!

		# saving the model
		model.save(f'models/delay_{delay}/CV_{CV}.h5')

	if not first_time:

		# loading the model if it has already been trained.
		model = load_model(f'models/delay_{delay}/CV_{CV}.h5')				# loading the models if already trained

	return model


def making_predictions(model, Xtest, test_mlt, CV, boxcox_mean):
	'''
	Function using the trained models to make predictions with the testing data.

	Args:
		model (object): pre-trained model
		test_dict (dict): dictonary with the testing model inputs and the real data for comparison
		split (int): which split is being tested

	Returns:
		dict: test dict now containing columns in the dataframe with the model predictions for this split
	'''


	Xtest = Xtest.reshape((Xtest.shape[0], Xtest.shape[1], Xtest.shape[2], 1))			# reshpaing for one channel input

	predicted = model.predict([Xtest, test_mlt], verbose=1)						# predicting on the testing input data
	predicted = tf.gather(predicted, 0, axis=1)					# grabbing the positive node
	predicted = predicted.numpy()									# turning to a numpy array
	# predicted = predicted + boxcox_mean
	# predicted = inv_boxcox(predicted, 0)

	return predicted