import pandas as pd
from pathlib import Path
import requests
import config 
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
#TODO Sanitize ticker input
#TODO Fix line length

def ticker_input(message = "Enter the ticker of the stock you want to examine: "):
	""" Ask the user for the ticker they want to examine and return the chosen ticker

	message: Message to display to the user when asking for a ticker input
	"""
	ticker = input(message)
	return ticker.upper()

def source_input(message = "Choose your data source. Enter k for Kaggle or a for Alphavantage: "):
	""" Ask the user for the data source they want to use and return the chosen data source. Blank input defaults to Alphavantage.

	message: Message to display to the user when asking for a data source input
	"""
	source = input(message)
	if source == "a" or source == "":
		return "alphavantage"
	elif source == "k":
		return "kaggle"
	else:
		return source_input("Invalid data source. Enter k for Kaggle or a for Alphavantage")

def load_data_kaggle(ticker):
	""" Load stock market data from Kaggle and return Pandas dataframe.

	ticker: Market identifier that specifies the stock to get data for. Ex. "AAL" for American Airlines
	"""
	file_path = Path("Stocks/") / (ticker + ".us.txt")
	if file_path.exists():
		df = pd.read_csv(file_path, delimiter=",", usecols=["Date","Low","High","Close","Open"]) #ignore volume, openInt columns
		print("Loaded data for ticker " + ticker + " from Kaggle")
		return df.sort_values("Date")
	else:
		return load_data(ticker_input("Invalid ticker, please try again: "), "kaggle") #Set new input to load_data's ticker parameter

def load_data_alphavantage(ticker):
	""" Load stock market data from Alphavantage and return Pandas dataframe.

	ticker: Market identifier that specifies the stock to get data for. Ex. "AAL" for American Airlines
	"""
	api_key = config.api_key #Pull api_key from config.py
	filename = ticker + ".csv" #Cache requested csv files
	file_path = Path("Cache/") / filename
	if file_path.exists():  #Call cached csv- testing purposes only
		print("Data for ticker " + ticker + " already exists. Loading Alphavantage data from CSV")
		df = pd.read_csv(file_path, delimiter=",", usecols=["Date","Low","High","Close","Open"])
		return df
	else:  #considered directly downloading csv but wanted to avoid direct download without warning
		url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=" + ticker + "&outputsize=full&apikey=" + api_key
		r = requests.get(url_string)
		data_json = r.json()
		if len(data_json) != 1:  #valid response
			data = data_json["Time Series (Daily)"]
			df = pd.DataFrame(columns=["Date","Low","High","Close","Open"]) 
			for date, values in data.items():
				formatted_date = dt.datetime.strptime(date, "%Y-%m-%d") #https://docs.python.org/3/library/datetime.html#datetime-objects
				row = [formatted_date.date(),float(values["3. low"]),float(values["2. high"]), 
						float(values["5. adjusted close"]),float(values["1. open"])] #new row
				df.loc[-1,:] = row  #select row -1 and all columns and set to created data row
				df.index = df.index + 1 #increment row to add data to
				df = df.sort_values("Date")
			df.to_csv(file_path)
			print("Data saved to: " + filename)
			return df
		else:
			return load_data(ticker_input("Invalid ticker, please try again: "), "alphavantage")

def load_data(ticker, data_source):
	""" Load stock market data from designated data source.
	Return a pandas DataFrame containing the data sorted by date.

	ticker:      Market identifier that specifies the stock to get data for. Ex. "AAL" for American Airlines
	data_source: Designates which online resource to get data from, Kaggle or Alphavantage
	"""
	if data_source == "kaggle":  #Load directly into DataFrame from csv file
		return load_data_kaggle(ticker)

	if data_source == "alphavantage":  #Load JSON data into csv into DataFrame. Reference: https://www.alphavantage.co/documentation/#daily
		return load_data_alphavantage(ticker)

def graph_imported_data(df, ticker):
	""" Graph stock market data for designated ticker.

	df: 	Pandas DataFrame containing ticker stock data
	ticker:	Market identifier that specifies the stock to get data for. Ex. "AAL" for American Airlines
	"""
	plt.figure(num="Imported Data", figsize = (10,7))
	plt.title("Historical Stock Market Data of " + ticker)
	plt.xlabel("Date")
	plt.ylabel("Daily Adjusted Closing Price")  #adjusted closing price factors in dividends and splits
	plt.xticks(range(0,df.shape[0],int(df.shape[0]/10)),df['Date'].loc[::int(df.shape[0]/10)], rotation=25)  #convert x values to dates
	plt.plot(range(df.shape[0]), df["Close"])  #range created from number of rows in df
	plt.show()

def split_train_test_data(df, training_percentage):
	""" Split stock data into training and test data and return training and test sets.

	df: 				 Pandas DataFrame containing ticker stock data
	training_percentage: Percentage of data that will be included in the training set
	"""
	closing_prices = df.loc[:,"Close"].values  #select all rows, close column and convert to np array
	length = len(closing_prices)
	training_prices = closing_prices[:int(training_percentage*length)]
	test_prices = closing_prices[int(training_percentage*length):]
	print("Training Set Length: " + str(len(training_prices)))
	print("Test Set Length: " + str(len(test_prices)))
	return training_prices, test_prices

def scale_data(train_data, test_data, window_count):
	""" Scale and normalize the data to train network and return training and test sets.

	train_data:	  Set of imported data to train network
	test_data:	  Set of imported data to test network
	window_count: Number of smoothing windows to create
	"""
	scaler = MinMaxScaler()  #scale all data from 0 to 1
	train_data = train_data.reshape(-1,1)  #n rows 1 column 2D array- 1 column because 1 feature(close price)
	test_data = test_data.reshape(-1,1)
	smooth_window_size = int(len(train_data)/window_count)
	for step in range(0, len(train_data), smooth_window_size):
		scaler.fit(train_data[step:step+smooth_window_size])  #Calculate min and max for scaling
		train_data[step:step+smooth_window_size] = scaler.transform(train_data[step:step+smooth_window_size])
	
	if len(train_data[step+smooth_window_size:]) > 0:  #if any data remaining
		scaler.fit(train_data[step+smooth_window_size:])  #fit and scale remaining data
		train_data[step+smooth_window_size:] = scaler.transform(train_data[step+smooth_window_size:])

	train_data = train_data.reshape(-1)  #transform back to 1D array
	test_data = scaler.transform(test_data).reshape(-1)  #Don't fit scaler to test_data and transform back to 1D array
	return train_data, test_data

def smooth_data(train_data, test_data, EMA, gamma):
	""" Use exponential moving average smoothing to remove jaggedness from training set data.
	Return a concatenation of the training and test set as well as the smoothed training set and test set
	
	train_data:  Set of scaled and normalized data to train network
	test_data:   Set of scaled and normalized data to test network
	EMA:		 Exponential moving average value
	gamma:		 Contribution of previous data to EMA
	"""
	for i in range(len(train_data)):
		EMA = gamma*train_data[i] + (1-gamma)*EMA
		train_data[i] = EMA  #only smooth training set

	all_preprocessed_data = np.concatenate([train_data, test_data])
	return train_data, test_data, all_preprocessed_data

def graph_preprocessed_data(df,train_data, ticker):
	""" Graph preprocessed stock market data for designated ticker.

	df: 		Pandas DataFrame containing ticker stock data
	train_data: Set of scaled, normalized, and smoothed data to train network
	ticker:		Market identifier that specifies the stock to get data for. Ex. "AAL" for American Airlines
	"""
	plt.figure(num="Preprocessed Data", figsize = (10,7))
	plt.title("Preprocessed Data for " + ticker)
	plt.xlabel("Date")
	plt.ylabel("Daily Adjusted Closing Price")  
	plt.xticks(range(0,train_data.shape[0],int(train_data.shape[0]/10)), 
		df['Date'].loc[:train_data.shape[0]:int(train_data.shape[0]/10)], rotation=25)
	plt.plot(range(train_data.shape[0]), train_data)  #range created from number of rows in df
	plt.show()

class DataGeneratorSeq(object):

	def __init__(self, prices, batch_size, time_steps):
		""" Initialize data generator.
		prices: 	Full training set
		batch_size: Size of batches of data
		time_steps: Number of steps in each training
		"""
		self.prices = prices  #train_data
		self.prices_len = len(prices) - time_steps  #length of remaining dataset
		self.batch_size = batch_size  #size of batch of data
		self.time_steps = time_steps  #number of time steps in one training
		self.segments = self.prices_len // self.batch_size  #number of full batches
		self.cursor = [x * self.segments for x in range(self.batch_size)]  #array of starting index of each batch

	def next_batch(self):
		""" Return next batch of data and its labels."""
		batch_data = np.zeros(self.batch_size, dtype=np.float32)  #Instantiate empty batch_data np arrays
		batch_labels = np.zeros(self.batch_size, dtype=np.float32)
		for data_point_index in range(self.batch_size):  #Loop through each data point in batch
			if self.cursor[data_point_index] + 1 >= self.prices_len:  #if overall index of start of batch + 1 is greater than or equal to dataset length
				self.cursor[data_point_index] = np.random.randint(0,(data_point_index+1)*self.segments)  #set overall index to a randow index in the dataset

			batch_data[data_point_index] = self.prices[self.cursor[data_point_index]]  #batch data at index data_point_index set to data in prices at data_point_index
			batch_labels[data_point_index] = self.prices[self.cursor[data_point_index] + np.random.randint(0,5)]  #batch label at index data_point_index set to price data close to data_point_index
			self.cursor[data_point_index] = (self.cursor[data_point_index]+1) % self.prices_len  #update cursor position to select index

		return batch_data, batch_labels

	def output_batches(self):
		""" Return set of batches of input data."""
		all_batch_data = []
		all_batch_labels = []
		for t in range(self.time_steps):
			data, labels = self.next_batch()
			all_batch_data.append(data)
			all_batch_labels.append(labels)

		return all_batch_data, all_batch_labels


def preprocessing():
	""" Perform preprocessing for dataset."""
	training_percentage = .9
	window_count = 5
	EMA = 0.0
	gamma = 0.1
	ticker = ticker_input()
	source = source_input()
	df = load_data(ticker, source)
	print(df.head()) #Test data import
	graph_imported_data(df, ticker)
	train_data, test_data = split_train_test_data(df, training_percentage)
	print("Data Split")
	train_data, test_data = scale_data(train_data, test_data, window_count)
	print("Data Scaled")
	train_data, test_data, all_preprocessed_data = smooth_data(train_data, test_data, EMA, gamma)
	print("Data Normalized")
	graph_preprocessed_data(df, train_data, ticker) #order of scaling and normalizing matters
	print("Preprocessing Completed")
	return train_data, test_data, all_preprocessed_data, training_percentage, df

def data_augmentation(train_data, batch_size, future_steps):
	""" Create more data for training.
	train_data:   Set of scaled, normalized, and smoothed data to train network
	batch_size:   Number of augmented data points per batch
	future_steps: Number of time steps into future
	"""
	dgs = DataGeneratorSeq(train_data, batch_size, future_steps)  #generate data
	all_batch_data, all_batch_labels = dgs.output_batches()  
	for index, (data, label) in enumerate(zip(all_batch_data, all_batch_labels)):
		print("Batch " + str(index))
		print("\tInputs: ", data)
		print("\tOutputs: ", label)
	return all_batch_data, all_batch_labels

def create_tf_input_output(batch_size, dimensionality, time_steps):
	""" Create tf placeholders based on model parameters.
	batch_size: 	Number of data points per batch
	dimensionality: Dimensionality of input data
	time_steps: 	Number of time steps into future
	"""
	train_inputs = []
	train_outputs = []
	for step in range(time_steps):
		train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, dimensionality], name="train_inputs_" + str(step)))  #create inputs placeholder
		train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size, 1], name="train_outputs_" + str(step)))  #create outputs placeholder
	return train_inputs, train_outputs

def create_LSTM_layers(neurons_per_cell, num_layers, dropout):
	"""  Define parameters for LSTM layers and regression layer.  Use tf's MultiRNNCell for LSTM layers
	neurons_per_cell: Number of hidden nodes in each LSTM layer
	num_layers: 	  Number of LSTM layers
	dropout: 		  Amount of output and state to forget
	"""
	lstm_cells = [tf.contrib.rnn.LSTMCell(  #https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/LSTMCell
		num_units=neurons_per_cell[layer],  #number of nodes in layer
		state_is_tuple=True,  #states are 2-tuples of c_state and m_state
		initializer= tf.contrib.layers.xavier_initializer()  #weight initialization strategy https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer
		) for layer in range(num_layers)]  #create a 1D array of LSTM layers
		
	drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(  #Adds dropout to input and output of layer https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/DropoutWrapper
		lstm,  #layer
		input_keep_prob = 1.0,  #proportion of input to keep
		output_keep_prob = 1.0 - dropout,  #proportion of output to keep
		state_keep_prob = 1.0 - dropout  #proportion of state to keep
		) for lstm in lstm_cells]  #apply dropout to each layer
	drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)  #store LSTM drop in one MultiRNNCell object
	multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)  #store LSTM layers in one MultiRNNCell object
	w = tf.get_variable("w", shape = [neurons_per_cell[-1], 1], initializer = tf.contrib.layers.xavier_initializer())  #create weights variable
	b = tf.get_variable("b", initializer = tf.random_uniform([1], -.1, .1))  #create linear regression variable
	return drop_multi_cell, multi_cell, w, b

def calculate_LSTM_output(neurons_per_cell, num_layers, batch_size, train_inputs, time_steps, w, b, drop_multi_cell):
	"""  Calculate output of LSTM layers
	neurons_per_cell: Number of hidden nodes in each LSTM layers
	num_layers: 	  Number of LSTM layers
	batch_size: 	  Number of data points per batch
	train_inputs: 	  List of tf placeholders for inputs
	time_steps: 	  Number of time steps into future
	w: 				  LSTM weight variable
	b: 				  Linear regression weight variable
	drop_multi_cell:  MultiRNNCell with dropout included
	"""
	cell_state = []
	hidden_state = []  
	initial_state = []
	for layer in range(num_layers):
		cell_state.append(tf.Variable(tf.zeros([batch_size, neurons_per_cell[layer]]), trainable = False))  #do not add vairable to graph collection
		hidden_state.append(tf.Variable(tf.zeros([batch_size, neurons_per_cell[layer]]), trainable = False))  #variable holding matrix filled with zeros of size batch, nodes in layer
		initial_state.append(tf.contrib.rnn.LSTMStateTuple(cell_state[layer], hidden_state[layer]))  #add LSTM initial state object to list

	all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs], axis = 0)  #tensor transformation to fit dynamic_rnn- [time_steps, batch_size, dimensionality]
	all_lstm_outputs, state_tensor = tf.nn.dynamic_rnn(  #calculate LSTM outputs
		drop_multi_cell, 
		all_inputs, 
		initial_state = tuple(initial_state), 
		time_major = True,
		dtype = tf.float32
		)

	all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*time_steps, neurons_per_cell[-1]])
	all_outputs = tf.nn.xw_plus_b(all_lstm_outputs, w, b)  #calculate all outputs
	split_outputs = tf.split(all_outputs, time_steps, axis=0)  #Split output to list of tensors of size time_steps- loss between prediction and price
	return cell_state, hidden_state, state_tensor, split_outputs

def calculate_loss_and_optimize(cell_state, hidden_state, state_tensor, num_layers, time_steps, split_outputs, train_outputs, learning_decay_rate, clipping_ratio):
	"""  Calculate total Mean Squared Error of predictions and outputs and define optimizer
	cell_state:	   		 List of tf variables storing LSTM cell state
	hidden_state:  		 List of tf variables storing LSTM hidden state
	state_tensor:  		 Tensor with size [batch_size, cell_state_size]
	num_layers:    		 Number of LSTM layers
	time_steps:	   		 Number of time steps into future
	split_outputs: 		 List of tensors of size time_steps storing output
	train_outputs: 		 List of tf placeholders for outputs
	learning_decay_rate: Rate of learning decay for optimizer
	clipping_ratio: 	 Clipping ratio for gradients
	"""
	print("Calculating Training Loss")
	loss = 0.0
	with tf.control_dependencies([tf.assign(cell_state[layer], state_tensor[layer][0]) for layer in range(num_layers)]+  #execute for loop after assigns
								 [tf.assign(hidden_state[layer], state_tensor[layer][1]) for layer in range(num_layers)]):  #fill cell and hidden state
		for step in range(time_steps):  #calculate total loss
			loss += tf.reduce_mean(.5*(split_outputs[step]-train_outputs[step])**2)

	print("Defining Learning Rate Decay")
	global_step = tf.Variable(0, trainable = False)  #total number of training steps
	inc_global_step = tf.assign(global_step, global_step + 1)
	tf_learning_rate = tf.placeholder(shape = None, dtype = tf.float32)
	tf_min_learning_rate = tf.placeholder(shape = None,dtype = tf.float32)
	learning_rate = tf.maximum(
		tf.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=learning_decay_rate, staircase=True),  #decay learning rate discrete intervals
		tf_min_learning_rate)

	print("Defining TensorFlow Optimizations")
	optimizer = tf.train.AdamOptimizer(learning_rate)  #instantiate adam optimizer
	gradients, v = zip(*optimizer.compute_gradients(loss))  #computs gradients of loss for variables- returns list of gradient variable pairs
	gradients, global_norm = tf.clip_by_global_norm(gradients, 5.0)  #clips garidents
	optimizer = optimizer.apply_gradients(zip(gradients, v))  #applies new gradients to optimizer
	return optimizer, loss, inc_global_step, tf_learning_rate, tf_min_learning_rate

def define_prediction_functions(num_layers, neurons_per_cell, dimensionality, multi_cell, w, b):
	"""  Define prediction-related tf functions
	num_layers: 	  Number of LSTM layers
	neurons_per_cell: Number of hidden nodes in each LSTM layer
	dimensionality:   Dimensionality of input data
	multi_cell: 	  LSTM layers object
	w: 				  LSTM weights variable
	b:				  Linear regression variable
	"""
	print("Defining Prediction Functions")
	sample_inputs = tf.placeholder(tf.float32, shape=[1, dimensionality])
	sample_cell_state = [] 
	sample_hidden_state = []
	initial_sample_state = []
	for layer in range(num_layers):
		sample_cell_state.append(tf.Variable(tf.zeros([1, neurons_per_cell[layer]]), trainable=False))
		sample_hidden_state.append(tf.Variable(tf.zeros([1, neurons_per_cell[layer]]), trainable=False))
		initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_cell_state[layer], sample_hidden_state[layer]))

	reset_sample_states = tf.group(*[tf.assign(sample_cell_state[layer],tf.zeros([1, neurons_per_cell[layer]])) for layer in range(num_layers)],
                               *[tf.assign(sample_hidden_state[layer],tf.zeros([1, neurons_per_cell[layer]])) for layer in range(num_layers)])

	sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs,0),
                                   initial_state=tuple(initial_sample_state),
                                   time_major = True,
                                   dtype=tf.float32)

	with tf.control_dependencies([tf.assign(sample_cell_state[layer],sample_state[layer][0]) for layer in range(num_layers)]+
								 [tf.assign(sample_hidden_state[layer],sample_state[layer][1]) for layer in range(num_layers)]):
		sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]), w, b)


	return sample_inputs, sample_prediction, reset_sample_states

def run_LSTM(train_data, batch_size, time_steps, train_inputs, train_outputs, optimizer, loss, sample_inputs, sample_prediction, all_preprocessed_data, reset_sample_states, epochs, n_predict_once, training_percentage, inc_gstep, tf_learning_rate, tf_min_learning_rate):
	"""  Train network over multiple epochs
	train_data: 		   Set of scaled, normalized, and smoothed data to train network
	batch_size: 		   Number of augmented data points per batch
	time_steps: 		   Number of time steps into future
	train_inputs: 		   TF variable containing inputs
	train_outputs: 		   TF variable containing outputs
	optimizer: 			   Network optimizer
	loss: 				   Total training loss
	sample_inputs: 		   Prediction placeholder
	sample_prediction:	   Sample prediction function
	all_preprocessed_data: All train/test data after scaling
	reset_sample_states:   Reset sample cell and hidden states to zeros
	epochs: 			   Number of times to run network
	n_predict_once:		   Number of steps to continuously predict
	training_percentage:   Percentage of data that goes into training_set
	inc_gstep:			   Increment global step
	tf_learning_rate:	   Current learning rate
	tf_min_learning_rate:  Minimum learning rate allowed
	"""
	valid_summary = 1  #Interval to make test predictions
	train_data_len = train_data.size
	train_loss_ot = []  #Store training loss over time
	test_loss_ot = []  #Store test loss over time
	predictions_over_time = []  #Store predictions
	session = tf.InteractiveSession()  #tf session for shell

	tf.global_variables_initializer().run()  #initializes tf variables
	loss_nondecrease_count = 0 	#Steps that loss hasn't decreased for learning decay
	loss_nondecrease_threshold = 2  #If test loss hasn't increased in loss_nondecrease_threshold steps, decrease learning rate
	average_loss = 0
	dgs = DataGeneratorSeq(train_data,batch_size,time_steps)  #data generation
	x_axis_sequence = []
	start = int(training_percentage*len(all_preprocessed_data))
	end = len(all_preprocessed_data)
	test_points_seq = np.arange(start,end - n_predict_once,n_predict_once).tolist()  #Start of test predictions
	print("Initialized")

	for epoch in range(epochs):

		for step in range(train_data_len//batch_size):  #for step in batch

			all_batch_data, all_label_data = dgs.output_batches()  #data augmentation
			feed_dict = {}  #dict to update tf placeholders
			for i,(data,label) in enumerate(zip(all_batch_data, all_label_data)):            
				feed_dict[train_inputs[i]] = data.reshape(-1,1)  #set up dict to set train_inputs to data
				feed_dict[train_outputs[i]] = label.reshape(-1,1) #set up dict to set train_outputs to label

			feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate:0.000001})  #add keys/values to feed_dict
			_, l = session.run([optimizer, loss], feed_dict=feed_dict) #returns one step of calculations
			average_loss += l  #currently total loss

		if (epoch + 1) % valid_summary == 0:  #Make test prediction
			average_loss = average_loss/(valid_summary*(train_data_len//batch_size))  #compute actual avg loss
			if (epoch + 1) % valid_summary == 0:
				print("Average loss at step " + str(epoch + 1) + ": " + str(average_loss))

			train_loss_ot.append(average_loss)
			average_loss = 0  #reset
			predictions_seq = []
			test_loss_seq = []

			for data_index in test_points_seq:  #for each point in test set
				test_loss = 0.0
				predictions = []

				if (epoch + 1) - valid_summary == 0:  #make prediction
					x_axis = []

				for price in range(data_index - time_steps + 1, data_index - 1):
					current_price = all_preprocessed_data[price]  #current price
					feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)  #set sample_inputs to current_price array  
					out = session.run(sample_prediction,feed_dict=feed_dict) #perform tf calculations

				feed_dict = {}  #reset feed_dict
				current_price = all_preprocessed_data[data_index - 1]
				feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)

				for pred_i in range(n_predict_once):  #steps to  continuously predict
					pred = session.run(sample_prediction,feed_dict=feed_dict) #predicted price 
					predictions.append(np.asscalar(pred))
					feed_dict[sample_inputs] = np.asarray(pred).reshape(-1,1)
					if (epoch + 1) - valid_summary == 0:  #make prediction
						x_axis.append(data_index+pred_i)  #add to list of x_axis data
					test_loss += 0.5*(pred - all_preprocessed_data[data_index+pred_i])**2  #calculate MSE for test data

				session.run(reset_sample_states)
				predictions_seq.append(np.array(predictions))  #add predictions to list of predictions
				test_loss /= n_predict_once  #average test loss per step
				test_loss_seq.append(test_loss)  #add to list of test_loss
				if (epoch + 1) - valid_summary == 0: 
					x_axis_sequence.append(x_axis)  #add x axis data to overall x axis sequence

			current_test_error = np.mean(test_loss_seq)  #current overall test error

			if len(test_loss_ot) > 0 and current_test_error > min(test_loss_ot):  #loss has not decreased
				loss_nondecrease_count += 1
			else:  #loss decreased
				loss_nondecrease_count = 0

			if loss_nondecrease_count > loss_nondecrease_threshold:  #if loss hasn't decreased in too long
				session.run(inc_gstep)  #increment global step
				loss_nondecrease_count = 0  #reset loss decrease counter
				print("\tDecreasing learning rate")

			test_loss_ot.append(current_test_error)
			print("\tTest Error: " + str(np.mean(test_loss_seq)))
			predictions_over_time.append(predictions_seq)
			print("\tFinished Predictions")
	return predictions_over_time, x_axis_sequence

def graph_results(best_epoch, all_preprocessed_data, predictions_over_time, x_axis_sequence, df, start_alpha, training_percentage):
	""" Graph model results. 
	best_epoch: 		   Epoch with best model performance
	all_preprocessed_data: Set of all preprocessed data
	predictions_over_time: List of all predictions
	x_axis_sequence: 	   List of x axis data
	df: 				   Pandas DataFrame of all data
	start_alpha: 		   High alpha = newer predictions
	training_percentage:   Percentage of data that goes into training_set
	"""
	plt.figure(figsize = (10,12))
	plt.subplot(2,1,1)
	plt.plot(range(df.shape[0]), all_preprocessed_data, color = "b")

	alpha = np.arange(start_alpha, 1.1, (1.0 - start_alpha)/len(predictions_over_time[::3]))
	for i, p in enumerate(predictions_over_time[::3]):
		for x, y in zip(x_axis_sequence, p):
			plt.plot(x, y, color = "r", alpha = alpha[i])

	plt.title("Change in Test Predictions Over Time", fontsize=14)
	plt.xlabel("Date", fontsize=14)
	plt.xlabel("Closing Price", fontsize=14)
	plt.xlim(int(training_percentage*len(all_preprocessed_data)), len(all_preprocessed_data))
	plt.subplot(2,1,2)

	plt.plot(range(df.shape[0]),all_preprocessed_data,color="b")
	for x,y in zip(x_axis_sequence,predictions_over_time[best_epoch]):
		plt.plot(x,y,color="r")

	plt.title("Best Test Prediction Over Time",fontsize=14)
	plt.xlabel("Date",fontsize=14)
	plt.ylabel("Closing Price",fontsize=14)
	plt.xlim(int(training_percentage*len(all_preprocessed_data)), len(all_preprocessed_data))
	plt.show()




def training(train_data, test_data, all_preprocessed_data, training_percentage, df):
	""" Perform model training.
	train_data: 		   Set of scaled, normalized, and smoothed data to train network
	test_data: 			   Set of scaled and normalized data to test network
	all_preprocessed_data: Set of all preprocessed data
	"""
	augment_batch_size = 5
	augment_time_steps = 5
	all_augment_batch_data, all_augment_batch_labels = data_augmentation(train_data, augment_batch_size, augment_time_steps)
	dimensionality = 1  #Dimensionality of data: 1D
	time_steps = 50  #Number of time steps into future
	batch_size = 500  #Data points in a batch
	neurons_per_cell = [200, 200, 150]  #Number of hidden nodes in each LSTM layer
	num_layers = len(neurons_per_cell)
	dropout = 0.2  #proportion of data to drop
	learning_decay_rate = 0.5
	clipping_ratio = 5.0  #gradient clipping ratio
	epochs = 30
	n_predict_once = 50
	best_epoch = 25
	start_alpha = .25
	tf.reset_default_graph()
	train_inputs, train_outputs = create_tf_input_output(batch_size, dimensionality, time_steps)  #create placeholder tf inputs and outputs
	drop_multi_cell, multi_cell, w, b = create_LSTM_layers(neurons_per_cell, num_layers, dropout)  #define LSTM layers
	cell_state, hidden_state, state_tensor, split_outputs = calculate_LSTM_output(neurons_per_cell, num_layers, batch_size, train_inputs, time_steps, w, b, drop_multi_cell)  #Calculate LSTM layer output
	optimizer, loss, inc_global_step, tf_learning_rate, tf_min_learning_rate = calculate_loss_and_optimize(cell_state, hidden_state, state_tensor, num_layers, time_steps, split_outputs, train_outputs, learning_decay_rate, clipping_ratio)
	sample_inputs, sample_prediction, reset_sample_states = define_prediction_functions(num_layers, neurons_per_cell, dimensionality, multi_cell, w, b)
	predictions_over_time, x_axis_sequence = run_LSTM(train_data, batch_size, time_steps, train_inputs, train_outputs, optimizer, loss, sample_inputs, sample_prediction, 
													  all_preprocessed_data, reset_sample_states, epochs, n_predict_once, training_percentage, inc_global_step, tf_learning_rate, tf_min_learning_rate)
	
	graph_results(best_epoch, all_preprocessed_data, predictions_over_time, x_axis_sequence, df, start_alpha, training_percentage)
def main():
	train_data, test_data, all_preprocessed_data, training_percentage, df = preprocessing()
	training(train_data, test_data, all_preprocessed_data, training_percentage, df)

main()





