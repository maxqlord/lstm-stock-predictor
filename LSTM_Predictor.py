import pandas as pd
from pathlib import Path
import requests
import config 
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
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

def main():
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

main()





