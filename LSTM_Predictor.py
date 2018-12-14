import pandas as pd
import os 
#TODO Check out pathlib package to replace os.path.join
#TODO Add Alphavantage data source
#TODO Add error handling to load_data() once input is created

def ticker_input(message = "Enter the ticker of the stock you want to examine: "):
	""" Ask the user for the ticker they want to examine and return the chosen ticker

	message: Message to display to the user when asking for a ticker input
	"""
	ticker = input(message)
	return ticker

def load_data(ticker, data_source="kaggle"):
	""" Load stock market data from designated data source.
	Return a pandas DataFrame containing the data sorted by date.

	ticker:      Market identifier that specifies the stock to get data for. Ex. "AAL" for American Airlines
	data_source: Designates which online resource to get data from, Kaggle or Alphavantage
	"""

	if data_source == "kaggle":  #Load directly into DataFrame from csv file
		file_path = os.path.join("Stocks",ticker + ".us.txt")
		if os.path.isfile(file_path):
			df = pd.read_csv(file_path, delimiter=",", usecols=["Date","Open","High","Low","Close"]) #ignore volume, openInt columns
			print("Loaded data for ticker " + ticker + " from Kaggle")
			return df.sort_values("Date")
		else:
			return load_data(ticker_input("Invalid ticker, please try again: ")) #Set new input to load_data's ticker parameter

ticker = ticker_input()
df = load_data(ticker)
print(df.head()) #Test data import