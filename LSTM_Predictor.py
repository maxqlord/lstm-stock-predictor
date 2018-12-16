import pandas as pd
from pathlib import Path
import requests
import config 
import datetime as dt
#TODO Sanitize ticker input

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
	""" Load stock market data from Kaggle.

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
	""" Load stock market data from Alphavantage.

	ticker: Market identifier that specifies the stock to get data for. Ex. "AAL" for American Airlines
	"""
	api_key = config.api_key #Pull api_key from config.py
	filename = ticker + ".csv" #Cache requested csv files
	file_path = Path("Cache/") / filename
	if file_path.exists():  #Call cached csv
		print("Data for ticker " + ticker + " already exists. Loading Alphavantage data from CSV")
		df = pd.read_csv(file_path, delimiter=",", usecols=["Date","Low","High","Close","Open"])
		return df
	else:  #considered directly downloading csv but wanted to avoid direct download without warning
		url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + ticker + "&outputsize=full&apikey=" + api_key
		r = requests.get(url_string)
		data_json = r.json()
		if len(data_json) != 1:  #valid response
			data = data_json["Time Series (Daily)"]
			df = pd.DataFrame(columns=["Date","Low","High","Close","Open"]) 
			for date, values in data.items():
				formatted_date = dt.datetime.strptime(date, "%Y-%m-%d") #https://docs.python.org/3/library/datetime.html#datetime-objects
				row = [formatted_date.date(),float(values["3. low"]),float(values["2. high"]), 
						float(values["4. close"]),float(values["1. open"])] #new row
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


ticker = ticker_input()
source = source_input()
df = load_data(ticker, source)
print(df.head()) #Test data import