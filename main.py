# Importing Dependencies
import argparse
import yfinance as yf
import sys

# Custom modules 
from py_modules.data_handler import compute_elements
from egarch import estimate

# Argument object
parser = argparse.ArgumentParser(description="Process stocks and investments")

# Adding arguments 'stock' and 'investment'
parser.add_argument('--stock', action='append', help="Ticker Symbol", required=True)
parser.add_argument('--investment', action='append', type=float, help="Investment amount", required=True)

# Parsing arguments
try:
    args = parser.parse_args()
except:
    print("Please include the above arguments and try again.")
    sys.exit(0)

# Obtaining stocks list and the corresponding investments list
stocks = args.stock
investments = args.investment

# Checking whether each stock has an investment
# Displaying message and exiting with 0 status
if len(stocks) != len(investments):
    print("Every stock does not have a corresponding investment. Please Try again.")
    sys.exit(0)

'''
Stuff to obtain:  mean (for JDM)
Optimization data not yet identified
'''

# For each stock in stocks
for i in stocks:

    # Defining ticker object
    ticker_obj = yf.Ticker(i)

    # Obtaining current price of the stock
    current_price = ticker_obj.history(period="1d")["Close"].iloc[-1]

    # Obtaining volatility before 180d
    data = ticker_obj.history(period="211d")
    prices = data["Close"][0:30]

    # Returns array to compute shock value
    returns_array = data["Close"][30:]

    # Obtaining volatility and shock array
    vol, shock_array = compute_elements(prices, returns_array)

    # Computing expected volatility using EGARCH
    expected_volatility = estimate(len(shock_array),vol,shock_array)
    
    print(expected_volatility)
    