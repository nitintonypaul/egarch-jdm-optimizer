# Importing Dependencies
import argparse
import yfinance as yf
from py_modules.data_handler import compute_elements

# Argument object
parser = argparse.ArgumentParser(description="Process stocks and investments")

# Adding arguments 'stock' and 'investment'
parser.add_argument('--stock', action='append', help="Ticker Symbol", required=True)
parser.add_argument('--investment', action='append', type=float, help="Investment amount", required=True)

# Parsing arguments
args = parser.parse_args()

# Obtaining stocks list and the corresponding investments list
stocks = args.stock
investments = args.investment

'''
Stuff to obtain include and mean (for EGARCH and JDM)
Optimization data not yet identified
'''

# For each stock in stocks
for i in stocks:

    # Defining ticker object
    ticker_obj = yf.Ticker(i)

    # Obtaining current price of the stock
    current_price = ticker_obj.history(period="1d")["Close"].iloc[-1]

    # Obtaining volatility before 180d
    data = ticker_obj.history(period="210d")
    prices = data["Close"][0:30]

    # Returns array to compute shock value
    returns_array = data["Close"][30:]

    # Obtaining volatility and shock array
    vol, shock_array = compute_elements(prices, returns_array)
    
    # Computing expected volatility using EGARCH
    # Soon
    print(len(shock_array))
    print(vol)