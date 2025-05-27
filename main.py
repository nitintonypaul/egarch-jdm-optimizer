# Importing Dependencies
import argparse
import yfinance as yf

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

# For each stock in stocks
for i in stocks:

    # Defining ticker object
    ticker_obj = yf.Ticker(i)

    # Obtaining current price of the stock
    current_price = ticker_obj.history(period="1d")["Close"].iloc[-1]
    print(current_price)