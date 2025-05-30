# Importing Dependencies
import argparse
import yfinance as yf
import sys
import math
import numpy as np

# Custom modules 
from py_modules.data_handler import compute_elements, compute_mean
from build_modules.egarch import estimate
from build_modules.merton import simulate

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

# Defining time in terms of trading years
time = 1/252

# For each stock in stocks
for i in range(len(stocks)):

    # Obtaining stocks and corresponding investment
    stock = stocks[i]
    investment = investments[i]

    # Defining ticker object
    ticker_obj = yf.Ticker(stock)

    # Obtaining current price of the stock
    current_price = ticker_obj.history(period="1d")["Close"].iloc[-1]

    # Obtaining volatility before 180d
    data = ticker_obj.history(period="211d")
    prices = data["Close"][0:30]

    # Obtaining jump returns from jump data taken over a period of 1 year
    jumpdata = ticker_obj.history(period="1y")
    jump_prices = jumpdata["Close"]
    jump_returns = jump_prices.pct_change().dropna()

    # Returns array to compute shock value
    returns_array = data["Close"][30:]

    # Computing mean for JDM
    mean = compute_mean(returns_array)

    # Obtaining volatility and shock array
    vol, shock_array = compute_elements(prices, returns_array)

    # Computing expected volatility using EGARCH
    expected_volatility = estimate(len(shock_array),vol,shock_array)

    # Computing threshold for jumps
    threshold = 3 * jump_returns.std()

    # Obtaining jumps in fractions
    jumps = jump_returns[abs(jump_returns) > threshold]
    jumps = jumps.tolist()

    # Obtaining average jump (Logarithmic)
    ksum = 0
    for i in jumps:
        ksum+=math.log(1+i)
    
    # Considering case if jumps array is empty
    if len(jumps) == 0:
        k = 0
    else:
        k = ksum/len(jumps)

    # Computing jump volatility
    jump_vol_array = []
    for i in jumps:
        jump_vol_array.append(math.log(1+i))
    
    # Considering case if jump_vol_array is empty
    if len(jump_vol_array) == 0:
        sig_j = 0
    else:
        sig_j = np.std(jump_vol_array, ddof=1)

    # Computing Average jump frequency over given time
    lambda_ = len(jumps)/(1/time)

    # Annualizing mean and volatility before plugging into JDM
    mean = mean * 252
    expected_volatility = expected_volatility * math.sqrt(252)

    # Simulating prices using MERTON
    expected_price = simulate(current_price, mean, expected_volatility, lambda_, k, sig_j, time)
    
    print(f"Invested amount: {investment}")
    print(f"Current Stock price of {stock}: {current_price}")
    print(f"Expected Stock price of {stock}: {expected_price}")
    print(f"Expected return: {(investment/current_price)*expected_price}")
    print(expected_volatility)
    