# Importing Dependencies
import argparse
import yfinance as yf
import sys
import math
import numpy as np

# Custom modules 
from py_modules.data_handler import compute_elements, compute_mean
from py_modules.data_display import display_data, display_summary
from py_modules.mvo import optimize    # Mean Variance Optimization
from build_modules.egarch import estimate    # EGARCH
from build_modules.merton import simulate    # Merton Jump Diffusion Model

# Argument object
parser = argparse.ArgumentParser(description="Process stocks and investments")

# Adding arguments required, both mandate and optional
parser.add_argument('--stock', action='append', help="Ticker Symbol", required=True)
parser.add_argument('--investment', action='append', type=float, help="Investment amount", required=True)
parser.add_argument('--risk', type=float, help="Risk Aversion Factor", required=False, default=0.35)
parser.add_argument('--nsim', type=int, help="Number of Simulations", required=False, default=10)

# Parsing arguments
try:
    args = parser.parse_args()
except:
    print("Please include the above arguments and try again.")
    sys.exit(0)

# Obtaining stocks Arguments from the user
stocks = args.stock
investments = args.investment
RA = args.risk
SIMULATIONS = args.nsim

# Checking whether each stock has an investment
# Displaying message and exiting with 0 status
if len(stocks) != len(investments):
    print("Every stock does not have a corresponding investment. Please Try again.")
    sys.exit(0)

# Introduction to the program
print("----- EGARCH & JDM BASED PORTFOLIO OPTIMIZER -----\n")

# Defining time in terms of trading years
time = 1/252

# Table headers and data list
headers = ["STOCK","INVESTMENT","CURRENT PRICE", "EXPECTED PRICE (1 DAY)", "EXPECTED RETURN", "GAIN/LOSS (%)"]
datalist = []

# Some decoration
print("ESTIMATING VOLATILITY USING EGARCH...")

# For each stock in stocks
for i in range(len(stocks)):

    # Obtaining stocks and corresponding investment
    stock = stocks[i]
    investment = investments[i]

    # Defining ticker object
    ticker_obj = yf.Ticker(stock)
    
    # Obtaining jump returns from jump data taken over a period of 1 year
    jump_prices = ticker_obj.history(period="1y")["Close"]
    jump_returns = np.log(jump_prices / jump_prices.shift(1)).dropna()
    
    # Obtaining data to compute volatility, prices and current price
    data = jump_prices[-211:]
    prices = data[0:30]
    current_price = data.iloc[-1]
    returns_array = data[30:]

    # Computing mean for JDM
    mean = compute_mean(returns_array) * 252

    # Obtaining volatility and shock array
    vol, shock_array = compute_elements(prices, returns_array)

    # Computing expected volatility using EGARCH
    expected_volatility = estimate(len(shock_array), vol, shock_array) * (252**0.5)

    # print(f"Debug - {stock}: vol={vol:.8f}, expected_vol={expected_volatility:.8f}")
    # print(f"Debug - {stock}: shock_array length={len(shock_array)}")

    # Computing threshold for jumps
    threshold = 3 * jump_returns.std()

    # Obtaining jumps in fractions
    jumps = jump_returns[abs(jump_returns) > threshold].tolist()

    # Obtaining average jump (Logarithmic)
    ksum = 0
    for j in jumps:
        ksum += math.log(1+j)
    
    # Considering case if jumps array is empty
    k = 0 if len(jumps) == 0 else ksum/len(jumps)

    # Computing jump volatility
    jump_vol_array = [math.log(1+j) for j in jumps]
    
    # Number of jumps are taken as 0 when array length is less than 2
    # Standard deviation of the jumparray is not possible for certain stocks
    sig_j = 0 if len(jump_vol_array) < 2 else np.std(jump_vol_array, ddof=1)

    # Computing Average jump frequency over given time
    lambda_ = len(jumps)/ (1/time)

    # Computing drift
    # Drift is assumed to be constant 
    # Time varying drift is possible, but adds unwanted complexity and is not beneficial compared to the computing power spent
    drift = mean + (0.5 * (expected_volatility**2)) + (lambda_*k)

    # Simulating prices using MERTON
    expected_price = simulate(current_price, drift, expected_volatility, lambda_, k, sig_j, time, SIMULATIONS)
    
    # Appending data to data list
    datalist.append([stock, investment, current_price, expected_price, (investment/current_price)*expected_price, ((expected_price-current_price)/current_price)*100])
    print(f"{stock}: {expected_volatility*100:.3f}%")

# Displaying current (base) portfolio
print("\n----- CURRENT PORTFOLIO -----")
display_data(datalist, headers)
print(" ")

# Displaying Summary
display_summary(datalist)
print("\nOPTIMIZING PORTFOLIO...")

# Obtaining optimized data list using mean variance optimization
datalist = optimize(datalist, RA)

# Checking if datalist is None
# None means the portfolio cannot be optimized since all investments result in losses
if datalist == None:
    print("----- PORTFOLIO CANNOT BE OPTIMIZED. ALL INVESTMENTS ARE FOUND TO RESULT IN LOSSES -----")
    sys.exit(0)

# Displaying optimized portfolio
print("\n----- OPTIMIZED PORTFOLIO -----")
display_data(datalist, headers)
print(" ")

# Displaying Summary
display_summary(datalist)

# Demo Argument (10 stocks)
# cd documents/vscode/opti
# python main.py --stock AAPL --investment 5000 --stock TSLA --investment 3000 --stock GOOGL --investment 4000 --stock MSFT --investment 3500 --stock AMZN --investment 4500 --stock NVDA --investment 2500 --stock META --investment 2000 --stock JPM --investment 1500 --stock DIS --investment 1800 --stock NFLX --investment 2200