# Importing Dependencies
import argparse
import yfinance as yf
import sys
import numpy as np

# Custom modules 
from py_modules.data_handler import compute_EGARCH_elements, compute_jump_elements
from py_modules.data_display import display_data, display_summary
from py_modules.mvo import optimize   
from build_modules.egarch import estimate    
from build_modules.merton import simulate

# Argument object
parser = argparse.ArgumentParser(
    description="""
EGARCH-JDM PORTFOLIO OPTIMIZER

INPUT FORMAT:
  - US Stocks (NASDAQ/NYSE):  AAPL, MSFT, NVDA
  - Indian Stocks (NSE):      RELIANCE.NS, INFY.NS

USAGE TIPS:
  - Set `--risk` to reflect your personal risk appetite (DEFAULTS TO 0.35)
  - Use `--nsim` to set the number of Monte Carlo simulations (DEFAULTS TO 10)
  - Use `--model` to choose between available models

AVAILABLE MODELS:
  - MERTON : Jump Diffusion with normally distributed jumps
  - GBM    : Basic Geometric Brownian Motion
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="Example: python main.py --stock AAPL --investment 1000 --stock NVDA --investment 1200 --risk 0.4 --nsim 100 --model merton"
)

# Adding and parsing arguments
parser.add_argument('--stock', action='append', help="Ticker Symbol (Stock)", required=True)
parser.add_argument('--investment', action='append', type=float, help="Investment amount", required=True)
parser.add_argument('--risk', type=float, help="Risk Aversion Factor", required=False, default=0.35)
parser.add_argument('--nsim', type=int, help="Number of Simulations", required=False, default=10)
parser.add_argument('--model', type=str, help="Model Specification", required=False, default="merton")
args = parser.parse_args()

# Obtaining stocks Arguments from the user
stocks = args.stock
investments = args.investment
RA = args.risk
SIMULATIONS = args.nsim
MODEL = args.model.lower()

# Checking validity of model given
if MODEL not in ["gbm", "merton"]:
    print("INVALID MODEL. PLEASE TRY AGAIN.")
    sys.exit(0)

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
egarch_volatility_array = []

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

    # Obtaining volatility elements and jump elements for MJD and EGARCH
    vol, shock_array, mean = compute_EGARCH_elements(prices, returns_array)
    k, lambda_, sig_j = compute_jump_elements(jump_returns, time)

    # Computing expected volatility using EGARCH and storing
    # Volatility is annualized for MJD forecasts
    expected_volatility = estimate(len(shock_array), vol, shock_array)
    egarch_volatility_array.append(expected_volatility)

    # Changing jump arrival rate for GBM (λ = 0)
    if MODEL == "gbm":
        lambda_ = 0
    
    # Simulating prices using MJD
    expected_price = simulate(current_price, mean, expected_volatility * (252**0.5), lambda_, k, sig_j, time, SIMULATIONS)
    
    # Appending to data list
    datalist.append([stock, investment, current_price, expected_price, (investment/current_price)*expected_price, ((expected_price-current_price)/current_price)*100])
    print(f"{stock}: {expected_volatility*100:.3f}%")

# Displaying current (base) portfolio
print(f"\n----- CURRENT PORTFOLIO USING {MODEL.upper()} -----")
display_data(datalist, headers)
print(" ")

# Displaying Summary
display_summary(datalist)
print("\nOPTIMIZING PORTFOLIO...")

# Obtaining optimized data list using mean variance optimization
datalist = optimize(datalist, RA, egarch_volatility_array)

# Checking if datalist is None
if datalist == None:
    print("----- PORTFOLIO CANNOT BE OPTIMIZED. ALL INVESTMENTS ARE FOUND TO RESULT IN LOSSES -----")
    sys.exit(0)

# Displaying optimized portfolio
print("\n----- OPTIMIZED PORTFOLIO -----")
display_data(datalist, headers)
print(" ")

# Displaying Summary
display_summary(datalist)