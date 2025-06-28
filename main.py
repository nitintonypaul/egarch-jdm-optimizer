# Importing Dependencies
import argparse
import yfinance as yf
import sys
import numpy as np

# Custom modules 
from py_modules.data_handler import compute_EGARCH_elements, compute_jump_elements
from py_modules.data_display import display_data, display_summary
from py_modules.mvo import optimize    # Mean Variance Optimization
from build_modules.egarch import estimate    # EGARCH
from build_modules.merton import simulate    # Merton Jump Diffusion Model

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
headers = ["STOCK","INVESTMENT","CURRENT PRICE", f"EXPECTED PRICE ({int(time*252)} DAY)", "EXPECTED RETURN", "GAIN/LOSS (%)"]
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

    # Obtaining volatility elements and jump elements for MJD and EGARCH
    vol, shock_array, mean = compute_EGARCH_elements(prices, returns_array)
    k, lambda_, sig_j = compute_jump_elements(jump_returns, time)

    # Computing expected volatility using EGARCH
    # Computed volatility is annualized for MJD forecasts
    expected_volatility = estimate(len(shock_array), vol, shock_array) * (252**0.5)

    # Computing drift
    # Drift is assumed to be constant 
    # Time varying drift is possible, but adds unwanted complexity and is not beneficial compared to the computing power spent
    drift = mean + (0.5 * (expected_volatility**2)) + (lambda_*k)

    # Changing jump value for GBM
    # Will be optimized later for different models
    # Jump component need not be computed for GBM model
    if MODEL == "gbm":
        lambda_ = 0
    
    # Simulating prices using MERTON
    expected_price = simulate(current_price, drift, expected_volatility, lambda_, k, sig_j, time, SIMULATIONS)
    
    # Appending data to data list
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