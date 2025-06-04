# Importing Dependencies
import argparse
import yfinance as yf
import sys
import math
import numpy as np

# Custom modules 
from py_modules.data_handler import compute_elements, compute_mean
from py_modules.data_display import display_data, display_summary
from py_modules.mvo import optimize
from build_modules.egarch import estimate
from build_modules.merton import simulate

# Decorative function
def decor():
    print("==============================================================")

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

# Introduction to the program
decor()
print("EGARCH & JDM BASED PORTFOLIO OPTIMIZER (MVO)")
decor()
print(" ")
print(" ")

# Disclaimer
decor()
print("DISCLAIMER")
decor()
print("This program is for demonstration purposes only and should not \nbe used for financial or investment decisions. The creator \nis not responsible for any outcomes or losses resulting from \nits use.")
print(" ")

# Defining time in terms of trading years
time = 1/252

# Table headers and data list
headers = ["STOCK","INVESTMENT","CURRENT PRICE", "EXPECTED PRICE (1 DAY)", "EXPECTED RETURN", "GAIN/LOSS (%)"]
datalist = []

# Some decoration
decor()
print("EGARCH VOLATILITY ESTIMATES")
decor()

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
        ksum += math.log(1+i)
    
    # Considering case if jumps array is empty
    k = 0 if len(jumps) == 0 else ksum/len(jumps)

    # Computing jump volatility
    jump_vol_array = [math.log(1+i) for i in jumps]
    
    # Considering case if jump_vol_array is empty
    sig_j = 0 if len(jump_vol_array) == 0 else np.std(jump_vol_array, ddof=1)

    # Computing Average jump frequency over given time
    lambda_ = len(jumps)/(1/time)

    # Annualizing mean and volatility before plugging into JDM
    mean = mean * 252
    expected_volatility = expected_volatility * math.sqrt(252)

    # Simulating prices using MERTON
    expected_price = simulate(current_price, mean, expected_volatility, lambda_, k, sig_j, time)
    
    # Appending data to data list
    datalist.append([stock, investment, current_price, expected_price, (investment/current_price)*expected_price, ((expected_price-current_price)/current_price)*100])
    print(f"{stock}: {expected_volatility}")

# Some decoration
print(" ")
decor()
print("CURRENT PORTFOLIO")
decor()

# Displaying table using tabulate
display_data(datalist, headers)

# For readability
print(" ")

# Displaying Summary table
display_summary(datalist)

# Obtaining optimized data list (MVO)
datalist = optimize(datalist)

# Some decoration
print(" ")
decor()
print("OPTIMIZED PORTFOLIO")
decor()

# Displaying optimized portfolio
display_data(datalist, headers)

# For readability
print(" ")

# Displaying Summary table
display_summary(datalist)

#Demo Argument (20 stocks)
#python main.py --stock AAPL --investment 5000 --stock TSLA --investment 3000 --stock GOOGL --investment 4000 --stock MSFT --investment 3500 --stock AMZN --investment 4500 --stock NVDA --investment 2500 --stock META --investment 2000 --stock JPM --investment 1500 --stock DIS --investment 1800 --stock NFLX --investment 2200 --stock BABA --investment 1700 --stock AMD --investment 1900 --stock ORCL --investment 2100 --stock INTC --investment 1600 --stock IBM --investment 1400 --stock PYPL --investment 2300 --stock BA --investment 2400 --stock T --investment 1300 --stock KO --investment 2600 --stock PEP --investment 2700