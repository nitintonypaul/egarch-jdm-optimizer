# SCRIPT FOR DISPLAYING DATA NICELY
# Importing dependencies
import matplotlib.pyplot as plt
from tabulate import tabulate

# Function to display data
def display_data(data, header):
    print(tabulate(data, headers=header, tablefmt="grid"))

# Function to display summary
def display_summary(data):

    # Creating summary list with a single row and 2 columns
    summary_list = [[0,0]]

    # Looping and incrementing investment and returns values
    for something in data:
        summary_list[0][0] += something[1]
        summary_list[0][1] += something[4]
    
    # Indicator and displaying data
    print("SUMMARY")
    print(tabulate(summary_list, headers=["TOTAL INVESTMENT", "TOTAL RETURNS"], tablefmt="grid"))

def plot_portfolio(array, plots):
    plt.plot([x*24/(plots-1) for x in range(plots)], array.T, alpha=0.8)
    plt.title("Portfolio Wealth Over Time")
    plt.xlabel("Time in Hours")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()