//Including header files
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include "models/merton/merton.hpp"

//pybind11 to build into python module
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// Forecasting function
// Function decides which model to use for forecasting depening on the user decision
std::vector<std::vector<double>> forecast(
    double price=0, 
    double mean=0, 
    double volatility=0, 
    double lam=0, 
    double kappa=0, 
    double sig_j=0, 
    double time=0, 
    int SIMULATIONS=0
) {

    // Setting function as merton
    double (*function)(double, double, double, double, double, double, double, std::mt19937&) = estimate;
    
    // Declaring price path array and random number engine
    // paths vector contains "SIMULATIONS" price paths
    std::vector<std::vector<double>> paths(SIMULATIONS);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Monte carlo simulation calling required function
    for (int i = 0; i < SIMULATIONS; i++) {

        // Declaring prices array to store forecasted prices
        std::vector<double> pricePath (100);

        // Declaring first price as the current price
        pricePath[0] = price;

        // Recursive variable and small time dt for compounding
        double dt = time / 100;
        double previousPrice = price;

        // Price path loop
        for (int j = 1; j < 100; j++) {
            // Obtaining price path and assigning it previousPrice
            // It is also stored in the price array
            previousPrice = function(previousPrice, mean, volatility, lam, kappa, sig_j, dt, gen);
            pricePath[j] = previousPrice;
        }

        // storing in price path array
        paths[i] = pricePath;

    }

    // Returning price paths for Asset
    return paths;
}

//pybind11 declaration
PYBIND11_MODULE(jump_engine, m) {
    m.def("forecast", &forecast,
          "Monte Carlo Simulation to simulate price paths.");
}