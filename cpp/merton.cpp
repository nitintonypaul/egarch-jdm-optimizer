// Including header files
#include <iostream>
#include <random>
#include <cmath>

// pybind11 to build into python module
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Monte Carlo Function
double simulate (double price, double mean, double vol, double lam, double k, double sig_j, double time, int M) {

    // Debugging code
    // std::cout << "{Debug} Running " << M << " Simulations" << std::endl;

    // Assigning price sum to 0 and a few constants to make things easier
    double price_sum = 0;
    double base_wt = std::pow(time, 0.5);

    // J distribution variables
    double J_mean = std::log(1+k)-(std::pow(sig_j, 2)/ (double)2);

    // Random number engine
    std::random_device rd;
    std::mt19937 gen(rd());

    // Distributions
    std::normal_distribution<double> standard_normal(0.0, 1.0);
    std::normal_distribution<double> J_dist(J_mean, sig_j);
    std::poisson_distribution<int> poisson(lam*time);
    
    // Monte Carlo simulation for M simulations 
    // M -> inf collapses to expectation, which can be computed analytically
    for (int i = 0; i < M; i++) {
        
        // Wiener Process
        double Z = standard_normal(gen);
        double Wt = Z * base_wt;

        // Obtaining number of jumps
        int N_t = poisson(gen);

        // Declaring Ji sum
        double Ji_sum = 0.00;

        // Inner loop for Ji sum
        for (int j = 1; j<= N_t; j++) {
            double J = J_dist(gen);
            Ji_sum += J;
        }

        // Computing simulated_price and incrementing into running sum
        double simulated_price = price * std::exp( (mean * time) + (vol * Wt) + Ji_sum);;
        price_sum += simulated_price;
    }

    // Computing expectation
    double expected_price = price_sum / (double) M;
    return expected_price;
}

//pybind11 declaration
PYBIND11_MODULE(merton, m) {
    m.def("simulate", &simulate);
}