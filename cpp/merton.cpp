//Including header files
#include <iostream>
#include <random>
#include <cmath>

//pybind11 to build into python module
#include <pybind11/pybind11.h>
namespace py = pybind11;

//Monte Carlo Function
double simulate (double price, double drift, double vol, double lam, double k, double sig_j, double time) {

    //Assigning price sum to 0 and a few constants to make things easier
    double price_sum = 0;
    double base_wt = std::pow(time, 0.5);
    double first_term = drift - (0.5 * std::pow(vol, 2)) - (lam*k);
    
    // The mathematical mistake lies below. I wrongly subtracted volatility and jump correction terms from mean
    // It should have been volatility
    // I take up this mistake. It's my fault lol. I'll keep this comment here so you can see how inaccurate it is
    // double first_term = mean - (std::pow(vol, 2)) - (lam*k);

    //J distribution variables
    double J_mean = std::log(1+k)-(std::pow(sig_j, 2)/ (double)2);

    //Random number engine
    std::random_device rd;
    std::mt19937 gen(rd());

    //Distributions
    std::normal_distribution<double> wiener_dist(0.0, 1.0);
    std::normal_distribution<double> J_dist(J_mean, sig_j);
    std::poisson_distribution<int> poisson(lam*time);
    
    //Monte Carlo simulations for 200,000 iterations (INDUSTRY STANDARD)
    for (int i = 0; i < 200000; i++) {
        
        //Wiener Process
        double Z = wiener_dist(gen);
        double Wt = Z * base_wt;

        //Obtaining Nt
        int N_t = poisson(gen);

        //Declaring Ji sum
        double Ji_sum = 0.000;

        //Inner loop for Ji sum
        for (int j = 1; j<= N_t; j++) {
            double J = J_dist(gen);
            Ji_sum += J;
        }

        //Declaring simulated_price
        double simulated_price;

        //Computing simulated_price
        simulated_price = price * std::exp( (first_term * time) + (vol * Wt) + Ji_sum);

        //Incrementing price_sum by simulated_price
        price_sum += simulated_price;
    }

    //Taking the average result
    double predicted_price = price_sum / (double) 200000;

    //Returning the average result as predicted_price
    return predicted_price;
}

//pybind11 declaration
PYBIND11_MODULE(merton, m) {
    m.def("simulate", &simulate);
}