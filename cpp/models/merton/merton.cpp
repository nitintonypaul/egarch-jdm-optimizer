//Including header files
#include <iostream>
#include <random>
#include <cmath>

//Merton Jump Diffusion expected price estimation function
double estimate (double price, double mean, double vol, double lam, double k, double sig_j, double time, std::mt19937 &gen) {

    // J distribution variables
    double J_mean = std::log(1+k)-(std::pow(sig_j, 2)/ (double)2);

    // Initializing distributions for Brownian noise and Jump component
    // Assigning functions for the respective distributions to be used in MJD
    std::normal_distribution<double> wiener_dist(0.0, 1.0);
    std::normal_distribution<double> J_dist(J_mean, sig_j);
    std::poisson_distribution<int> poisson(lam*time);
    
    // PREPARING VARIABLES FOR MJD
    // Wiener Process
    double Z = wiener_dist(gen);
    double Wt = Z * std::pow(time, 0.5);

    // Obtaining N_t
    int N_t = poisson(gen);

    // Computing J_i sum
    double Ji_sum = 0.000;
    for (int j = 1; j <= N_t; j++) {
        double J = J_dist(gen);
        Ji_sum += J;
    }

    // MJD EQUATION
    // Computing and returning expected price using MJD equation
    // MJD EQUATION roughly is price * exp ( mean.time + BROWNIAN NOISE + JUMP COMPONENT)
    double expected_price = price * std::exp( (mean * time) + (vol * Wt) + Ji_sum);

    // Why do I even use expected_price variable when I can return it directly?
    // I guess we'll never know
    return expected_price;
}