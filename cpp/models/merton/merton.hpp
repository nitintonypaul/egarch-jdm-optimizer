#pragma once

#include <random>
double estimate(double price, double mean, double vol, double lam, double k, double sig_j, double time, std::mt19937 &gen);