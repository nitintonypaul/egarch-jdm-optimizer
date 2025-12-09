//Using math defines to utilize PI
#define _USE_MATH_DEFINES

// Including modules
#include <iostream>
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

// Namespace for pybind11
namespace py = pybind11;

// Structure to carry optimizer state parameters
struct optimizerState {
    std::vector<double> parameters = {0.1, 0.9, -0.1, -0.1}; // Alpha, Beta, Omega, Gamma
    std::vector<double> gradient = {0, 0, 0, 0};
    double lambda = 1;
    int days;
    double vol;
};

// Function Declarations
double estimate(int no_days, double volatility, const std::vector<double> &shockarray);
void optimize(const std::vector<double> &shockarray, optimizerState &s);
void compute_gradient(const std::vector<double> &shockarray, optimizerState &s);
double compute_volatility(int t, const std::vector<double> &shockarray, optimizerState &s);
double backtrack_line_search(double dir[], const std::vector<double> &shockarray, optimizerState &s);
bool armijo_condition(double alpha, double dir[], const std::vector<double> &shockarray, optimizerState &s);


// Constants
const double ARMIJO_C = 1e-4;
const double LOG_CLAMP = 700.0;
const double NLL_Z_TERM_MAX = 1e9;

// Core function
double estimate(int no_days, double volatility, const std::vector<double> &shockarray) {

    // Initializing state structure
    optimizerState s;

    // Assigning variables
    s.days = no_days;
    s.vol = volatility;

    // Calling the optimizing function to find parameters
    optimize(shockarray, s);
    
    // Returning simulated volatility
    return compute_volatility(s.days, shockarray, s);
}

// Conjugate Gradient Optimization 
void optimize(const std::vector<double> &shockarray, optimizerState &s) {

    // Debugging code
    // std::cout << "ALPHA: " << lambda << " PARAMETERS: " << parameters[0] << "," <<  parameters[1] << "," <<  parameters[2] << "," <<  parameters[3] << std::endl;

    // First iteration is done manually
    // Computing initial gradient 
    compute_gradient(shockarray, s);

    // Declaring direction
    double direction[4] = {0, 0, 0, 0};
    
    // Finding direction
    // Direction is -gradient(f)
    for (int i = 0; i < 4; i++) {
        direction[i] = -s.gradient[i];
    }

    // Computing lambda (Step value)
    // Using backtrack line search
    s.lambda = backtrack_line_search(direction, shockarray, s);

    // Optimizing parameters using step value
    // Optimization formula: P = P + (λ * D)
    for (int i = 0; i < 4; i++) {
        s.parameters[i] = s.parameters[i] + (s.lambda * direction[i]);
    }

    // Debugging code
    // std::cout << "ALPHA: " << lambda << " PARAMETERS: " << parameters[0] << "," <<  parameters[1] << "," <<  parameters[2] << "," <<  parameters[3] << std::endl;

    // Rest of the optimization is done in a loop
    // Stable limit of 1000 iterations
    for (int i = 0; i < 1000; i++) {

        // FLETCHER REEVES
        // Fletcher Reeves involves in updating beta after each iteration with correspondence to
        // the gradient of the current and previous function values
        // Declaring beta as the denominator product and computing
        double beta = 0;
        for (int j = 0; j < 4; j++) {
            beta += s.gradient[j] * s.gradient[j];
        }

        // Declaring numerator product
        // betaplus is a temporary variable named for numerator
        double betaplus = 0;

        // Computing next gradient
        compute_gradient(shockarray, s);
        
        // Computing numerator product
        for (int j = 0; j < 4; j++) {
                betaplus += s.gradient[j] * s.gradient[j];
        }
        
        // Computing beta_k (beta)
        // beta is used to update direction
        beta = betaplus / beta;

        // Updating direction using Fletcher Reeves
        for(int j = 0; j < 4; j++) {
            direction[j] = -s.gradient[j] + (beta * direction[j]);
        }

        // DOT PRODUCT DIAGONOSTIC
        // When the dot product of the resulting direction and gradient is significant enough, it can lead to parameter value explosions
        // Fallback is set as -gradient
        // Computing dot product (Direction, Gradient)
        double dot = 0;
         for (int j = 0; j < 4; j++) {
            dot += s.gradient[j] * direction[j];
        }

        // Checking if the dot product is significant
        // if it is, it is reset normally
        // This is a diagnostic fix and is not intended to reflect theoretical accuracy
        if (dot >= -1e-12) { 
            for (int j = 0; j < 4; j++) {
                direction[j] = -s.gradient[j];
            }
        }

        // Computing lambda (step value)
        // Using backtrack line search
        s.lambda = backtrack_line_search(direction, shockarray, s);

        // Convergence condition
        // If the step value is insignificant, then there is no significant change to the parameters
        // Hence the loop is broken to retain efficiency
        // Regardless the loop executes at least 10 iterations to prevent premature convergence
        if (s.lambda < 1e-8 && i > 10) {
            // Debugging code
            //std::cout << "BREAKING SINCE STEP IS " << lambda << std::endl;
            break;
        }

        // Optimizing parameters
        for (int j = 0; j < 4; j++) {
            s.parameters[j] = s.parameters[j] + (s.lambda * direction[j]);
        }

        // Debugging code
        //std::cout << "ALPHA: " << lambda << " PARAMETERS: " << parameters[0] << "," <<  parameters[1] << "," <<  parameters[2] << "," <<  parameters[3] << std::endl;
        // Repeat until 1000. Or until convergence happens.
    }

}

// Function to compute gradient based on current parameter values
void compute_gradient(const std::vector<double> &shockarray, optimizerState &s) {

    // Resetting gradient
    s.gradient[0] = 0;
    s.gradient[1] = 0;
    s.gradient[2] = 0;
    s.gradient[3] = 0;

    // Declaring PSI
    double PSI[4] = {0, 0, 0, 0};

    // Parent Summation loop
    for (int i = 1; i < s.days; i++) {
        
        // Caching constants
        // Computing SIGMA
        // SIGMA is clamped to 1e-9 to prevent NaN values
        double sigma_t = compute_volatility(i, shockarray, s);
        double sigma_prev = compute_volatility(i-1, shockarray, s);
        if (std::abs(sigma_t) < 1e-9) sigma_t = 1e-9;
        if (std::abs(sigma_prev) < 1e-9) sigma_prev = 1e-9;

        // Computing Z_t
        double z_t = shockarray[i] / sigma_t;
        double z_prev = shockarray[i-1] / sigma_prev;

        // This section was done to clamp values of Z, but was later deemed unnecessary
        // Z values are clamped in the Armijo function, which has not hit the limit during testing
        //z_t = std::max(-NLL_Z_TERM_MAX, std::min(z_t, NLL_Z_TERM_MAX));
        //z_prev = std::max(-NLL_Z_TERM_MAX, std::min(z_prev, NLL_Z_TERM_MAX));

        // C_T is the vector corresponding to each parameter
        // Order: Alpha, Beta, Omega, Gamma
        double CT[] = {std::abs(z_prev), std::log(std::pow(sigma_prev,2)), 1, z_prev};

        // Assigning derivative to immediate derivative
        // Immediate derivative is common for every parameter
        double alpha_der = 0.5 * (1 - std::pow(z_t,2));
        double beta_der = 0.5 * (1 - std::pow(z_t,2));
        double omega_der = 0.5 * (1 - std::pow(z_t,2));
        double gamma_der = 0.5 * (1 - std::pow(z_t,2));

        // Updating PSI with the previous value
        // PSI is the recursive variable
        for(int j = 0; j < 4; j++) {
            PSI[j] = CT[j] + (s.parameters[1] * PSI[j]);
        }

        // Multiplying with parameter derivatives and adding to the gradient
        // gradient is accessed globally
        s.gradient[0] += alpha_der * PSI[0];
        s.gradient[1] += beta_der * PSI[1];
        s.gradient[2] += omega_der * PSI[2];
        s.gradient[3] += gamma_der * PSI[3];
    }
}

// Function to compute volatility (sigma_t) under current parameters
double compute_volatility(int t, const std::vector<double> &shockarray, optimizerState &s) {
    
    // Declaring sig (volatility)
    double sig = s.vol;

    // Checking if t is 0, then sig is returned
    if (t == 0) return sig;

    // Computing volatility under current parameters
    for (int i = 1; i <= t; i++) {

        // Clamping sig to 1e-9 to prevent NaN values
        if (std::abs(sig) < 1e-9) sig = 1e-9;

        // EGARCH equation to compute logarithmic volatility squared
        double logsigsq = s.parameters[2] + (s.parameters[1] * std::log(std::pow(sig, 2))) + (s.parameters[0] * std::abs(shockarray[i-1] / sig)) + (s.parameters[3] * (shockarray[i-1] / sig));

        // Ensuring logsigsq stays in range
        // Clamping logsigsq to global constant of (-700,700) depending upon it's magnitude
        logsigsq = std::max(-LOG_CLAMP, std::min(logsigsq, LOG_CLAMP));

        // Assigning volatility to setup for next iteration
        // The previous sigma value is used in the next equation
        // EGARCH equation is essentially a recurrence relation
        sig = std::sqrt(std::exp(logsigsq));
    }

    // Returning final volatility computed under current parameters
    return sig;
}

// Function for backtrack line searching (Armijo Condition)
// Used to compute the step value required for each independent iteration
double backtrack_line_search(double dir[], const std::vector<double> &shockarray, optimizerState &s) {

    // Declaring step value (ALPHA) and beta (step multilplier)
    // Beta acts as a step value for the step value
    double beta = 0.5;
    double alpha = 1;

    // While condition is false
    while (!armijo_condition(alpha, dir, shockarray, s)) {

        // ALPHA is multiplied with beta
        // ALPHA is essentially halved every step
        alpha *= beta;
    }

    // returning the computed step value
    // referenced as lambda outside this function
    return alpha;
}

// Function to check whether Armijo conditions retains false
// line search is to break when armijo condition is true
// ARMIJO CONDITON: f(x + ALPHA.d) <= f(x) + c.ALPHA.T(grad(f(x))).d
bool armijo_condition(double alpha, double dir[], const std::vector<double> &shockarray, optimizerState &s) {

    // COMPUTING LHS 
    // LHS = f(x + ALPHA.d)
    // f -> Negative Log Likelihood function
    
    // Declaring sig (initial volatility)
    double sig = s.vol;
    double sigN = s.vol;

    // Declaring total likelihood
    // Which is the LHS in the Armijo Condition ( with modified parameters)
    double LHS = 0;

    // Declaring log likelihood and computing the same to reduce complexity
    // this total_likelihood is used for RHS computation
    double total_likelihood = 0;

    // Computing volatility under current parameters
    for (int i = 1; i < s.days; i++) {

        // Clamping sig to 1e-9 to prevent NaN values
        if (std::abs(sig) < 1e-9) sig = 1e-9;

        // EGARCH equation to compute logarithmic volatility squared
        // Modified with different parameters
        // logsiqsqN is done with normal parameters to compute f(x) for RHS
        double logsigsq = (s.parameters[2] + dir[2]*alpha) + ((s.parameters[1] + dir[1]*alpha) * std::log(std::pow(sig, 2))) + ((s.parameters[0] + dir[0]*alpha) * std::abs(shockarray[i-1] / sig)) + ((s.parameters[3] + dir[3]*alpha) * (shockarray[i-1] / sig));
        double logsigsqN = s.parameters[2] + (s.parameters[1] * std::log(std::pow(sigN, 2))) + (s.parameters[0] * std::abs(shockarray[i-1] / sigN)) + (s.parameters[3] * (shockarray[i-1] / sigN));

        // Ensuring logsigsq stays in range
        // Clamping logsigsq to global constant of (-700,700) depending upon it's magnitude
        logsigsq = std::max(-LOG_CLAMP, std::min(logsigsq, LOG_CLAMP));
        logsigsqN = std::max(-LOG_CLAMP, std::min(logsigsqN, LOG_CLAMP));

        // Assigning volatility to setup for next iteration
        // The previous sigma value is used in the next equation
        // EGARCH equation is essentially a recurrence relation
        sig = std::sqrt(std::exp(logsigsq));
        sigN = std::sqrt(std::exp(logsigsqN));

        // Computing Z (LHS & RHS)
        // It is done so to clamp values and to not let it go astronomically high or low
        double Z_LHS = std::pow(shockarray[i] / sig, 2);
        double Z_RHS = std::pow(shockarray[i] / sigN, 2);
        //std::cout << shockarray[i] << std::endl;

        // Capping both Zs to a constant
        // Haven't reached the limit in testing but deploying just in case
        // Constant is very high
        if (Z_LHS > NLL_Z_TERM_MAX) Z_LHS = NLL_Z_TERM_MAX;
        if (Z_RHS > NLL_Z_TERM_MAX) Z_RHS = NLL_Z_TERM_MAX; 

        // Incrementing to negative log likelihood (LHS)
        // Capped Z terms are used here
        LHS += 0.5 * (std::log(2 * M_PI) + logsigsq + Z_LHS);
        total_likelihood += 0.5 * (std::log(2 * M_PI) + logsigsqN + Z_RHS);
    }

    // COMPUTING RHS
    // RHS = f(x) + c.ALPHA.T(gradient(f(x))).d
    // f -> negative log likelihood function
    double RHS = total_likelihood;
    
    // Defining product variable and a temporary sum variable
    double product = ARMIJO_C * alpha;
    double tempsum = 0;

    // Computing gradient.direction
    // Dot product of two vectors
    // gradient is accessed globally
    for (int i = 0; i < 4; i++) {
        tempsum += s.gradient[i]*dir[i];
    }

    // Multiplying into product
    product *= tempsum;

    // Adding product to RHS
    RHS += product;

    // Returning the Armijo Condition status
    if (LHS <= RHS) {
        return true;
    }
    return false;

}

//Pybind11 declaration
PYBIND11_MODULE(egarch, m) {
    m.def("estimate", &estimate);
}