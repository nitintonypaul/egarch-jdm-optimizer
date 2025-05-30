// Including modules
#include <iostream>
#include <cmath>
#include <vector>
/*#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

// Namespace for pybind11
namespace py = pybind11;*/

// Function Declarations
double estimate(int no_days, double volatility, double shockarray[]);
void optimize(double shockarray[]);
void compute_gradient(double shockarray[]);
double compute_volatility(int t, double shockarray[]);

// Parameters
// Order: Alpha, Beta, Omega, Gamma
double parameters[] = {0.1,0.9,-0.1,-0.1}; 

// Gradient array
double gradient[4] = {0, 0, 0, 0};

// Initialising global variables
double lambda = 1e-9;
int days;
double vol;
const double LOG_CLAMP = 700.0;

double estimate( int no_days, double volatility, double shockarray[]) {

    // Assigning global variables
    days = no_days;
    vol = volatility;

    // Calling the optimizing function to find parameters
    optimize(shockarray);

    // Returning computed volatility
    return compute_volatility(days, shockarray);
}

// Conjugate Gradient Optimization 
void optimize(double shockarray[]) {

    // Computing initial gradient 
    compute_gradient(shockarray);

    // Finding direction
    double direction[4];
    
    // Loop to iterate through direction and assign
    for (int i = 0; i < 4; i++) {
        direction[i] = -gradient[i];
    }

    // Optimizing parameters
    for (int i = 0; i < 4; i++) {
        parameters[i] = parameters[i] + (lambda * direction[i]);
    }

    // Optimization set to 1000 iterations as limit
    for (int i = 0; i < 1000; i++) {

        // FLETCHER REEVES
        // Declaring beta as the denominator product and computing
        double beta = 0;
        for (int j = 0; j < 4; j++) {
            beta += gradient[j] * gradient[j];
        }

        // Declaring numerator product
        double betaplus = 0;

        // Computing next gradient
        compute_gradient(shockarray);
        
        // Computing numerator product
        for (int j = 0; j < 4; j++) {
                betaplus += gradient[j] * gradient[j];
        }
        
        // Computing beta_k (beta)
        // Used for updating direction
        beta = betaplus / beta;

        // Updating direction
        for(int j = 0; j < 4; j++) {
            direction[j] = -gradient[j] + (beta * direction[j]);
        }

        // Pre-checking beta to see if it's going off-rail
        // In EGARCH, Beta must always be <= 1
        if (parameters[1] + (lambda * direction[1]) > 1) {
            break;
        }

        // Updating parameters
        for (int j = 0; j < 4; j++) {
            parameters[j] = parameters[j] + (lambda * direction[j]);
        }

        // Repeat until 1000. Can implement break-out conditions.
    }

}

// Function to compute gradient based on current parameter values
void compute_gradient(double shockarray[]) {

    // Resetting gradient
    gradient[0] = 0;
    gradient[1] = 0;
    gradient[2] = 0;
    gradient[3] = 0;

    // Declaring PSI
    double PSI[4] = {0,0,0,0};

    // Parent Summation loop
    for (int i = 1; i < days; i++) {
        
        // Caching constants
        // Computing sigma and checking for near zero cases
        double sigma_t = compute_volatility(i, shockarray);
        double sigma_prev = compute_volatility(i-1, shockarray);
        if (std::abs(sigma_t) < 1e-9) sigma_t = 1e-9;
        if (std::abs(sigma_prev) < 1e-9) sigma_prev = 1e-9;

        // Computing Z_t
        double z_t = shockarray[i] / sigma_t;
        double z_t_prev = shockarray[i-1] / sigma_prev;

        // Other parameters
        double M_t = -0.5 * ((parameters[0] * std::abs(z_t_prev)) + parameters[3] * (z_t_prev));
        double PSI_multiple = parameters[1] + M_t;
        double CT[] = {std::abs(z_t_prev), std::log(std::pow(sigma_prev,2)), 1, z_t_prev};

        // Adding immediate derivative
        double alpha_der = 0.5 * (1 - std::pow(z_t,2));
        double beta_der = 0.5 * (1 - std::pow(z_t,2));
        double omega_der = 0.5 * (1 - std::pow(z_t,2));
        double gamma_der = 0.5 * (1 - std::pow(z_t,2));

        // Updating PSI with the previous value
        for(int j = 0; j < 4; j++) {
            PSI[j] = CT[j] + (PSI_multiple * PSI[j]);
        }

        // Multiplying with parameter derivatives and adding to the gradient
        gradient[0] += alpha_der * PSI[0];
        gradient[1] += beta_der * PSI[1];
        gradient[2] += omega_der * PSI[2];
        gradient[3] += gamma_der * PSI[3];
    }
}

// Function to compute volatility (sigma_t) under current parameters
double compute_volatility(int t, double shockarray[]) {
    
    // Declaring sig (volatility)
    double sig = vol;

    // Checking if t is 0, then sig is returned
    if (t == 0) return sig;

    // Computing volatility under current parameters
    for (int i = 1; i <= t; i++) {

        // Ensuring sig is in range
        if (std::abs(sig) < 1e-9) sig = 1e-9;

        // EGARCH equation to compute logarithmic volatility squared
        double logsigsq = parameters[2] + (parameters[1] * std::log(std::pow(sig, 2))) + (parameters[0] * std::abs(shockarray[i-1] / sig)) + (parameters[3] * (shockarray[i-1] / sig));

        //Ensuring logsigsq stays in range
        logsigsq = std::max(-LOG_CLAMP, std::min(logsigsq, LOG_CLAMP));

        // Assigning volatility to setup for next iteration
        sig = std::sqrt(std::exp(logsigsq));
    }

    // returning final volatility
    return sig;
}

int main() {

    int days = 180;
    double vol = 0.0493214210047689;
    double shock_array[] = {0.018463812075787605, 0.0816180911308656, 0.022250794684010738, 0.0029345366193654038, -0.01640125254174894, -0.006971943067780907, -0.01612224532927674, 0.04219590338548338, -0.01272172343230254, 0.005509315390236693, 0.0421568728865089, 0.024876966996268308, 0.0075523647622918345, -0.018242794627747094, 0.00359976405169561, -0.033975980864226485, 0.018958640872043058, 0.03637232098047601, 0.019979803465607116, 0.025437256187179246, 0.042951731124987715, 0.001462769045949902, 0.019422694522652488, 0.0031962417803506867, 0.027239081089598955, -0.04472341759615024, 0.03409721291227231, 0.012146322594607085, 0.01105420641701551, 0.04381415742290549, 0.002435036282748144, -0.025197048520631284, 0.00934253114174655, 0.01128595956206113, -0.003962144888187403, 0.00845198192265429, -0.010344027252428036, -0.04510355124708672, 0.0229607437999277, 0.008059549677436597, 0.031247257119981595, 0.04320273302863684, 0.025479373583842023, -0.005161195623356821, -0.012913474555815553, 0.023914852964071056, -0.010445150203745457, 0.006614715695611053, -0.02984193286469375, -0.009702576405183853, 0.05105777019470136, -0.0043772512120634054, 0.008602661249924143, -0.02943997801703111, -0.039402450188137345, 0.009865283734219379, -0.008336375840778827, 0.024544006185036403, 0.006015192997845647, 0.014959778831694732, 0.03747131925563825, 0.0027879619580693142, -0.014956247549329114, -0.02254439540002993, -0.024042412344864733, 0.03417877653873128, -0.010971663987707523, -0.019485406792826912, -0.013631481204766092, -0.009001539567336365, -0.00814503641741967, 0.01690737129926386, 0.03356895151681649, 0.03950281328373513, 0.007200570816451922, 0.001200040443104478, -0.01781804113612788, 0.006767733773984279, -0.02027916219882764, 0.03276624224716587, 0.046844766442642446, 0.0370265090719797, -0.06091569557514737, 0.003056397550088077, -0.02716454333999087, -0.016645625989633154, -0.007824471321632663, 0.036706426346032626, -0.01652195292236939, 0.03379472241777015, 0.02567380174013886, 0.04662566982484018, 0.0042897454012684145, -0.028474040882074343, -0.18267570103221453, 0.08876779354629183, -0.03860532074727339, 0.010920928044879473, -0.034164475250558726, -0.025540762807830266, 0.02018463615930031, 0.05404519316729707, 0.03364617873634411, 0.012244643186479925, 0.031593295424048685, -0.002511044326078823, -0.009308415660658357, 0.03442551698715665, 0.02924401800181876, 0.007223625539078644, 0.0020501482035802303, 0.00957104867922682, -0.03811390003106363, -0.028087198410766505, -0.025146149241824962, 0.03933340701534612, -0.08532107622352343, 0.042202905668971204, -0.08767851423848161, 0.02004976945135601, 0.014501200069556977, -0.05581559447878779, 0.022262337808840186, -0.048728438722316116, 0.0197721336455542, 0.06556500190279398, 0.0018870632264471824, 0.054619894201103386, -0.014474676889596716, -0.03163269322368256, 0.021214609213242554, 0.011827973674932461, -0.003756700271722839, 0.03430468356413156, -0.002677588677457068, -0.0558639158421712, -0.01742399685259912, -0.012650359538562665, -0.00856190109721383, 0.019469936488313074, 0.005718567355217994, -0.0780107319958215, -0.07315253724259392, 0.03797043078800511, -0.01054849459648322, 0.17489106325825357, -0.057676829779439734, 0.03402797307776175, 0.0012851812831857733, 0.016639198470596157, -0.06792121910461782, -0.02586070800657462, -0.04290710569377836, 0.02349575875769397, 0.041171753337170894, 0.03884841483938108, 0.045403205850847825, -0.017482136521406736, 0.005933939143582116, 0.002352724333369348, 0.02766740032472433, 0.02883456004110512, -0.0026861783174863267, 0.0008073494953390152, 0.03380179138403547, 0.005915152689083245, -0.002882955781775115, 0.056276738007192066, 0.05808182633477255, 0.04406471040438901, -0.0005049703359937697, 0.007488970357101936, 0.004525244899195622, -0.005546128105756946, -0.016115604421949824, 0.011054877889537903, -0.008391168937987291, 0.0348334686944711, -0.0018348845437176522, 0.03524395636945756};

    std::cout << estimate(days, vol, shock_array) << std::endl;

    return 0;
}