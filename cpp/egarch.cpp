// Including modules
#include <iostream>
#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

// Namespace for pybind11
namespace py = pybind11;

// Function Declarations
double estimate(int no_days, double volatility, const std::vector<double> &shockarray);
void optimize(const std::vector<double> &shockarray);
void compute_gradient(const std::vector<double> &shockarray);
double compute_volatility(int t, const std::vector<double> &shockarray);

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

double estimate( int no_days, double volatility, const std::vector<double> &shockarray) {

    // Assigning global variables
    days = no_days;
    vol = volatility;

    // Calling the optimizing function to find parameters
    optimize(shockarray);

    // Returning computed volatility
    return compute_volatility(days, shockarray);
}

// Conjugate Gradient Optimization 
void optimize(const std::vector<double> &shockarray) {

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
void compute_gradient(const std::vector<double> &shockarray) {

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
double compute_volatility(int t, const std::vector<double> &shockarray) {
    
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

//Pybind11 declaration
PYBIND11_MODULE(egarch, m) {
    m.def("estimate", &estimate);
}

/*TEST OUTPUT*/
/*
int main() {
    
    // Stock Chosen for test: RELIANCE.NS
    int days = 180;
    double vol = 0.01058767442405473;
    const std::vector<double> &shock_array = {-0.0017527888369720385, -0.0007627691558581706, -0.007013042237308608, 0.019179353144027696, -0.0049905728007749875, -0.0009963113599895366, 0.0005153880712439762, -0.006159327405549098, 0.004114573728832392, 0.01086606680447724, 0.0048710334570512565, -0.0028122091635543675, 0.002936898514492704, 0.0025437689605181473, 0.018537044196617003, -0.033169572415035715, -0.00811955444816003, -0.04042387463346191, -0.014771476667475653, -0.011590974277875452, 0.019107670275681547, -0.01654492522338198, -0.0027159755647238235, 0.0006353710136417545, 0.00017961292864993137, -0.021113384555458033, 0.007319548166198156, 0.001603951030544575, 0.001987180163003538, 0.007126575852341338, -0.01919023490388163, -0.0037283151989873126, 0.0008219869896168964, -0.009089429934223996, 0.004753070239259238, 0.004095231483897186, 0.0027761198954648933, -0.008986826009036993, 0.004812397195667129, -0.027775017288777758, 0.002286054668192061, 0.015113501379654344, -0.015105656196376832, -0.01704565638036032, -0.008775011489717877, 0.0010870617087548379, -0.01770564213592713, 0.012212963714127261, -0.005548666479035645, -0.01539574093674042, -0.015264422091268085, 0.03395132143999171, 0.01679551271061655, 0.006607004474719088, -0.002061441735511909, -0.01760318615435357, 0.01656937363957487, 0.01290181706824997, 0.01062044451391635, -0.011033520776859578, 0.009828228453416996, -0.00810403689712347, -0.012713262581833537, -0.008114695274557835, -0.00531927986035268, -0.012172235910790714, 0.007717659677845936, -0.0037111208478030782, -0.018431074514401367, 0.006233554327408334, -0.01849037958906244, -0.020781496207419668, 0.013875711746308321, 0.0002379318093351625, -0.005213515262397053, 0.003562041386444998, -0.008642638797830556, 0.003785555139588479, 0.0046304670758547375, 0.016556936015068156, 0.00737104973862127, -0.026983097478358795, 0.018456320809004848, 0.019540573147985456, -0.008661070329308813, -0.010423983801591437, -0.0017822186429803204, -0.0010176970837927639, 0.010669039793967742, 0.011185588031991638, 0.0278225149941077, 0.0022473465193848484, -0.024751878119457567, 0.0025357322461881607, -0.01071760431522629, -0.013955295817739459, -0.013823765349094025, 0.00396936840698063, 0.0007605857224603021, 0.013974752989345, 0.009440413946139134, -0.0005254226829142093, -0.015027786297586347, 0.030926011469561338, -0.0055916291127598005, 0.002487402026378231, -0.011785384858131921, -0.01048585796962559, -0.0152399493060337, -0.015060580267698066, -0.0005001460041650552, 0.000815100272386559, 0.006134892983130252, 0.00027799441060869466, 0.001541346767297562, 0.0043812984238052214, -0.004071351080249935, -0.011265432839561164, -0.008854450013286079, 0.0024413027337566295, -0.005946021066830334, -0.024463430620579762, -0.008145056230799945, 0.01159189869726309, 0.028380952585070632, 0.03256375244287346, -0.009293452893244425, 0.00703089184988705, 0.007656371732097114, -0.007435705691237417, -0.007408766739300527, -0.00013011887917306446, 0.006547338617705618, 0.01735631768131491, 0.0055268996813062285, 0.01984378527010463, -0.012999632271286041, -0.009823298032958077, 0.003907046697290755, -0.0025583311519478606, -0.017933335021283518, -0.001288342610747984, -0.0020902957302698097, -0.036002557155654535, -0.0330389285822224, 0.013925226169932773, 0.00253088204606055, 0.027821604463363618, 0.017072090587104113, -0.0007753772301373848, 0.027877082466407237, 0.016212657306821386, -0.003454859181736085, 0.006662164870853314, 0.001099874819064145, -0.0010524488230168798, 0.051132432167001536, 0.022193294912523004, 0.003649290846780853, 0.010348629349113047, 0.007936986429393902, -0.007422784357731046, -0.010671816334703953, 0.0005808658646058641, -0.02153747998383025, 0.04202706353043925, -0.014715644388512826, 0.00599649675433729, 0.02057507921595747, 0.0013816475467545591, -0.010899224192407691, -0.011437612223744065, 0.0030936386986586636, -0.014080832648541012, 0.012139990612454622, 0.005461173103405996, -0.009020984357573527, -0.006620407039049737, 0.0033319429299430545, 0.0020539630866911575};

    std::cout << estimate(days, vol, shock_array) << std::endl;

    return 0;
}*/