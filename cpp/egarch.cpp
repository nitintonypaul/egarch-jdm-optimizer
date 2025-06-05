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

// Function Declarations
double estimate(int no_days, double volatility, const std::vector<double> &shockarray);
void optimize(const std::vector<double> &shockarray);
void compute_gradient(const std::vector<double> &shockarray);
double compute_volatility(int t, const std::vector<double> &shockarray);
double backtrack_line_search(double dir[], const std::vector<double> &shockarray);
bool armijo_condition(double alpha, double dir[], const std::vector<double> &shockarray);

// Parameters
// Order: Alpha, Beta, Omega, Gamma
double parameters[] = {0.1,0.9,-0.1,-0.1}; 

// Gradient array
double gradient[4] = {0, 0, 0, 0};

// Initialising global variables
double lambda = 1;
const double ARMIJO_C = 1e-4;
int days;
double vol;
const double LOG_CLAMP = 700.0;
const double NLL_Z_TERM_MAX = 1e9;

// Core function
double estimate(int no_days, double volatility, const std::vector<double> &shockarray) {

    // Assigning global variables
    days = no_days;
    vol = volatility;
    
    // Resetting parameters and step value before optimization
    parameters[0] = 0.1;
    parameters[1] = 0.9;
    parameters[2] = -0.1;
    parameters[3] = -0.1;
    lambda = 1.0;
    
    // Zeroing out gradient completely
    std::fill(gradient, gradient + 4, 0.0);

    // Calling the optimizing function to find parameters
    optimize(shockarray);
    
    // Assigning computed volatility
    // Didn't have to assign, but makes it easier for debugging
    double computed_volatility =  compute_volatility(days, shockarray);

    // Debugging code
    // std::cout << "VOLATILITY: " << k << std::endl;
    return computed_volatility;
}

// Conjugate Gradient Optimization 
void optimize(const std::vector<double> &shockarray) {

    // Debugging code
    // std::cout << "ALPHA: " << lambda << " PARAMETERS: " << parameters[0] << "," <<  parameters[1] << "," <<  parameters[2] << "," <<  parameters[3] << std::endl;

    // First iteration is done manually
    // Computing initial gradient 
    compute_gradient(shockarray);

    // Declaring direction
    double direction[4] = {0,0,0,0};
    
    // Finding direction
    // Direction is -gradient(f)
    for (int i = 0; i < 4; i++) {
        direction[i] = -gradient[i];
    }

    // Computing lambda (Step value)
    // Using backtrack line search
    lambda = backtrack_line_search(direction, shockarray);

    // Optimizing parameters using step value
    // Optimization formula: P = P + (λ * D)
    for (int i = 0; i < 4; i++) {
        parameters[i] = parameters[i] + (lambda * direction[i]);
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
            beta += gradient[j] * gradient[j];
        }

        // Declaring numerator product
        // betaplus is a temporary variable named for numerator
        double betaplus = 0;

        // Computing next gradient
        compute_gradient(shockarray);
        
        // Computing numerator product
        for (int j = 0; j < 4; j++) {
                betaplus += gradient[j] * gradient[j];
        }
        
        // Computing beta_k (beta)
        // beta is used to update direction
        beta = betaplus / beta;

        // Updating direction using Fletcher Reeves
        for(int j = 0; j < 4; j++) {
            direction[j] = -gradient[j] + (beta * direction[j]);
        }

        // DOT PRODUCT DIAGONOSTIC
        // When the dot product of the resulting direction and gradient is significant enough, it can lead to parameter value explosions
        // To combat this phenomonen the direction is reset to simply as -gradient
        // essentially nullifying the FLETCHER REEVES PROCESS
        // "Math is math" doesn't really apply here... lol
        // Computing dot product (Direction, Gradient)
        double dot = 0;
         for (int j = 0; j < 4; j++) {
            dot += gradient[j] * direction[j];
        }

        // Checking if the dot product is significant
        // if it is, it is reset normally
        // This is a diagnostic fix and is not intended to reflect theoretical accuracy
        if (dot >= -1e-12) { 
            for (int j = 0; j < 4; j++) {
                direction[j] = -gradient[j];
            }
        }

        // Computing lambda (step value)
        // Using backtrack line search
        lambda = backtrack_line_search(direction, shockarray);

        // Convergence condition
        // If the step value is insignificant, then there is no significant change to the parameters
        // Hence the loop is broken to retain efficiency
        // Regardless the loop executes at least 10 iterations to prevent premature convergence
        if (lambda < 1e-8 && i > 10) {
            // Debugging code
            //std::cout << "BREAKING SINCE STEP IS " << lambda << std::endl;
            break;
        }

        // Optimizing parameters
        for (int j = 0; j < 4; j++) {
            parameters[j] = parameters[j] + (lambda * direction[j]);
        }

        // Debugging code
        //std::cout << "ALPHA: " << lambda << " PARAMETERS: " << parameters[0] << "," <<  parameters[1] << "," <<  parameters[2] << "," <<  parameters[3] << std::endl;
        // Repeat until 1000. Or until convergence happens.
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
        // Computing SIGMA
        // SIGMA is clamped to 1e-9 to prevent NaN values
        double sigma_t = compute_volatility(i, shockarray);
        double sigma_prev = compute_volatility(i-1, shockarray);
        if (std::abs(sigma_t) < 1e-9) sigma_t = 1e-9;
        if (std::abs(sigma_prev) < 1e-9) sigma_prev = 1e-9;

        // Computing Z_t
        double z_t = shockarray[i] / sigma_t;
        double z_prev = shockarray[i-1] / sigma_prev;

        // Z values are clamped to ensure absence of step value explosion
        // Changing the clamp values of Zs are not recommended since it leads to insigificant step values such as 2e-30
        //z_t = std::max(-NLL_Z_TERM_MAX, std::min(z_t, NLL_Z_TERM_MAX));
        //z_prev = std::max(-NLL_Z_TERM_MAX, std::min(z_prev, NLL_Z_TERM_MAX));


        // C_T is the vector corresponding to each parameter
        double CT[] = {std::abs(z_prev), std::log(std::pow(sigma_prev,2)), 1, z_prev};

        // Adding immediate derivative
        // Immediate derivative is common for every parameter
        double alpha_der = 0.5 * (1 - std::pow(z_t,2));
        double beta_der = 0.5 * (1 - std::pow(z_t,2));
        double omega_der = 0.5 * (1 - std::pow(z_t,2));
        double gamma_der = 0.5 * (1 - std::pow(z_t,2));

        // Updating PSI with the previous value
        // PSI is the 'parent' recursive variable
        for(int j = 0; j < 4; j++) {
            PSI[j] = CT[j] + (parameters[1] * PSI[j]);
        }

        // Multiplying with parameter derivatives and adding to the gradient
        // gradient is accessed globally
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

        // Clamping sig to 1e-9 to prevent NaN values
        if (std::abs(sig) < 1e-9) sig = 1e-9;

        // EGARCH equation to compute logarithmic volatility squared
        double logsigsq = parameters[2] + (parameters[1] * std::log(std::pow(sig, 2))) + (parameters[0] * std::abs(shockarray[i-1] / sig)) + (parameters[3] * (shockarray[i-1] / sig));

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

// Function for backtrack line searching
// Used to compute the step value required for each independent iteration
// works under Armijo condition
double backtrack_line_search(double dir[], const std::vector<double> &shockarray) {

    // Declaring step value (ALPHA) and beta (step multilplier)
    // Beta acts as a step value for the step value
    double beta = 0.5;
    double alpha = 1;

    // While condition is false
    while (!armijo_condition(alpha, dir, shockarray)) {

        // ALPHA is multiplied with beta
        // ALPHA is essentially halved every step
        alpha *= beta;
    }

    // returning the computed step value
    return alpha;
}

// Function to check whether Armijo conditions retains false
// line search is to break when armijo condition is true
// ARMIJO CONDITON: f(x + ALPHA.d) <= f(x) + c.ALPHA.T(grad(f(x))).d
// Where ALPHA is the step value
bool armijo_condition(double alpha, double dir[], const std::vector<double> &shockarray) {

    // COMPUTING LHS 
    // LHS = f(x + ALPHA.d)
    // f -> Negative Log Likelihood function
    
    // Declaring sig (initial volatility)
    double sig = vol;
    double sigN = vol;

    // Declaring total likelihood
    // Which is the LHS in the Armijo Condition ( with modified parameters)
    double LHS = 0;

    // Declaring log likelihood and computing the same to reduce complexity
    // this total_likelihood is used for RHS computation
    // f(x)
    double total_likelihood = 0;

    // Computing volatility under current parameters
    for (int i = 1; i < days; i++) {

        // Clamping sig to 1e-9 to prevent NaN values
        if (std::abs(sig) < 1e-9) sig = 1e-9;

        // EGARCH equation to compute logarithmic volatility squared
        // Modified with different parameters
        // logsiqsqN is done with normal parameters to compute f(x) for RHS
        double logsigsq = (parameters[2] + dir[2]*alpha) + ((parameters[1] + dir[1]*alpha) * std::log(std::pow(sig, 2))) + ((parameters[0] + dir[0]*alpha) * std::abs(shockarray[i-1] / sig)) + ((parameters[3] + dir[3]*alpha) * (shockarray[i-1] / sig));
        double logsigsqN = parameters[2] + (parameters[1] * std::log(std::pow(sigN, 2))) + (parameters[0] * std::abs(shockarray[i-1] / sigN)) + (parameters[3] * (shockarray[i-1] / sigN));

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
        tempsum += gradient[i]*dir[i];
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

/*TEST OUTPUT

int main() {

    // Stock Chosen for test: TSLA
    
    int days = 180;
    double vol = 0.03634437544828279;
    const std::vector<double> &shock_array = {0.0012297495368080837, -0.00650973683920264, 0.06744448098003801, -0.027084893417118134, 0.044575234549721024, 0.01337065273083922, 0.007192007984448459, -0.014518962243894536, 0.020684086286020115, 0.000916904059319064, -0.01745939263063975, -0.039068967814471264, -0.03771320823262106, 0.034830568989869345, -0.041254716777913245, 0.011558870141369783, -0.017776056181240646, -0.013068770910621152, -0.09548867964710306, 0.0026597080911496635, -0.0016960925198351783, 0.004418546310355509, -0.005555112219541625, -0.004425678628740895, -0.011982848687334811, -0.0075942880634069215, -0.023583451496752103, 0.19462185069099375, 0.029326193277898618, -0.02869338608061214, -0.015020625929560247, -0.011185040757149548, -0.03391820717413026, -0.007053348203363615, -0.028534920390102938, 0.031236486238900563, 0.13402951946916286, 0.0250648716861814, 0.07513203474596467, 0.08224176327028512, -0.06699192892949678, 0.0017481261714555139, -0.0630132747101637, 0.026631832012274488, 0.05109920971511139, 0.017640811645799743, -0.015105466891194508, -0.010577313462489999, 0.0337693965702532, -0.04399602931270749, -0.004628895932059315, -0.019479161615941674, 0.032630798388913376, 0.030414611564264043, -0.019570853831010624, 0.01479014777521441, 0.02822109312124024, 0.04845594881852576, -0.0021017258828989935, 0.02476316972314454, 0.0540462143366207, -0.019392307918283117, 0.03888385863926566, 0.056035432813588415, 0.03215899741755826, -0.08998935603032143, -0.012603183294903772, -0.0388219116310389, 0.01883912278232164, 0.0674260255837058, -0.021352394827650573, -0.05431040238079732, -0.037134429629081867, -0.03661535089737744, -0.06630917624740101, 0.07538990478867226, -0.002080069819539433, -0.04501581211715883, -0.0020954405293401412, -0.004071706002516027, 0.017913054591346785, -0.020947779501058802, 0.07374909205016995, -0.03777111146959018, 0.02661611178322025, -0.009278954493688867, -0.024920183933157144, -0.01016338463739529, -0.01772972841348825, -0.027031791634221366, -0.0012010673142449357, -0.026406846276289128, 0.024762762717671406, 0.007169499868106744, -0.05665522784741188, 0.018423395317938473, -0.04001866414833721, -0.013797940129306678, -0.03808228368250438, -0.03414233598760291, -0.06904503086456125, 0.020525916642673545, 0.05256914805711156, -0.003846143628378932, -0.008438764146744299, 0.014485672870844696, -0.020797310880639457, -0.05153737405019097, -0.02532168597772474, -0.09119008256609133, -0.04400191834571329, -0.03447105091195446, 0.034809450473757886, -0.032409191623483345, -0.04887632901515568, 0.022055875914651905, -0.06127166019847188, -0.00653024118405064, -0.17111088145579012, 0.03367993906050802, 0.06962854627202306, -0.033888489653546086, 0.03434753687185593, -0.05263337183311424, -0.058400586412649155, 0.04219604958985143, -0.0018706817579388956, 0.0477895750349268, 0.10917033403209081, 0.030858322299713827, -0.06098904351795177, 0.0003601281859736301, -0.039269973990397236, -0.020362552049278032, 0.031691108739425794, 0.048331479831251356, -0.05986686313639594, -0.11360086629378592, -0.02954392268052511, -0.05380087178011993, 0.20092543922338743, -0.0790871693562535, -0.003921766723283929, -0.0034065843453072817, 0.0033850687872823467, -0.05425588368221071, -0.0043106387637591, -0.06274590803614198, 0.041429241266716946, 0.04870680638153779, 0.030813566766049894, 0.089953353202046, -0.00030675046671064035, 0.017719232326954937, -0.03794737440846216, -0.009394455180293842, 0.02000350047515987, -0.02806098794919671, -0.02123988149028043, -0.0004105245839973906, 0.027094632659224276, 0.04254305379187294, 0.06171489040858284, 0.044539779738598045, 0.0363667808836973, -0.01764208677682675, 0.01710535116315947, -0.026367332530845767, 0.0014792955053417496, -0.03068788165663621, 0.015439111696818088, -0.008562394467716845, 0.06353211380376568, -0.02020932578848646, 0.0007126083477966144, -0.03753115611158822, -0.014506232427285493, 0.0010348050058056066, -0.03970581887634815, -0.09276814903795098};

    std::cout << estimate(days, vol, shock_array) << std::endl;

    return 0;
}*/