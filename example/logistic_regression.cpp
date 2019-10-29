#include "../src/logreg.h"
#include "../src/traintest.h"
#include "../src/standardscale.h"
#include "../test/testdata.h"
#include <iostream>
#include <algorithm>

int main(int argc, char** args)
{
    std::vector<std::vector<double>> x; 
    std::vector<double> y (std::begin(sale_success_by_age_and_interest.success), std::end(sale_success_by_age_and_interest.success));
    x.push_back(std::vector<double>(std::begin(sale_success_by_age_and_interest.age), std::end(sale_success_by_age_and_interest.age)));
    x.push_back(std::vector<double>(std::begin(sale_success_by_age_and_interest.interest), std::end(sale_success_by_age_and_interest.interest)));
    auto x_ = standard_scale::transform(x);

    std::vector<std::vector<double>> x_test, x_train;
    std::vector<double> y_test, y_train;
    
    test_train::split(x_, y, x_train, x_test, y_train, y_test);
    logistic_regression lr;

    lr.train(x_train, y_train);
    std::cout << "coeffs: ";
    for (auto coeff : lr.coeffs())
        std::cout << coeff << " ";
    std::cout << std::endl;
    std::cout << "score: " << lr.score(x_test, y_test) << std::endl;
    
    return 0;
}
