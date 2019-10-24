#include "../src/linreg.h"
#include "../src/kfold.h"
#include <iostream>
#include <algorithm>

int main(int argc, char** args)
{
    double X1[] = { 70, 72, 91, 58, 49, 50, 48, 33, 61, 51, 78, 70, 35, 81, 70, 47, 55, 70, 89, 68, 42, 93, 54, 52, 72, 62, 65, 98, 39, 50, 62, 45, 11, 60, 60, 74, 64, 56, 71, 40, 76, 88, 55, 60, 79, 109, 51, 48, 25, 88};
    double Y[] = { 351000, 390000, 473000, 282000, 300000, 286000, 228000, 181000, 308000, 289000, 414000, 358000, 165000, 397000, 352000, 239000, 322000, 376000, 499000, 383000, 229000, 424000, 256000, 256000, 363000, 328000, 331000, 465000, 273000, 215000, 287000, 207000, 7000, 328000, 282000, 322000, 305000, 317000, 406000, 225000, 407000, 443000, 294000, 277000, 393000, 576000, 254000, 263000, 101000, 426000};

    std::vector<std::vector<double>> x; 
    std::vector<double> y (std::begin(Y), std::end(Y));
    x.push_back(std::vector<double>(std::begin(X1), std::end(X1)));

    std::vector<std::vector<double>> x_test, x_train;
    std::vector<double> y_test, y_train;

    kfold kf(4, false);
    linear_regression lr;
    std::vector<double> mean_coeffs;

    for (auto i = 0; i < kf.k(); ++i) {
        kf.split(i, x, y, x_train, x_test, y_train, y_test);
        lr.train(x_train, y_train);
        std::cout << "KFOLD " << i + 1 << std::endl;
        std::cout << "coeffs: ";
        for (auto coeff : lr.coeffs())
            std::cout << coeff << " ";
        std::cout << std::endl;
        std::cout << "score: " << lr.score(x_test, y_test) << std::endl;

        if (mean_coeffs.size())
            std::transform(mean_coeffs.begin(), mean_coeffs.end(), lr.coeffs().begin(), mean_coeffs.begin(), std::plus<>());
        else 
            mean_coeffs = lr.coeffs();
    }

    std::transform(mean_coeffs.begin(), mean_coeffs.end(), mean_coeffs.begin(), std::bind(std::divides<>(), std::placeholders::_1, kf.k()));
    lr.set_coeffs(mean_coeffs);
    std::cout << "mean coeffs: ";
    for (auto coeff : lr.coeffs())
        std::cout << coeff << " ";
    std::cout << std::endl;
    std::cout << "mean score: " << lr.score(x_test, y_test) << std::endl;
}
