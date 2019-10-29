#include "polyfeatures.h"
#include <numeric>
#include <algorithm>

polynomial_features::polynomial_features(size_t degree, bool bias)
    : m_degree(degree)
    , m_bias(bias)
{
}

Eigen::MatrixXd polynomial_features::transform(const Eigen::MatrixXd &x)
{
    Eigen::MatrixXd result = x;

    std::vector<double> polynom(x.cols());
    for (auto i = 2u; i <= m_degree; i++) {
        std::fill(polynom.begin(), polynom.end(), 0.0);

        int digit = 0;
        while (true) {
            if (i == std::accumulate(polynom.begin(), polynom.end(), 0)) {
                // create data
                result.conservativeResize(Eigen::NoChange, result.cols() + 1);
                Eigen::VectorXd newcol = Eigen::VectorXd::Ones(x.rows());
                for (auto k = 0; k < polynom.size(); k++) {
                    Eigen::VectorXd p = result.col(k).array().pow(polynom[k]);
                    newcol = newcol.cwiseProduct(p);
                }
                result.col(result.cols() - 1) = newcol;
            }

            digit = 0;
            while (digit < x.cols()) {
                polynom[digit]++;
                if (polynom[digit] > i) {
                    polynom[digit++] = 0;
                } else
                    break;
            }
            if (digit >= x.cols())
                break;
        }
    }

    if (m_bias) {
        result.conservativeResize(Eigen::NoChange, result.cols() + 1);
        result.col(result.cols() - 1) = Eigen::VectorXd::Ones(result.rows()); // add bias
    }

    return result;
}