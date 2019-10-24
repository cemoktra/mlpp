#include "polyfeatures.h"

#include <iostream>
#include <math.h>
#include <numeric>

polynomial_features::polynomial_features(size_t degree, bool bias)
    : m_degree(degree)
    , m_bias(bias)
{
}

std::vector<std::vector<double>> polynomial_features::transform(const std::vector<std::vector<double>> &x)
{
    std::vector<std::vector<double>> result;

    if (m_bias) {
        std::vector<double> bias(x[0].size());
        std::fill(bias.begin(), bias.end(), 1.0);
        result.push_back(bias);
    }
    for (auto _x : x)
        result.push_back(_x);

    std::vector<double> polynom(x.size());
    for (auto i = 2u; i <= m_degree; i++) {
        std::fill(polynom.begin(), polynom.end(), 0.0);

        int digit = 0;
        while (true) {
            if (i == std::accumulate(polynom.begin(), polynom.end(), 0)) {
                // create data
                std::vector<double> data;
                for (auto j = 0; j < x[0].size(); j++) {
                    double value = 1.0;
                    for (auto k = 0; k < polynom.size(); k++) {
                        value *= std::pow(x[k][j], polynom[k]);
                    }
                    data.push_back(value);
                }
                result.push_back(data);
            }

            digit = 0;
            while (digit < x.size()) {
                polynom[digit]++;
                if (polynom[digit] > i) {
                    polynom[digit++] = 0;
                } else
                    break;
            }
            if (digit >= x.size())
                break;
        }
    }


    return result;
}