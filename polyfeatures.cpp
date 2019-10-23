#include "polyfeatures.h"

#include <iostream>
#include <numeric>

polynomial_features::polynomial_features(size_t degree, bool bias)
    : m_degree(degree)
    , m_bias(bias)
{
}

std::vector<std::vector<double>> polynomial_features::transform(const std::vector<std::vector<double>> &x)
{
    // TODO
    return std::vector<std::vector<double>>();
}