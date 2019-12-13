#include "logreg.h"
#include <preprocessing/one_hot.h>
#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <iostream>

logistic_regression::logistic_regression() 
    : classifier()
{
    register_param("epsilon", 1e-9);
    register_param("rcond", -1.0);
    register_param("threshold", 0.0001);
    register_param("max_iterations", 100);
    register_param("C", 1.0);
}

xt::xarray<double> logistic_regression::predict(const xt::xarray<double>& x) const
{
    if (x.shape()[1] != m_weights.shape()[0])
        throw std::invalid_argument("x dimension is wrong");
    return activation(xt::linalg::dot(x, m_weights));
}

void logistic_regression::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    auto y_onehot = (y.dimension() > 1 && m_classes > 2 && y.shape()[1] == m_classes) ? y : one_hot::transform(y);
    calc_weights(x, y_onehot);
}

void logistic_regression::set_weights(const xt::xarray<double>& weights)
{
    m_weights = weights;
}

xt::xarray<double> logistic_regression::weights() const
{
    return m_weights;
}

void logistic_regression::calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y)
{      
    m_weights = std::get<0>(xt::linalg::lstsq(x, reverse_activation(y), get_param("rcond")));
}

xt::xarray<double> logistic_regression::activation(const xt::xarray<double>& x) const
{
    // sigmoid
    return 1.0 / (xt::exp(-x) + 1);
}

xt::xarray<double> logistic_regression::reverse_activation(const xt::xarray<double>& y) const
{
    // reverse sigmoid
    double epsilon = get_param("epsilon");
    xt::xarray<double> y_ = xt::where(xt::equal(y, 0), epsilon, 1 - epsilon);
    return -xt::log((1.0 / y_) - 1.0);
}