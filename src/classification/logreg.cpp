#include "logreg.h"
#include <core/one_hot.h>
#include <stdexcept>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <iostream>

logistic_regression::logistic_regression() 
    : classifier()
{
    register_param("learning_rate", 0.02);
    register_param("threshold", 0.0001);
    register_param("max_iterations", 0);
}

xt::xarray<double> logistic_regression::predict(const xt::xarray<double>& x) 
{
    if (x.shape()[1] != m_weights.shape()[0])
        throw std::invalid_argument("x dimension is wrong");
    return sigmoid(xt::linalg::dot(x, m_weights));
}

void logistic_regression::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    auto y_onehot = (y.size() > 1 && y.shape()[1] == m_classes) ? y : one_hot::transform(y, m_classes);
    calc_weights(x, y_onehot);
}

void logistic_regression::set_weights(const xt::xarray<double>& weights)
{
    m_weights = weights;
}

xt::xarray<double> logistic_regression::weights()
{
    return m_weights;
}

void logistic_regression::calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    double current_cost, last_cost;
    m_weights = xt::zeros<double>( { x.shape()[1], y.shape()[1] } );
    size_t max_iterations = static_cast<size_t>(get_param("max_iterations"));
    size_t iteration = 0;
    last_cost = std::numeric_limits<double>::max();

    while (true) {
        xt::xarray<double> p = predict(x);
        xt::xarray<double> g = gradient(x, y, p);
        m_weights -= g;

        current_cost = cost(y, p);
        if (last_cost - current_cost < get_param("threshold"))
            break;
        last_cost = current_cost;
        iteration++;
        if (max_iterations > 0 && iteration >= max_iterations)
            break;
    }
}

double logistic_regression::cost(const xt::xarray<double>& y, const xt::xarray<double>& p)
{
    xt::xarray<double> c1 = -y * xt::eval(xt::log(p));
    xt::xarray<double> c2 = xt::eval(1 - y) * xt::eval(xt::log(xt::eval(1 - p)));
    return xt::eval(xt::sum(xt::eval(c1 - c2)))(0) / p.shape()[0];
}

xt::xarray<double> logistic_regression::gradient(const xt::xarray<double>& x, const xt::xarray<double>& y, const xt::xarray<double>& p)
{
    xt::xarray<double> gradient = xt::linalg::dot(xt::transpose(x), (p - y));
    return get_param("learning_rate") * gradient / x.shape()[0];
}


xt::xarray<double> logistic_regression::sigmoid(const xt::xarray<double>& x)
{    
    return 1.0 / (xt::exp(-x) + 1);
}