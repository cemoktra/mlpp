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

void logistic_regression::init_classes(size_t number_of_classes)
{
    m_classes = number_of_classes;
}

void logistic_regression::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    auto y_onehot = (y.shape()[1] == m_classes) ? y : one_hot::transform(y, m_classes);
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

double logistic_regression::score(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    xt::xarray<double> p = predict(x);
    xt::xarray<size_t> predict_class = xt::argmax(p, {1});
    xt::xarray<size_t> target_class;
    
    if (y.shape()[1] > 1)
        target_class = xt::argmax(p, {1});
    else
        target_class = y;
    target_class.reshape(predict_class.shape());
    return xt::sum(xt::equal(predict_class, target_class))(0) / static_cast<double>(y.shape()[0]);
}


void logistic_regression::calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    double current_cost, last_cost;
    m_weights = xt::ones<double>( { x.shape()[1], y.shape()[1] } );
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
    xt::xarray<double> c1 = -y * xt::log(p);
    xt::xarray<double> c2 = (1 - y) * xt::log(1.0 - p);
    return xt::sum(c1 - c2)(0) / p.shape()[0];
}

xt::xarray<double> logistic_regression::gradient(const xt::xarray<double>& x, const xt::xarray<double>& y, const xt::xarray<double>& p)
{
    xt::xarray<double> gradient = xt::linalg::dot(xt::transpose(x), (p - y));
    return get_param("learning_rate") * gradient / x.shape()[0];
}


xt::xarray<double> logistic_regression::sigmoid(const xt::xarray<double>& x)
{    
    xt::xarray<double> e = xt::exp(-x);
    xt::xarray<double> q = e + 1;
    xt::xarray<double> result = 1.0 / q;
    if (xt::any(result < 0.0) || xt::any(result > 1.0))
        throw std::out_of_range("ERROR IN SIGMOID");    
    return result;
}