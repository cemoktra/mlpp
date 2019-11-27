#include "svm.h"
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xio.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <iostream>
#include <core/one_hot.h>

svm::svm()
{
    register_param("c", 1.0);
    register_param("learning_rate", 0.02);
    register_param("threshold", 0.001);
    register_param("max_iterations", 0);
}

xt::xarray<double> svm::predict(const xt::xarray<double>& x)
{
    return xt::linalg::dot(x, m_weights);
}

double svm::score(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    xt::xarray<double> p = predict(x);
    xt::xarray<size_t> predict_class = xt::greater(p, 0);
    xt::xarray<size_t> target_class;
    
    if (y.shape().size() > 1 && y.shape()[1] > 1)
        target_class = xt::argmax(y, {1});
    else
        target_class = y;

    target_class.reshape(predict_class.shape());
    return xt::sum(xt::equal(predict_class, target_class))(0) / static_cast<double>(y.shape()[0]);
}

void svm::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    size_t max_iterations = static_cast<size_t>(get_param("max_iterations"));
    size_t iteration = 0;
    double current_cost, last_cost;
    m_weights = xt::ones<double>({x.shape()[1], y.shape()[1]});
    last_cost = std::numeric_limits<double>::max();
    
    if (m_classes > 2)
        throw std::exception();

    while (true) {
        auto p = predict(x);
        auto prod = xt::eval(p * y);

        current_cost = 0;
        for (auto c = 0; c < prod.shape()[1]; c++) 
        {    
            for (auto r = 0; r < prod.shape()[0]; r++)
            {
                auto wcol= xt::view(m_weights, xt::all(),xt::range(c, c + 1));
                if (prod(r, c) >= 1.0) {
                    wcol = wcol - get_param("learning_rate") * 2.0 * (1.0 / (iteration + 1)) * wcol;
                }
                else {
                    current_cost += 1 - prod(r, c);
                    auto xy = xt::eval(xt::view(x, xt::range(r, r + 1), xt::all()) * y(r, c));
                    wcol = wcol + get_param("learning_rate") * xy - 2.0 * (1.0 / (iteration + 1)) * wcol;
                }     
            }
        }
        current_cost /= (y.shape()[0] * y.shape()[1]);

        if (fabs(last_cost - current_cost) < get_param("threshold"))
            break;
        last_cost = current_cost;
        iteration++;
        if (max_iterations > 0 && iteration >= max_iterations)
            break;
    }
}

void svm::init_classes(size_t number_of_classes)
{
    m_classes  = number_of_classes;
}

void svm::set_weights(const xt::xarray<double>& weights)
{
}

xt::xarray<double> svm::weights()
{
    return xt::xarray<double>();
}