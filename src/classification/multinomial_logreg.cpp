#include "multinomial_logreg.h"
#include <xtensor-blas/xlinalg.hpp>

multinomial_logistic_regression::multinomial_logistic_regression() 
    : logistic_regression()
{
}

xt::xarray<double> multinomial_logistic_regression::activation(const xt::xarray<double>& x) const
{
    // softmax function
    xt::xarray<double> e = xt::exp(x);
    xt::xarray<double> sum = xt::sum(e, { 1 });
    sum.reshape({ sum.shape()[0], 1 });
    return e / sum;
}

xt::xarray<double> multinomial_logistic_regression::reverse_activation(const xt::xarray<double>& y) const
{
    // reverse softmax without C
    // C could be added to have prediction values in the range of 0 to 1. This  makes no difference for the
    // classification but has a huge impact on performance
    double epsilon = get_param("epsilon");
    xt::xarray<double> y_ = xt::eval(xt::where(xt::equal(y, 0), epsilon, 1 - epsilon));
    return xt::log(y_);
}