#include "multinomial_logreg.h"
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>

multinomial_logistic_regression::multinomial_logistic_regression() 
    : logistic_regression()
{
}

xt::xarray<double> multinomial_logistic_regression::predict(const xt::xarray<double>& x) 
{
    return softmax(xt::linalg::dot(x, m_weights));
}

xt::xarray<double> multinomial_logistic_regression::softmax(const xt::xarray<double>& z)
{
    xt::xarray<double> e = xt::exp(z);
    return e / xt::sum(e, { 0 });
}

double multinomial_logistic_regression::cost(const xt::xarray<double>& y, const xt::xarray<double>& p)
{
    return xt::sum(-y * xt::log(p))(0) / y.shape()[0];
}