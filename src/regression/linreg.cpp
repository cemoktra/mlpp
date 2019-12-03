#include "linreg.h"
#include <stdexcept>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

linear_regression::linear_regression() 
{
}

xt::xarray<double> linear_regression::predict(const xt::xarray<double>& x) const
{
    if (x.shape()[1] != m_weights.shape()[0])
        throw std::invalid_argument("x dimension is wrong");
    return xt::linalg::dot(x, m_weights);
}

void linear_regression::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    calc_weights(x, y);
}

double linear_regression::score(const xt::xarray<double>& x, const xt::xarray<double>& y) const
{
    xt::xarray<double> p = predict(x);
    xt::xarray<double> e = y - p;
    xt::xarray<double> mean = xt::mean(xt::view(y, xt::all(), xt::range(0, 1)));
    xt::xarray<double> ymean = y - mean;
    return 1.0 - (pow(xt::linalg::norm(e), 2) / pow(xt::linalg::norm(ymean), 2));
}

void linear_regression::set_weights(const xt::xarray<double>& weights)
{
    m_weights = weights;
}

xt::xarray<double> linear_regression::weights() const
{
    return m_weights;
}

void linear_regression::calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    xt::xarray<double> xTx = xt::linalg::dot(xt::transpose(x), x);
    xt::xarray<double> xTy = xt::linalg::dot(xt::transpose(x), y);
    xt::xarray<double> xTx_inverse = xt::linalg::inv(xTx);
    m_weights = xt::linalg::dot(xTx_inverse, xTy);
}
