#include "binomial_distribution.h"
#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

void binomial_distribution::calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    for (auto cls = 0; cls < y.shape()[1]; cls++) {
        auto cls_col = xt::eval(xt::view(y, xt::all(), xt::range(cls, cls + 1)));
        cls_col.reshape({cls_col.shape()[0]});
        
        m_class_prior(cls) = xt::sum(cls_col)(0) / y.shape()[0];
    }

    update_feature_log_prior(xt::linalg::dot(xt::transpose(y), x));
    update_class_log_prior();
}

xt::xarray<double> binomial_distribution::predict(const xt::xarray<double>& x) const
{
    return xt::linalg::dot(x, xt::transpose(m_feature_log_prob)) + m_class_log_prior;
}

xt::xarray<double> binomial_distribution::weights() const
{
    return xt::xarray<double>();
}

void binomial_distribution::set_weights(const xt::xarray<double>& weights)
{
}

void binomial_distribution::update_feature_log_prior(const xt::xarray<double>& feature_count)
{
    // TODO: alpha as parameter
    double alpha = 1e-10;

    auto sfc = xt::eval(feature_count + alpha);
    auto scc = xt::eval(xt::sum(feature_count, {1}));
    scc.reshape({-1,1});
    m_feature_log_prob = xt::log(sfc) - xt::log(scc);
}

void binomial_distribution::update_class_log_prior()
{
    m_class_log_prior = xt::log(m_class_prior);
}