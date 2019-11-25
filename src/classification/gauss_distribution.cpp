#include "gauss_distribution.h"
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xio.hpp>
#include "xtensor/xreducer.hpp"
#include <xtensor-blas/xlinalg.hpp>

void gauss_distribution::calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    m_theta = xt::zeros<double>( { x.shape()[1], y.shape()[1] } );
    m_sigma = xt::zeros<double>( { x.shape()[1], y.shape()[1] } );
    m_class_prior.resize( { y.shape()[1] } );

    // TODO: add parameter for epsilon (on/off)
    m_epsilon = 0.0;
    // m_epsilon = xt::eval(xt::amax(xt::variance(x, {0})))[0];

    for (auto cls = 0; cls < y.shape()[1]; cls++) {
        auto cls_col = xt::eval(xt::view(y, xt::all(), xt::range(cls, cls + 1)));
        cls_col.reshape({cls_col.shape()[0]});
        auto idx = xt::flatten_indices(xt::argwhere(cls_col > 0.0));
        auto x_class = xt::view(x, xt::keep(idx), xt::all());

        auto mean = xt::eval(xt::mean(x_class, {0}));
        // TODO: use xt::stddev when pull request https://github.com/xtensor-stack/xtensor/pull/1627#issuecomment-558170772 has been merged into xtensor
        auto s1 = x_class.shape();
        auto s2 = mean.shape();
        auto var = xt::eval(xt::mean(xt::square(x_class - mean), {0}));

        mean.reshape({ mean.shape()[0], 1 });
        xt::view(m_theta, xt::all(), xt::range(cls, cls + 1)) = mean;
        var.reshape({ mean.shape()[0], 1 });
        xt::view(m_sigma, xt::all(), xt::range(cls, cls + 1)) = var;

        m_class_prior(cls) = xt::sum(cls_col)(0) / y.shape()[0];
    }
}

xt::xarray<double> gauss_distribution::predict(const xt::xarray<double>& x)
{
    auto shape = x.shape();
    shape[1] = m_class_prior.shape()[0];
    xt::xarray<double> joint_log_likelihood (shape);

    for (auto cls = 0; cls < m_class_prior.shape()[0]; cls++) {
        auto jointi = std::log(m_class_prior(cls));
        auto tmp1 = -0.5 * xt::sum(xt::square(x - xt::transpose(xt::view(m_theta, xt::all(), xt::range(cls, cls + 1)))) / xt::transpose(xt::view(m_sigma, xt::all(), xt::range(cls, cls + 1))), {1});
        auto tmp2 = -0.5 * xt::sum(xt::log(2.0 * xt::numeric_constants<double>::PI * xt::transpose(xt::view(m_sigma, xt::all(), xt::range(cls, cls + 1)))));
        xt::view(joint_log_likelihood, xt::all(), xt::range(cls, cls + 1)) = xt::transpose(xt::eval(tmp1 + tmp2) + jointi);
    }

    return joint_log_likelihood;
}

xt::xarray<double> gauss_distribution::weights()
{
    return xt::xarray<double>();
}

void gauss_distribution::set_weights(const xt::xarray<double>& weights)
{
}    