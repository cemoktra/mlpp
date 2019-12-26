#include "gauss_distribution.h"
#include <xtensor/xview.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor-blas/xlinalg.hpp>

void gauss_distribution::calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    m_class_prior.resize( { y.shape()[1] } );

    auto all_means = xt::eval(xt::mean(x, {0}));
    double epsilon = xt::eval(xt::amax(xt::variance(x, {0})))[0];

    for (auto cls = 0; cls < y.shape()[1]; cls++) {
        auto cls_col = xt::eval(xt::view(y, xt::all(), xt::range(cls, cls + 1)));
        cls_col.reshape({cls_col.shape()[0]});
        auto idx = xt::flatten_indices(xt::argwhere(cls_col > 0.0));
        auto x_class = xt::view(x, xt::keep(idx), xt::all());

        auto mean = xt::eval(xt::mean(x_class, {0}));
        auto var = xt::eval(xt::variance(x_class, {0}));

        mean.reshape({ mean.shape()[0], 1 });
        var.reshape({ mean.shape()[0], 1 });

        if (cls == 0) {
            m_theta = mean;
            m_sigma = var;
        } else {
            m_theta = xt::concatenate(std::make_tuple<>(m_theta, mean), 1);
            m_sigma = xt::concatenate(std::make_tuple<>(m_sigma, var), 1);
        }
        m_class_prior(cls) = xt::sum(cls_col)(0) / x.shape()[0];
    }
    m_sigma += epsilon;
}

xt::xarray<double> gauss_distribution::predict(const xt::xarray<double>& x) const
{
    xt::xarray<double> joint_log_likelihood;
    auto sigma_t = xt::transpose(m_sigma);

    for (auto cls = 0; cls < m_class_prior.shape()[0]; cls++) {
        xt::xarray<double> jointi = std::log(m_class_prior(cls));
        xt::xarray<double> tmp1 = -0.5 * xt::sum(xt::eval(xt::square(x - xt::transpose(xt::view(m_theta, xt::all(), xt::range(cls, cls + 1)))) / xt::view(sigma_t, xt::range(cls, cls + 1), xt::all())), {1});
        xt::xarray<double> tmp2 = -0.5 * xt::sum(xt::eval(xt::log(2.0 * xt::numeric_constants<double>::PI * xt::transpose(xt::view(sigma_t, xt::range(cls, cls + 1), xt::all())))));

        xt::xarray<double> log_likelihood = xt::transpose(xt::eval(tmp1 + tmp2) + jointi);
        log_likelihood.reshape(std::vector<size_t>({log_likelihood.shape()[0], 1}));
        if (cls == 0)
            joint_log_likelihood = log_likelihood;
        else
            joint_log_likelihood = xt::concatenate(std::make_tuple<>(joint_log_likelihood, log_likelihood), 1);
    }
    return joint_log_likelihood;
}

xt::xarray<double> gauss_distribution::weights() const
{
    return xt::xarray<double>();
}

void gauss_distribution::set_weights(const xt::xarray<double>& weights)
{
}    