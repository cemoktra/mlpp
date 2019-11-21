#include "gauss_distribution.h"
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

void gauss_distribution::calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    m_mean.resize( { x.shape()[1], y.shape()[1] } );
    m_var.resize( { x.shape()[1], y.shape()[1] } );
    m_pre_prop.resize( { y.shape()[1] } );

    for (auto cls = 0; cls < y.shape()[1]; cls++) {
        auto cls_col = xt::eval(xt::view(y, xt::all(), xt::range(cls, cls + 1)));
        cls_col.reshape({cls_col.shape()[0]});
        auto idx = xt::flatten_indices(xt::argwhere(cls_col > 0.0));
        auto x_class = xt::view(x, xt::keep(idx), xt::all());

        auto mean = xt::eval(xt::mean(x_class, {0}));
        mean.reshape({ mean.shape()[0], 1 });
        xt::view(m_mean, xt::all(), xt::range(cls, cls + 1)) = mean;

        auto var = xt::eval(xt::variance(x_class, {0}));
        var.reshape({ mean.shape()[0], 1 });
        xt::view(m_var, xt::all(), xt::range(cls, cls + 1)) = var;

        m_pre_prop(cls) = xt::sum(cls_col)(0) / y.shape()[0];
    }
}

xt::xarray<double> gauss_distribution::predict(const xt::xarray<double>& x)
{
    auto s = x.shape();
    s[1] = m_pre_prop.shape()[0];
    xt::xarray<double> pfc (s);
    xt::xarray<double> pcf (s);
    xt::xarray<double> total_prop = xt::zeros<double>({ s[0] });
    total_prop.reshape({ total_prop.shape()[0], 1 });

    for (auto c = 0; c < m_pre_prop.shape()[0]; c++) {
        xt::view(pfc, xt::all(), xt::range(c, c + 1)) = calc_pfc(x, c);
        total_prop = total_prop + xt::view(pfc, xt::all(), xt::range(c, c + 1)) * m_pre_prop(c);
    }

    for (auto c = 0; c < m_pre_prop.size(); c++)
        xt::view(pcf, xt::all(), xt::range(c, c + 1)) = (xt::view(pfc, xt::all(), xt::range(c, c + 1)) * m_pre_prop(c)) / total_prop;
    return pcf;    
}

xt::xarray<double> gauss_distribution::weights()
{
    auto s = m_mean.shape();
    s[0] = 2 * s[0] + 1;
    xt::xarray<double> weights (s);
    xt::view(weights, xt::range(0, 1), xt::all()) = m_pre_prop;
    xt::view(weights, xt::range(1, 1 + m_mean.shape()[0]), xt::all()) = m_mean;
    xt::view(weights, xt::range(1 + m_mean.shape()[0], xt::placeholders::_), xt::all()) = m_var;
    return weights;
}

void gauss_distribution::set_weights(const xt::xarray<double>& weights)
{
    auto meansize = (weights.shape()[0] - 1) / 2;
    m_pre_prop = xt::view(weights, xt::range(0, 1), xt::all());
    m_mean = xt::view(weights, xt::range(1, 1 + meansize), xt::all());
    m_var = xt::view(weights, xt::range(1 + meansize, xt::placeholders::_), xt::all());
}

xt::xarray<double> gauss_distribution::calc_pfc(const xt::xarray<double>& x, size_t _class)
{
    static const double pi = 3.14159265359;
    xt::xarray<double> product = xt::ones<double>( { x.shape()[0] } );
    xt::xarray<double> denom = xt::sqrt(xt::view(m_var, xt::all(), xt::range(_class, _class + 1)) * 2.0 * pi);
    xt::xarray<double> sqr = xt::square(x - xt::transpose(xt::view(m_mean, xt::all(), xt::range(_class, _class + 1))));
    xt::xarray<double> var2_inv = xt::linalg::pinv(2.0 * xt::transpose(xt::view(m_var, xt::all(), xt::range(_class, _class + 1))));
    xt::xarray<double> e = xt::exp((xt::linalg::dot(sqr, var2_inv) * -0.5));
    
    product.reshape({ product.shape()[0], 1 });
    for (auto feature = 0; feature < x.shape()[1]; feature++)
        product = product * (e / denom(feature, 0));
    return product;
}