#include "binomial_distribution.h"
#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
// #include <xtensor-blas/xlinalg.hpp>

void binomial_distribution::calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    m_pre_prop.resize( { y.shape()[1] } );
    m_smooth.resize( { y.shape()[1] } );
    m_feature_prop.resize( { x.shape()[1], y.shape()[1] } );

    for (auto cls = 0; cls < y.shape()[1]; cls++) {
        auto cls_col = xt::eval(xt::view(y, xt::all(), xt::range(cls, cls + 1)));
        cls_col.reshape({cls_col.shape()[0]});
        auto idx = xt::flatten_indices(xt::argwhere(cls_col > 0.0));
        auto class_features = xt::eval(xt::sum(xt::view(x, xt::keep(idx), xt::all()), {0}));
        class_features = xt::eval(xt::log((class_features + 1.0) / (class_features.shape()[0] + y.shape()[0])));
        xt::view(m_feature_prop, xt::all(), xt::range(cls, cls + 1)) = class_features;

        m_pre_prop(cls) = xt::sum(cls_col)(0) / y.shape()[0];
        m_smooth(cls) = log(1.0 / (class_features.shape()[0] + y.shape()[0]));
    }    
}

xt::xarray<double> binomial_distribution::predict(const xt::xarray<double>& x)
{
    auto s = x.shape();
    s[1] = m_pre_prop.shape()[0];
    xt::xarray<double> result (s);

    for (auto r = 0; r < x.shape()[0]; r++) {
        xt::view(result, xt::range(r, r + 1), xt::all()) = m_pre_prop;

        for (auto c = 0; c < m_pre_prop.shape()[0]; c++) {
            for (auto feature = 0; feature < x.shape()[1]; feature++) {
                if (x(r, feature) > 0.0) {
                    if (m_feature_prop(feature, c) > 0.0)
                        result(r, c) += (x(r, feature) * m_feature_prop(feature, c));
                    else {
                        bool add_smoothing = false;
                        for (auto c2 = 0; c2 < m_pre_prop.size(); c2++) {
                            if (c2 == c)
                                continue;
                            if (m_feature_prop(feature, c2) > 0.0)
                                add_smoothing = true;
                        }
                        if (add_smoothing)
                            result(r, c) += (x(r, feature) * m_smooth(c));
                    }
                }
            }
        }
    }
    return result;
}

xt::xarray<double> binomial_distribution::weights()
{
    auto s = m_feature_prop.shape();
    s[0] += 2;
    xt::xarray<double> weights (s);
    xt::view(weights, xt::range(0, 1), xt::all()) = m_pre_prop;
    xt::view(weights, xt::range(1, 2), xt::all()) = m_smooth;
    xt::view(weights, xt::range(2, xt::placeholders::_), xt::all()) = m_feature_prop;
    return weights;
}

void binomial_distribution::set_weights(const xt::xarray<double>& weights)
{
    m_pre_prop = xt::view(weights, xt::range(0, 1), xt::all());
    m_smooth = xt::view(weights, xt::range(1, 2), xt::all());
    m_feature_prop = xt::view(weights, xt::range(2, xt::placeholders::_), xt::all());    
}
