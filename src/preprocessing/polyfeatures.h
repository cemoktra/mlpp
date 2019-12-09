#ifndef _POLYFEATURES_H_
#define _POLYFEATURES_H_

#include <xtensor/xarray.hpp>

class polynomial_features
{
public:
    polynomial_features(size_t degree = 2, bool bias = false);
    ~polynomial_features() = default;

    xt::xarray<double> transform(const xt::xarray<double> &x);

private:
    size_t m_degree;
    bool m_bias;
};

#endif