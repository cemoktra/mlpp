#ifndef _NORMALIZE_H_
#define _NORMALIZE_H_

#include <xtensor/xarray.hpp>

class normalize
{
public:
    normalize() = default;
    ~normalize() = default;

    void fit(const  xt::xarray<double>& x);
    xt::xarray<double> transform(const  xt::xarray<double>& x);

private:
    xt::xarray<double> m_mean;
    xt::xarray<double> m_scale;
};

#endif