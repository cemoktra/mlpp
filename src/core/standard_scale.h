#ifndef _STANDARD_SCALE_H_
#define _STANDARD_SCALE_H_

#include <xtensor/xarray.hpp>

class standard_scale
{
public:
    standard_scale() = default;
    ~standard_scale() = default;

    void fit(const  xt::xarray<double>& x);
    xt::xarray<double> transform(const  xt::xarray<double>& x);

private:
    xt::xarray<double> m_mean;
    xt::xarray<double> m_stddev;
};

#endif