#ifndef _STANDARD_SCALE_H_
#define _STANDARD_SCALE_H_

#include <xtensor/xarray.hpp>

class standard_scale
{
public:
    standard_scale() = delete;
    ~standard_scale() = delete;

    static xt::xarray<double> transform(const  xt::xarray<double>& x);
};

#endif