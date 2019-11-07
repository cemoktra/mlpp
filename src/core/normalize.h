#ifndef _NORMALIZE_H_
#define _NORMALIZE_H_

#include <xtensor/xarray.hpp>

class normalize
{
public:
    normalize() = delete;
    ~normalize() = delete;

    static xt::xarray<double> transform(const xt::xarray<double>& x);
};

#endif