#ifndef _ONEHOT_H_
#define _ONEHOT_H_

#include <xtensor/xarray.hpp>

class one_hot
{
public:
    one_hot() = delete;
    ~one_hot() = delete;

    static xt::xarray<double> transform(const xt::xarray<double>& y);
};

#endif