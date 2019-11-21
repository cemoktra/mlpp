#include "standard_scale.h"
#include <xtensor/xview.hpp>

xt::xarray<double> standard_scale::transform(const  xt::xarray<double>& x)
{
    auto mean = xt::eval(xt::mean(x, {0}));
    auto stddev = xt::eval(xt::stddev(x, {0}));
    return (x - mean) / stddev;
}