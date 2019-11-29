#include "standard_scale.h"
#include <xtensor/xview.hpp>
#include <xtensor/xreducer.hpp>

xt::xarray<double> standard_scale::transform(const  xt::xarray<double>& x)
{
    xt::xarray<double> mean = xt::mean(x, {0}, xt::evaluation_strategy::immediate);
    xt::xarray<double> stddev = xt::stddev(x, {0});
    return (x - mean) / stddev;
}