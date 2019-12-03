#include "standard_scale.h"

xt::xarray<double> standard_scale::transform(const xt::xarray<double>& x)
{
    xt::xarray<double> mean = xt::mean(x, {0});
    xt::xarray<double> stddev = xt::stddev(x, {0});
    xt::xarray<double> scaled = (x - mean) / stddev;
    scaled = xt::where(xt::isnan(scaled), 0.0, scaled);
    return scaled;
}
