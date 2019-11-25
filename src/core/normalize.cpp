
#include "normalize.h"
#include <xtensor/xreducer.hpp>

xt::xarray<double> normalize::transform(const xt::xarray<double>& x)
{
    xt::xarray<double> mean = xt::mean(x, { 0 }, xt::evaluation_strategy::immediate);
    xt::xarray<double> result = x - mean;
    xt::xarray<double> min = xt::abs(xt::amin(result, { 0 }, xt::evaluation_strategy::immediate));
    xt::xarray<double> max = xt::amax(result, { 0 }, xt::evaluation_strategy::immediate);
    xt::xarray<double> scale = xt::maximum(min, max);
    return result / scale;
}
