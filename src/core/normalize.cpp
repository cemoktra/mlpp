
#include "normalize.h"

xt::xarray<double> normalize::transform(const xt::xarray<double>& x)
{
    xt::xarray<double> result = xt::eval(x - xt::eval(xt::mean(x, { 0 })));
    auto minmax = xt::eval(xt::minmax(result));
    auto scale = std::max(std::fabs(minmax[0][0]), std::fabs(minmax[0][1]));
    return result / scale;
}