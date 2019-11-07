
#include "normalize.h"

xt::xarray<double> normalize::transform(const xt::xarray<double>& x)
{
    xt::xarray<double> mean = xt::mean(x, { 0 });
    xt::xarray<double> result = x - mean;
    xt::xarray<double> min = xt::abs(xt::amin(result, { 0 }));
    xt::xarray<double> max = xt::amax(result, { 0 });
    xt::xarray<double> scale = xt::maximum(min, max);
    return result / scale;
}
