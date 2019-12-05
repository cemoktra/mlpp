#include "standard_scale.h"

void standard_scale::fit(const xt::xarray<double>& x)
{
    m_mean = xt::mean(x, {0});
    m_stddev = xt::stddev(x, {0});
}

xt::xarray<double> standard_scale::transform(const xt::xarray<double>& x)
{
    xt::xarray<double> scaled = (x - m_mean) / m_stddev;
    scaled = xt::where(xt::isnan(scaled), 0.0, scaled);
    return scaled;
}
