#include "scaler.h"

void standard_scaler::fit(const xt::xarray<double>& x)
{
    m_mean = xt::mean(x, {0});
    m_stddev = xt::stddev(x, {0});
}

xt::xarray<double> standard_scaler::transform(const xt::xarray<double>& x)
{
    xt::xarray<double> scaled = (x - m_mean) / m_stddev;
    scaled = xt::where(xt::isnan(scaled), 0.0, scaled);
    return scaled;
}

void normal_scaler::fit(const xt::xarray<double>& x)
{
    m_mean = xt::mean(x, {0});
    m_scale = xt::amax(xt::abs(x - m_mean), { 0 });
}

xt::xarray<double> normal_scaler::transform(const xt::xarray<double>& x)
{
    return (x - m_mean) / m_scale;
}
