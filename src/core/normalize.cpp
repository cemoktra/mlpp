
#include "normalize.h"

void normalize::fit(const xt::xarray<double>& x)
{
    m_mean = xt::mean(x, {0});
    m_scale = xt::amax(xt::abs(x - m_mean), { 0 });
}

xt::xarray<double> normalize::transform(const xt::xarray<double>& x)
{
    return (x - m_mean) / m_scale;
}
