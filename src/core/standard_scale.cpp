#include "standard_scale.h"
#include <xtensor/xview.hpp>
#include <xtensor/xreducer.hpp>

xt::xarray<double> standard_scale::transform(const  xt::xarray<double>& x)
{
    xt::xarray<double> mean = xt::mean(x, {0}, xt::evaluation_strategy::immediate);
    // TODO: use xt::stddev when pull request https://github.com/xtensor-stack/xtensor/pull/1627#issuecomment-558170772 has been merged into xtensor
    xt::xarray<double> stddev = xt::sqrt(xt::mean(xt::square(x - mean)));
    return (x - mean) / stddev;
}