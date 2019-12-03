
#include "one_hot.h"
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>

xt::xarray<double> one_hot::transform(const xt::xarray<double>& y)
{
    return xt::view(xt::eye(xt::unique(y).shape()[0]), xt::keep(y));
}