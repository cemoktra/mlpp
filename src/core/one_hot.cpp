
#include "one_hot.h"
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include "xtensor/xindex_view.hpp"

xt::xarray<double> one_hot::transform(const xt::xarray<double>& x, size_t classes)
{
    xt::xarray<int> int_classes = x;
    return xt::view(xt::eye(classes), xt::keep(int_classes));
}