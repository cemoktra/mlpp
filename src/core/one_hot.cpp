
#include "one_hot.h"
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

xt::xarray<double> one_hot::transform(const xt::xarray<double>& x, size_t classes)
{
    auto shape = x.shape();
    shape[1] = classes;
    xt::xarray<double> result ( shape );
    for (auto i = 0; i < classes; i++)
        xt::view(result, xt::all(), xt::range(i, i + 1)) = xt::where(xt::equal(x, i), 1, 0);
    return result;
}