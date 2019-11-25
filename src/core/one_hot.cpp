
#include "one_hot.h"
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include "xtensor/xindex_view.hpp"

xt::xarray<double> one_hot::transform(const xt::xarray<double>& x, size_t classes)
{
    xt::xarray<double> result (std::vector<size_t>({x.shape()[0], classes}));
    for (auto i = 0; i < classes; i++) 
        for (auto r = 0; r < x.shape()[0]; r++) 
            result(r, i) = x(r) == i ? 1 : 0;  

    return result;
}