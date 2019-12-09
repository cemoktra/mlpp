#include "polyfeatures.h"
#include <numeric>
#include <algorithm>
#include <xtensor/xview.hpp>

polynomial_features::polynomial_features(size_t degree, bool bias)
    : m_degree(degree)
    , m_bias(bias)
{
}

xt::xarray<double> polynomial_features::transform(const xt::xarray<double> &x)
{
    xt::xarray<double> result = x;

    std::vector<double> polynom(x.shape()[1]);
    for (auto i = 2u; i <= m_degree; i++) {
        std::fill(polynom.begin(), polynom.end(), 0.0);

        int digit = 0;
        while (true) {
            if (i == std::accumulate(polynom.begin(), polynom.end(), 0)) {
                // create data
                xt::xarray<double> new_result = xt::ones<double>({ result.shape()[0], result.shape()[1] + 1});
                xt::view(new_result, xt::all(), xt::range(0, result.shape()[1])) = result;
                auto new_col = xt::view(new_result, xt::all(), xt::range(result.shape()[1], xt::placeholders::_));
                for (auto k = 0; k < polynom.size(); k++) {
                    xt::xarray<double> p = xt::pow(xt::view(new_result, xt::all(), xt::range(k, k + 1)), polynom[k]);
                    new_col = new_col * p;
                }
                result = new_result;
            }

            digit = 0;
            while (digit < x.shape()[1]) {
                polynom[digit]++;
                if (polynom[digit] > i) {
                    polynom[digit++] = 0;
                } else
                    break;
            }
            if (digit >= x.shape()[1])
                break;
        }
    }

    if (m_bias) {
        xt::xarray<double> new_result = xt::ones<double>({ result.shape()[0], result.shape()[1] + 1});
        xt::view(new_result, xt::all(), xt::range(0, result.shape()[1])) = result;
        result = new_result;
    }

    return result;
}
