#include "knn.h"
#include <algorithm>
#include <iterator>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor-blas/xlinalg.hpp>

knn::knn()
{
    register_param("k", 3);
}

xt::xarray<double> knn::predict(const xt::xarray<double>& x) const
{
    size_t k = static_cast<size_t>(get_param("k"));
    xt::xarray<double> result = xt::zeros<double>(std::vector<size_t>({x.shape()[0], m_classes}));

    for (auto i = 0; i < x.shape()[0]; i++)
    {
        xt::xarray<double> x_row = xt::view(x, xt::range(i, i + 1), xt::all());
        xt::xarray<double> distances = xt::sum(xt::square(m_x_train - x_row), { 1 });
        auto sorted_indices = xt::argsort(distances);
        auto best_classes = xt::eval(xt::index_view(m_y_train, sorted_indices));
        for (auto j = 0; j < k; j++)
            result(i, best_classes(j)) += 1.0;
    }
    return result / k;
}

void knn::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    m_x_train = x;
    m_y_train = y;
    m_y_train.reshape({m_y_train.shape()[0]});
}

void knn::set_weights(const xt::xarray<double>& weights)
{
    // TODO: weight may be a combined matrix of x_train and y_train
}

xt::xarray<double> knn::weights() const
{
    return xt::xarray<double>();
}