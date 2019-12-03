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
    auto k = static_cast<size_t>(get_param("k"));
    xt::xarray<double> result = xt::zeros<double>(std::vector<size_t>({x.shape()[0], m_classes}));
    auto kneighbors = static_cast<size_t>(get_param("k"));
    xt::xarray<double> kdistances = xt::zeros<double>(std::vector<size_t>({k}));
    xt::xarray<size_t> knearest_classes = xt::zeros<size_t>(std::vector<size_t>({k}));
    xt::xarray<size_t> class_count = xt::zeros<size_t>(std::vector<size_t>({m_classes}));

    for (auto i = 0; i < x.shape()[0]; i++)
    {
        kdistances.fill(std::numeric_limits<double>::max());
        class_count.fill(0);

        xt::xarray<double> x_row = xt::view(x, xt::range(i, i + 1), xt::all());

        for (auto j = 0; j < m_x_train.shape()[0]; j++)
        {
            xt::xarray<double> x_train_row = xt::view(m_x_train, xt::range(j, j + 1), xt::all()) - x_row;
            auto distance = xt::linalg::norm(x_train_row);

            if (distance < kdistances(kneighbors - 1)) {
                 kdistances(kneighbors - 1) = distance;
                 knearest_classes(kneighbors - 1) = static_cast<size_t>(m_y_train(j, 0));
            }
            
            auto idx = xt::argsort(kdistances);
            kdistances = xt::index_view(kdistances, idx);
            knearest_classes = xt::index_view(knearest_classes, idx);
        }
        
        for (auto k = 0; k < kneighbors; k++)
            class_count(knearest_classes(k))++;
        result(i, xt::argmax(class_count)(0)) = 1.0;
    }

    return result;
}

void knn::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    m_x_train = x;
    m_y_train = y;
}

void knn::set_weights(const xt::xarray<double>& weights)
{
    // TODO: weight may be a combined matrix of x_train and y_train
}

xt::xarray<double> knn::weights() const
{
    return xt::xarray<double>();
}