#include "knn.h"
#include <algorithm>
#include <iterator>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor-blas/xlinalg.hpp>

knn::knn()
{
    register_param("k", 3);
}

xt::xarray<double> knn::predict(const xt::xarray<double>& x)
{
    xt::xarray<double> result = xt::zeros<double>(std::vector<size_t>({x.shape()[0], m_classes}));
    auto kneighbors = static_cast<size_t>(get_param("k"));
    
    for (auto i = 0; i < x.shape()[0]; i++)
    {
        m_kdistances.fill(std::numeric_limits<double>::max());
        m_class_count.fill(0);

        xt::xarray<double> x_row = xt::view(x, xt::range(i, i + 1), xt::all());

        for (auto j = 0; j < m_x_train.shape()[0]; j++)
        {
            xt::xarray<double> x_train_row = xt::view(m_x_train, xt::range(j, j + 1), xt::all()) - x_row;
            auto distance = xt::linalg::norm(x_train_row);

            if (distance < m_kdistances(kneighbors - 1)) {
                 m_kdistances(kneighbors - 1) = distance;
                 m_knearest_classes(kneighbors - 1) = static_cast<size_t>(m_y_train(j, 0));
            }
            
            auto idx = xt::argsort(m_kdistances);
            m_kdistances = xt::index_view(m_kdistances, idx);
            m_knearest_classes = xt::index_view(m_knearest_classes, idx);
        }
        
        for (auto k = 0; k < kneighbors; k++)
            m_class_count(m_knearest_classes(k))++;
        result(i, xt::argmax(m_class_count)(0)) = 1.0;
    }

    return result;
}

void knn::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    auto k = static_cast<size_t>(get_param("k"));

    m_kdistances       = xt::zeros<double>(std::vector<size_t>({k}));
    m_knearest_classes = xt::zeros<size_t>(std::vector<size_t>({k}));
    m_class_count      = xt::zeros<size_t>(std::vector<size_t>({m_classes}));
    m_x_train = x;
    m_y_train = y;
}

void knn::set_weights(const xt::xarray<double>& weights)
{
    // TODO: weight may be a combined matrix of x_train and y_train
}

xt::xarray<double> knn::weights()
{
    return xt::xarray<double>();
}