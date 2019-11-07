#include "knn.h"
#include <algorithm>
#include <iterator>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>

knn::knn()
    : m_classes(0)
    , m_kdistances(nullptr)
    , m_knearest_classes(nullptr)
{
    register_param("k", 3);
    m_class_count = nullptr;
}

knn::~knn()
{
    delete [] m_kdistances;
    delete [] m_knearest_classes;
    delete [] m_class_count;
}

xt::xarray<double> knn::predict(const xt::xarray<double>& x)
{
    auto shape = x.shape();
    shape[1] = 1;
    xt::xarray<double> result (shape);
    auto kneighbors = static_cast<size_t>(get_param("k"));

    for (auto i = 0; i < x.shape()[0]; i++)
    {
        xt::xarray<double> x_row = xt::view(x, xt::range(i, i + 1), xt::all());
        for (auto k = 0; k < kneighbors; k++)
            m_kdistances[k] = std::numeric_limits<double>::max();
        memset(m_class_count, 0, m_classes * sizeof(size_t));
        
        for (auto j = 0; j < m_x_train.shape()[0]; j++)
        {
            xt::xarray<double> x_train_row = xt::view(m_x_train, xt::range(j, j + 1), xt::all());
            auto distance = xt::linalg::norm(x_train_row - x_row);
            
            if (distance < m_kdistances[kneighbors - 1]) {
                m_kdistances[kneighbors - 1] = distance;
                m_knearest_classes[kneighbors - 1] = static_cast<size_t>(m_y_train(i, 0));
            }

            for (auto k = kneighbors - 1; k > 0; k--)
            {
                if (m_kdistances[k] < m_kdistances[k - 1]) {
                    std::swap(m_kdistances[k], m_kdistances[k - 1]);
                    std::swap(m_knearest_classes[k], m_knearest_classes[k - 1]);
                }
            }
        }

        for (auto k = 0; k < kneighbors; k++)
            m_class_count[m_knearest_classes[k]]++;
        auto max_elem = std::max_element(m_class_count, m_class_count + m_classes);
        result(i, 0) = max_elem - m_class_count;
    }

    return result;
}

double knn::score(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    size_t pos = 0, neg = 0;
    auto p = predict(x);

    for (auto i = 0; i < p.shape()[0]; i++) {
        if (p(i, 0) == y(i, 0))
            pos++; 
        else
            neg++;
    }

    return static_cast<double>(pos) / static_cast<double>(pos + neg);
}

void knn::init_classes(size_t number_of_classes)
{
    m_classes = number_of_classes;
}

void knn::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    auto k = static_cast<size_t>(get_param("k"));

    m_kdistances = new double[k];
    m_knearest_classes = new size_t[k];

    m_class_count = new size_t[m_classes];
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