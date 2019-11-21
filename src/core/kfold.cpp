#include "kfold.h"
#include <random>
#include <numeric>
#include <iostream>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

kfold::kfold(size_t k, bool shuffle)
    : m_k(k)
    , m_shuffle(shuffle)
    , m_blockSize(0)
{
}

size_t kfold::k()
{
    return m_k;
}

void kfold::split(size_t index, const xt::xarray<double>& x, const xt::xarray<double>& y, xt::xarray<double>& x_train, xt::xarray<double>& x_test, xt::xarray<double>& y_train, xt::xarray<double>& y_test)
{
    if (m_indices.size() != y.shape()[0]) {
        prepareIndices(y.shape()[0]);
        m_blockSize = ceil(static_cast<double>(y.shape()[0]) / m_k);
    }
    if (index >= m_k)
        throw std::out_of_range ("invalid split index");

    auto test_begin = std::next(m_indices.begin(), index * m_blockSize);
    auto test_end   = std::next(test_begin, m_blockSize - 1);
    test_end = std::min(test_end, m_indices.end());

    x_train.resize({ x.shape()[0] - m_blockSize, x.shape()[1] });
    y_train.resize({ x.shape()[0] - m_blockSize, y.shape()[1] });
    x_test.resize({ m_blockSize, x.shape()[1] });
    y_test.resize({ m_blockSize, y.shape()[1] });

    size_t train_idx = 0;
    size_t test_idx = 0;
    for (auto it = m_indices.begin(); it != m_indices.end(); ++it)
    {
        if (it < test_begin || it > test_end) {
            for (auto j = 0; j < x.shape()[1]; j++)
                x_train(train_idx, j) = x(*it, j);
            for (auto j = 0; j < y.shape()[1]; j++)
                y_train(train_idx, j) = y(*it, j);
            train_idx++;
        } else {
            for (auto j = 0; j < x.shape()[1]; j++)
                x_test(test_idx, j) = x(*it, j);
            for (auto j = 0; j < y.shape()[1]; j++)
                y_test(test_idx, j) = y(*it, j);
            test_idx++;
        }
    }
}

void kfold::prepareIndices(size_t count)
{
    m_indices.resize(count);
    std::iota(m_indices.begin(), m_indices.end(), 0);

    if (m_shuffle) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(m_indices.begin(), m_indices.end(), gen);
    }
}