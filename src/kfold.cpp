#include "kfold.h"

#include <numeric>
#include <random>
#include <algorithm>

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

void kfold::split(size_t index, const std::vector<std::vector<double>>& x, const std::vector<double>& y, std::vector<std::vector<double>>& x_train, std::vector<std::vector<double>>& x_test, std::vector<double>& y_train, std::vector<double>& y_test)
{
    if (m_indices.size() != y.size()) {
        prepareIndices(y.size());
        m_blockSize = ceil(static_cast<double>(y.size()) / m_k);
    }
    if (index >= m_k)
        throw std::out_of_range ("invalid split index");

    auto test_begin = std::next(m_indices.begin() + index * m_blockSize);
    auto test_end   = std::next(test_begin + m_blockSize - 1);
    test_end = std::min(test_end, m_indices.end());

    x_train.clear();
    x_test.clear();
    y_train.clear();
    y_test.clear();
    x_train.resize(x.size());    
    x_test.resize(x.size());

    for (auto it = m_indices.begin(); it != m_indices.end(); ++it)
    {
        if (it < test_begin || it >= test_end) {
            for (auto j = 0; j < x.size(); j++) {
                x_train[j].push_back(x[j][*it]);
            }
            y_train.push_back(y[*it]);
        } else {
            for (auto j = 0; j < x.size(); j++) {
                x_test[j].push_back(x[j][*it]);
            }
            y_test.push_back(y[*it]);
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