#include "traintest.h"

#include <random>
#include <numeric>

void train_test_split::init(size_t rows, double test_proportion, bool shuffle)
{
    std::vector<size_t> indices(rows);

    std::iota(indices.begin(), indices.end(), 0);
    if (shuffle) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    size_t test_size = rows * test_proportion;
    size_t train_size = rows - test_size;
    
    m_test_indices = { indices.begin(), std::next(indices.begin(), test_size) };
    m_train_indices = { std::next(indices.begin(), test_size), indices.end() };
}

void train_test_split::split(const xt::xarray<double>& x, xt::xarray<double>& x_train, xt::xarray<double>& x_test)
{
    if (x.shape()[0] != m_train_indices.size() + m_test_indices.size())
        throw std::invalid_argument("vector x size does not match initialized size");

    size_t train_idx = 0;
    size_t test_idx = 0;

    x_train.resize({ m_train_indices.size(), x.shape()[1] });
    x_test.resize({ m_test_indices.size(), x.shape()[1] });

    for (auto i : m_train_indices)
    {
        for (auto j = 0; j < x.shape()[1]; j++)
            x_train(train_idx, j) = x(i, j);
        train_idx++;
    }
    for (auto i : m_test_indices)
    {
        for (auto j = 0; j < x.shape()[1]; j++)
            x_test(test_idx, j) = x(i, j);
        test_idx++;
    }
}


void train_test_split::split(const xt::xarray<double>& x, const xt::xarray<double>& y, xt::xarray<double>& x_train, xt::xarray<double>& x_test, xt::xarray<double>& y_train, xt::xarray<double>& y_test)
{
    if (x.shape()[0] != y.shape()[0])
        throw std::invalid_argument("vector x size does not match initialized size");

    split(x, x_train, x_test);
    split(y, y_train, y_test);
}

std::vector<size_t> train_test_split::train_indices() const
{
    return m_train_indices;
}

std::vector<size_t> train_test_split::test_indices() const
{
    return m_test_indices;
}
