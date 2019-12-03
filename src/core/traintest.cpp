#include "traintest.h"
#include <xtensor/xrandom.hpp>

void train_test_split::init(size_t rows, double test_proportion, bool shuffle)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    size_t test_size  = rows * test_proportion;
    size_t train_size = rows - test_size;
    
    auto indices = xt::eval(xt::arange<int64_t>(rows));
    if (shuffle) 
        xt::random::shuffle(indices, gen);

    m_train_indices = xt::view(indices, xt::range(0, train_size));
    m_test_indices  = xt::view(indices, xt::range(train_size, xt::placeholders::_));
}

std::pair<xt::xarray<double>, xt::xarray<double>> train_test_split::split(const xt::xarray<double>& x)
{
    return std::make_pair<>(xt::eval(xt::view(x, xt::keep(m_train_indices), xt::all())),
                            xt::eval(xt::view(x, xt::keep(m_test_indices), xt::all())));
}

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> train_test_split::split(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    auto [x_train, x_test] = split(x);
    auto [y_train, y_test] = split(y);
    return std::make_tuple<>(x_train, x_test, y_train, y_test);
}

xt::xarray<int64_t> train_test_split::train_indices() const
{
    return m_train_indices;
}

xt::xarray<int64_t> train_test_split::test_indices() const
{
    return m_test_indices;
}
