#include "kfold.h"
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

kfold::kfold(size_t k, bool shuffle)
    : m_k(k)
    , m_shuffle(shuffle)
{
    
}

size_t kfold::k()
{
    return m_k;
}

std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>> kfold::split(size_t index, const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    if (m_indices.size() != y.shape()[0])
        prepareIndices(y.shape()[0]);
    size_t block_size = ceil(static_cast<double>(y.shape()[0]) / m_k);

    auto test_indices = xt::view(m_indices, xt::range(index * block_size, std::min(index * block_size + block_size, y.shape()[0])));
    return std::make_tuple<>(
        xt::eval(xt::view(x, xt::keep(test_indices), xt::all())),
        xt::eval(xt::view(x, xt::drop(test_indices), xt::all())),
        xt::eval(xt::view(y, xt::keep(test_indices), xt::all())),
        xt::eval(xt::view(y, xt::drop(test_indices), xt::all()))
    );
}

void kfold::prepareIndices(size_t count)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    m_indices = xt::arange<int64_t>(count);
    if (m_shuffle) 
        xt::random::shuffle(m_indices, gen);    
}