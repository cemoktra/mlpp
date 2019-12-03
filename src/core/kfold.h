#ifndef _KFOLD_H_
#define _KFOLD_H_

#include <xtensor/xarray.hpp>

class kfold {
public:
    kfold(size_t k, bool shuffle = true);
    ~kfold() = default;

    std::tuple<xt::xarray<double>, xt::xarray<double>, xt::xarray<double>, xt::xarray<double>>  split(size_t index, const xt::xarray<double>& x, const xt::xarray<double>& y);
    size_t k();

private:
    void prepareIndices(size_t count);

    size_t m_k;
    bool m_shuffle;
    xt::xarray<int64_t> m_indices;
};

#endif