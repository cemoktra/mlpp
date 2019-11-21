#ifndef _KFOLD_H_
#define _KFOLD_H_

#include <xtensor/xarray.hpp>

class kfold {
public:
    kfold(size_t k, bool shuffle = true);
    ~kfold() = default;

    void split(size_t index, const xt::xarray<double>& x, const xt::xarray<double>& y, xt::xarray<double>& x_train, xt::xarray<double>& x_test, xt::xarray<double>& y_train, xt::xarray<double>& y_test);
    size_t k();

private:
    void prepareIndices(size_t count);

    size_t m_k;
    size_t m_blockSize;
    bool m_shuffle;
    std::vector<size_t> m_indices;
};

#endif