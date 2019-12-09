#ifndef _PCA_H_
#define _PCA_H_

#include <xtensor/xarray.hpp>

class pca {
public:
    pca() = default;
    pca(const pca&) = delete;
    ~pca() = default;

    void fit(const xt::xarray<double>& x);
    xt::xarray<double> transform(const xt::xarray<double>& x);
    xt::xarray<double> transform(const xt::xarray<double>& x, double target_percentage);
    xt::xarray<double> transform(const xt::xarray<double>& x, size_t target_features);

private:
    xt::xarray<double> m_eigen_vector;
    xt::xarray<double> m_eigen_values;
    xt::xarray<double> m_cum_var;
};

#endif