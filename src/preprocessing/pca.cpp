#include "pca.h"
#include <xtensor/xsort.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor-blas/xlinalg.hpp>


void pca::fit(const xt::xarray<double>& x)
{
    auto covar_mat = xt::cov(xt::eval(xt::transpose(x)));
    auto [eig_val, eig_vec] = xt::linalg::eig(covar_mat);
    
    auto unsorted_evalues  = xt::eval(xt::real(eig_val));
    auto unsorted_evectors = xt::eval(xt::real(eig_vec));
    auto sorted_indices = xt::flip(xt::argsort(unsorted_evalues), 0);

    m_eigen_values = xt::index_view(unsorted_evalues, sorted_indices);
    m_eigen_vector = unsorted_evectors;
    size_t i = 0;

    for (auto idx : sorted_indices) {
        xt::view(m_eigen_vector, xt::range(i, i + 1), xt::all()) = xt::view(unsorted_evectors, xt::range(idx, idx + 1), xt::all());
        i++;
    }
    m_cum_var = xt::cumsum(m_eigen_values) / xt::sum(m_eigen_values);
}

xt::xarray<double> pca::transform(const xt::xarray<double>& x)
{
    return xt::transpose(xt::linalg::dot(m_eigen_vector, xt::transpose(x)));
}

xt::xarray<double> pca::transform(const xt::xarray<double>& x, double target_percentage)
{
    return transform(x, xt::flatten_indices(xt::where(m_cum_var > target_percentage))(0));
}

xt::xarray<double> pca::transform(const xt::xarray<double>& x, size_t target_features)
{
    return xt::view(transform(x), xt::all(), xt::range(0, target_features));
}