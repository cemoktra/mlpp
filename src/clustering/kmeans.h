#ifndef _KMEANS_H_
#define _KMEANS_H_

#include <core/parameters.h>
#include <xtensor/xarray.hpp>

class kmeans : public parameters {
public:
    kmeans();
    kmeans(const kmeans&) = delete;
    ~kmeans() = default;

    void fit(const xt::xarray<double>& X);

private:
    xt::xarray<size_t> m_labels;
    xt::xarray<double> m_centers;
    xt::xarray<double> m_inertia;
};

#endif