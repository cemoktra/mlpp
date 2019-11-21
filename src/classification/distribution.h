#ifndef _DISTRIBUTION_H_
#define _DISTRIBUTION_H_

#include <xtensor/xarray.hpp>

class distribution
{
public:
    distribution() = default;
    distribution(const distribution&) = delete;
    ~distribution() = default;

    virtual void calc_weights(const xt::xarray<double>& x, const xt::xarray<double>& y) = 0;
    virtual xt::xarray<double> predict(const xt::xarray<double>& x) = 0;
    
    virtual xt::xarray<double> weights() = 0;
    virtual void set_weights(const xt::xarray<double>& weights) = 0;
};

#endif