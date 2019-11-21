#ifndef _MODEL_H_
#define _MODEL_H_

#include "parameters.h"
#include <xtensor/xarray.hpp>

class model : public parameters {
public:
    model() = default;
    model(const model&) = delete;
    ~model() = default;

    virtual xt::xarray<double> predict(const xt::xarray<double>& x) = 0;
    virtual double score(const xt::xarray<double>& x, const xt::xarray<double>& y) = 0;
    virtual void train(const xt::xarray<double>& x, const xt::xarray<double>& y) = 0;

    virtual void set_weights(const xt::xarray<double>& weights) = 0;
    virtual xt::xarray<double> weights() = 0;
};

#endif