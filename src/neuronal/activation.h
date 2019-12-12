// linear
// sigmoid
// softmax

#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_


#include <xtensor/xarray.hpp>
#include <functional>

typedef std::function<xt::xarray<double>(const xt::xarray<double>&)> activation_func_t;

class linear {
public:
    xt::xarray<double> operator()(const xt::xarray<double>& X) const {
        return X;
    }
};

class sigmoid {
public:
    xt::xarray<double> operator()(const xt::xarray<double>& X) const {
        return 1.0 / (xt::exp(-X) + 1);
    }
};

class softmax {
public:
    xt::xarray<double> operator()(const xt::xarray<double>& X) const {
        xt::xarray<double> e = xt::exp(X);
        xt::xarray<double> sum = xt::sum(e, { 1 });
        sum.reshape({ sum.shape()[0], 1 });
        return e / sum;
    }
};

#endif