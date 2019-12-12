// activation

//forward
//backward

#ifndef _DENSE_LAYER_H_
#define _DENSE_LAYER_H_

#include "activation.h"
#include "loss.h"
#include <xtensor/xarray.hpp>

class dense_layer {
public:
    dense_layer(size_t neurons, activation_func_t activation = sigmoid()); // TODO: add activation
    dense_layer(const dense_layer&) = delete;
    ~dense_layer() = default;    

    xt::xarray<double> forward(const xt::xarray<double>& X);
    xt::xarray<double> backward(const xt::xarray<double>& g);

    inline xt::xarray<double> weights() const { return m_w; }
    inline xt::xarray<double> input() { return m_X; }
    inline xt::xarray<double> output() { return m_y; }

    inline size_t neurons() { return m_neurons; }

    inline void update_weights(const xt::xarray<double>& w) { m_w = w; calculate(); }
    inline void update_input(const xt::xarray<double>& X) { m_X = X; calculate(); }

protected:
    void calculate();

    activation_func_t m_activation;
    size_t m_neurons;
    xt::xarray<double> m_w;
    xt::xarray<double> m_X;
    xt::xarray<double> m_y;
};

#endif
