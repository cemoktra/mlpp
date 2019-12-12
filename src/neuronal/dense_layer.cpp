#include "dense_layer.h"
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>

dense_layer::dense_layer(size_t neurons, activation_func_t activation)
    : m_neurons(neurons)
    , m_activation(activation)
{
}

xt::xarray<double> dense_layer::forward(const xt::xarray<double>& X)
{
    update_input(X);
    return m_y;
}

xt::xarray<double> dense_layer::backward(const xt::xarray<double>& g)
{
    return g * (1.0 - output()) * output();
    // return xt::linalg::dot(g, m_w); // LINEAR / FULLY CONNECTED

// SIGMOID: grads[i] = chainGradT[i] * (1 - output[i]) * output[i];
// SOFTMAX: grads[i] = ((y - 1)) - std::exp(output[i]))*sum;
// LINEAR: xt::linalg::dot(g, m_w)
}

void dense_layer::calculate()
{
    if (m_X.dimension()) {
        if (!m_w.dimension())
            m_w = xt::random::rand<double>({m_neurons, m_X.shape()[1]}, -0.1, 0.1); // TODO: init seed or own generator    
        m_y = m_activation(xt::linalg::dot(m_X, xt::transpose(m_w)));
    }
}