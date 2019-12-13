#include "dense_layer.h"
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>

dense_layer::dense_layer(size_t neurons, std::shared_ptr<activation> a)
    : m_neurons(neurons)
    , m_activation(a)
{
}

xt::xarray<double> dense_layer::forward(const xt::xarray<double>& X)
{
    update_input(X);
    return m_y;
}

void dense_layer::calculate()
{
    if (m_X.dimension()) {
        if (!m_w.dimension()) {
            std::random_device rd;
            std::mt19937 gen(rd());
            m_w = xt::random::rand<double>({m_neurons, m_X.shape()[1]}, -0.1, 0.1, gen);
        }
        m_y = m_activation->apply(xt::linalg::dot(m_X, xt::transpose(m_w)));
    }
}