#include "backprop_layer.h"

#include "dense_layer.h"

backprop_layer::backprop_layer(std::shared_ptr<dense_layer> layer)
    : m_layer(layer)
{
}

xt::xarray<double> backprop_layer::execute(const xt::xarray<double>& g)
{
    m_dy = g;
    if (!m_dX.dimension())
        m_dX = m_layer->activation_func()->revert(m_layer->input(), m_layer->output() * m_dy);
    else
        m_dX += m_layer->activation_func()->revert(m_layer->input(), m_layer->output() * m_dy);
    return m_dX;
}