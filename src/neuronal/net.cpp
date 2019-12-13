#include "net.h"
#include "dense_layer.h"
#include "backprop_layer.h"

#include <tuple>
#include <core/reverse_iterate.h>
#include <preprocessing/one_hot.h>
#include <xtensor/xsort.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <xtensor/xio.hpp>


net::net(solver_type stype, loss_type ltype)
    : m_solver_type(stype)
    , m_loss(loss_factory::create(ltype))
{
    register_param("max_iterations", 100);
    register_param("threshold", 0.001);
}

void net::add(std::shared_ptr<dense_layer> layer)
{
    m_layers.push_back(layer);
}

xt::xarray<double> net::predict(const xt::xarray<double>& x) const
{
    auto x_ = x;
    for (auto layer : m_layers)
        x_ = layer->forward(x_);
    return x_;
}

void net::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    xt::xarray<double> last_c;
    auto max_iter = static_cast<size_t>(get_param("max_iterations"));
    auto threshold = get_param("threshold");
    size_t iter = 0;

    // ensure y has correct format
    auto y_onehot = (y.dimension() > 1 && m_classes > 2 && y.shape()[1] == m_classes) || 
                    (m_layers.back()->neurons() > y.shape()[1]) 
                        ? y : one_hot::transform(y);

    // prepare solvers and backpropagation
    std::map<std::shared_ptr<dense_layer>, std::shared_ptr<backprop_layer>> backprop_layers;
    std::map<std::shared_ptr<dense_layer>, std::shared_ptr<gradient_descent>> layer_solvers;
    for (auto layer : reverse_iterate(m_layers)) {
        backprop_layers[layer] = std::make_shared<backprop_layer>(layer);
        layer_solvers[layer]   = solver_factory::create(m_solver_type);
    }

    // optimize layers
    while (true)
    {
        auto p = predict(x);
        auto g = m_loss->derivative(p, y);
        auto c = m_loss->cost(p, y);

        if (last_c.dimension() && xt::all(xt::abs(last_c - c) < threshold))
            break;
        last_c = c;

        for (auto layer : reverse_iterate(m_layers)) {
            layer_solvers[layer]->do_iteration(layer, g);
            g = backprop_layers[layer]->execute(g);
        }

        iter++;
        if (max_iter > 0 && iter >= max_iter)
            break;
    }
}

void net::set_weights(const xt::xarray<double>& weights)
{
}

xt::xarray<double> net::weights() const
{
    return xt::xarray<double>();
}