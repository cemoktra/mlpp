#include "net.h"
#include "dense_layer.h"

#include <tuple>
#include <core/reverse_iterate.h>
#include <preprocessing/one_hot.h>
#include <xtensor/xsort.hpp>

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
    m_layers.push_back(std::make_pair<>(layer, solver_factory::create(m_solver_type)));
}

xt::xarray<double> net::predict(const xt::xarray<double>& x) const
{
    auto x_ = x;
    for (auto [layer, solver] : m_layers)
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
                    (m_layers.back().first->neurons() > y.shape()[1]) 
                        ? y : one_hot::transform(y);

    // optimize layers
    while (true)
    {
        auto p = predict(x);
        auto g = m_loss->derivative(p, y);
        auto c = m_loss->cost(p, y);

        if (last_c.dimension() && xt::all(xt::abs(last_c - c) < threshold))
            break;
        last_c = c;

        // TODO: back propagation
        for (auto [layer, solver] : reverse_iterate(m_layers)) {
            solver->do_iteration(layer, g);
            g = layer->backward(g);
            
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