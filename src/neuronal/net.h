// optimizer
// loss

#ifndef _NET_H_
#define _NET_H_

#include "solver.h"
#include "loss.h"
#include <classification/classifier.h>
#include <xtensor/xarray.hpp>
#include <vector>
#include <memory>

class dense_layer;

class net : public classifier
{
public:
    net(solver_type stype, loss_type ltype);
    net(const net&) = delete;
    ~net() = default;

    void add(std::shared_ptr<dense_layer> layer);

    xt::xarray<double> predict(const xt::xarray<double>& x) const override;
    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;

    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() const override;

protected:
    solver_type m_solver_type;
    std::shared_ptr<loss> m_loss;
    std::vector<std::shared_ptr<dense_layer>> m_layers;
};


#endif