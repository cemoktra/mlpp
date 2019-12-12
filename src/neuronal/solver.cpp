#include "solver.h"
#include "dense_layer.h"
#include "loss.h"

#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>


gradient_descent::gradient_descent(double learning_rate) 
    : m_rate(learning_rate)
    , m_iteration(0) 
{};

void gradient_descent::reset() { 
    m_iteration = 0; 
}

void gradient_descent::do_iteration(std::shared_ptr<dense_layer> layer, const xt::xarray<double>& g)
{
    m_iteration++;

    auto gw = xt::transpose(xt::linalg::dot(xt::transpose(layer->input()), g));
    auto w = layer->weights();

    layer->update_weights(w - (m_rate * gw / layer->input().shape()[0]));
}


stochastic_gradient_descent::stochastic_gradient_descent(double learning_rate, size_t batches) 
    : gradient_descent(learning_rate)
    , m_batches(batches)
{};

void stochastic_gradient_descent::do_iteration(std::shared_ptr<dense_layer> layer, const xt::xarray<double>& g)
{
    m_iteration++;

    auto indices = xt::random::permutation(layer->input().shape()[0]);
    auto batch_size = static_cast<size_t>(ceil( layer->input().shape()[0] / static_cast<double>(m_batches)));
    auto fullX = layer->input();

    for (auto i = 0; i < layer->input().shape()[0]; i += batch_size) {
        auto batch_indices = xt::view(indices, xt::range(i, i + batch_size));
        auto batch_X = xt::view(fullX, xt::keep(batch_indices), xt::all());
        auto batch_g = xt::view(g, xt::keep(batch_indices), xt::all());
        
        layer->update_input(batch_X);
        auto gw = xt::transpose(xt::linalg::dot(xt::transpose(layer->input()), batch_g));
        layer->update_weights(layer->weights() - (m_rate * gw / layer->input().shape()[0]));
    }

    layer->update_input(fullX);
}


ada_grad_decay::ada_grad_decay(double learning_rate, double decay, double epsilon) 
    : gradient_descent(learning_rate)
    , m_decay(decay)
    , m_epsilon(epsilon)
{};

void ada_grad_decay::reset() {
    gradient_descent::reset();
    m_grad_history = xt::xarray<double>();
}

void ada_grad_decay::do_iteration(std::shared_ptr<dense_layer> layer, const xt::xarray<double>& g)
{
    m_iteration++;

    auto gw = xt::transpose(xt::linalg::dot(xt::transpose(layer->input()), g));
    auto w = layer->weights();

    // update history
    m_grad_history = m_grad_history.dimension() == 0 ? xt::eval(xt::square(gw)) : xt::eval(m_decay * m_grad_history + (1 - m_decay) * xt::square(gw));

    // update weights
    layer->update_weights(w - m_rate * gw / (xt::sqrt(m_grad_history) + m_epsilon));
}


adam::adam(double learning_rate, double beta1, double beta2, double epsilon) 
    : gradient_descent(learning_rate)
    , m_beta1(beta1)
    , m_beta2(beta2)
    , m_epsilon(epsilon)
{};

void adam::reset() {
    gradient_descent::reset();
    m = xt::xarray<double>();
    v = xt::xarray<double>();
}

void adam::do_iteration(std::shared_ptr<dense_layer> layer, const xt::xarray<double>& g)
{
    m_iteration++;

    auto gw = xt::transpose(xt::linalg::dot(xt::transpose(layer->input()), g));
    auto w = layer->weights();

    // update moments
    if (m.dimension() == 0) {
        m = (1 - m_beta1) * gw;
        v = (1 - m_beta2) * xt::square(gw);
    } else {
        m = m_beta1 * m + (1 - m_beta1) * gw;
        v = m_beta2 * v + (1 - m_beta2) * xt::square(gw);
    }

    auto mHat = m / (1 - pow(m_beta1, m_iteration));
    auto vHat = v / (1 - pow(m_beta2, m_iteration));
    auto s1 = mHat.shape();

    layer->update_weights(w - m_rate * mHat / (xt::sqrt(vHat) + m_epsilon));
}


adamax::adamax(double learning_rate, double beta1, double beta2) 
    : gradient_descent(learning_rate)
    , m_beta1(beta1)
    , m_beta2(beta2)
{};

void adamax::reset() {
    gradient_descent::reset();
    m = xt::xarray<double>();
    u = xt::xarray<double>();
}

void adamax::do_iteration(std::shared_ptr<dense_layer> layer, const xt::xarray<double>& g)
{
    m_iteration++;

    auto gw = xt::transpose(xt::linalg::dot(xt::transpose(layer->input()), g));
    auto w = layer->weights();

    // update moments
    if (m.dimension() == 0) {
        m = (1 - m_beta1) * gw;
        u = xt::abs(gw);
    } else {
        m = m_beta1 * m + (1 - m_beta1) * gw;
        u = xt::maximum(m_beta2 * u, xt::abs(gw));
    }

    auto mHat = m / (1 - pow(m_beta1, m_iteration));

    layer->update_weights(w - (m_rate * mHat / u));
}