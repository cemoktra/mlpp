#include "naive_bayes.h"
#include "distribution.h"
#include <core/one_hot.h>
#include <iostream>
#include <xtensor/xsort.hpp>
#include <xtensor/xio.hpp>

naive_bayes::naive_bayes(std::shared_ptr<distribution> distribution)
    : m_distribution(distribution)
{
}

xt::xarray<double> naive_bayes::predict(const xt::xarray<double>& x)
{    
    return m_distribution->predict(x);
}

double naive_bayes::score(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    xt::xarray<double> p = predict(x);
    xt::xarray<size_t> predict_class = xt::argmax(p, {1});
    xt::xarray<size_t> target_class;    
    
    if (y.shape()[1] > 1)
        target_class = xt::argmax(p, {1});
    else
        target_class = y;
    target_class.reshape(predict_class.shape());
    
    return xt::sum(xt::equal(predict_class, target_class))(0) / static_cast<double>(y.shape()[0]);
}

void naive_bayes::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    auto y_onehot = (y.shape()[1] == m_number_of_classes) ? y : one_hot::transform(y, m_number_of_classes);
    m_distribution->calc_weights(x, y_onehot);
}

void naive_bayes::init_classes(size_t number_of_classes)
{
    m_number_of_classes = number_of_classes;
}

void naive_bayes::set_weights(const xt::xarray<double>& weights)
{
    // m_pre_prop = weights.row(0);
    // m_mean = weights.block(1, 0, (weights.rows() - 1) / 2, weights.cols());
    // m_var = weights.block(1 + m_mean.rows(), 0, (weights.rows() - 1) / 2, weights.cols());
}

xt::xarray<double> naive_bayes::weights()
{
    return xt::xarray<double>();
    // xt::xarray<double> weights (2 * m_mean.rows() + 1, m_mean.cols());
    // weights.row(0) = m_pre_prop;
    // weights.block(1, 0, m_mean.rows(), m_mean.cols()) = m_mean;
    // weights.block(1 + m_mean.rows(), 0, m_var.rows(), m_var.cols()) = m_var;
    // return weights;
}