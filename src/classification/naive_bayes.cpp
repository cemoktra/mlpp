#include "naive_bayes.h"
#include "distribution.h"
#include <core/one_hot.h>
#include <iostream>

naive_bayes::naive_bayes(std::shared_ptr<distribution> distribution)
    : m_distribution(distribution)
{
}

xt::xarray<double> naive_bayes::predict(const xt::xarray<double>& x) const
{    
    return m_distribution->predict(x);
}

void naive_bayes::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    auto y_onehot = (y.size() > 1 && y.shape()[1] == m_classes) ? y : one_hot::transform(y);
    m_distribution->calc_weights(x, y_onehot);
}

void naive_bayes::set_weights(const xt::xarray<double>& weights)
{
    m_distribution->set_weights(weights);
}

xt::xarray<double> naive_bayes::weights() const
{
    return m_distribution->weights();
}