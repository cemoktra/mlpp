#ifndef _RANDOM_FOREST_H_
#define _RANDOM_FOREST_H_

#include "classifier.h"
#include <core/parameters.h>

class decision_tree;

class random_forest : public classifier
{
public:
    random_forest();
    ~random_forest();

    xt::xarray<double> predict(const xt::xarray<double>& x) override;
    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;

    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() override;

    void set_param(const std::string& name, double new_value) override;

private:
    decision_tree **m_trees;
};

#endif