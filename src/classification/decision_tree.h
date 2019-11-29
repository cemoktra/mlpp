#ifndef _DECISION_TREE_H_
#define _DECISION_TREE_H_

#include "classifier.h"
#include <core/parameters.h>

class decision_tree_node;

class decision_tree : public classifier
{
public:
    decision_tree();
    decision_tree(const decision_tree&) = delete;
    ~decision_tree();

    xt::xarray<double> predict(const xt::xarray<double>& x) override;
    void train(const xt::xarray<double>& x, const xt::xarray<double>& y) override;
    void set_weights(const xt::xarray<double>& weights) override;
    xt::xarray<double> weights() override;

private:
    decision_tree_node *m_root;
};

#endif