#include "decision_tree.h"
#include "decision_tree_node.h"
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>

decision_tree::decision_tree()
    : m_root(nullptr)
{
    register_param("max_depth", 256);
    register_param("min_leaf_items", 1);
    register_param("ignored_features", 0);
}

decision_tree::~decision_tree()
{
    delete m_root;
}

xt::xarray<double> decision_tree::predict(const xt::xarray<double>& x)
{
    xt::xarray<double> result = xt::zeros<double>(std::vector<size_t>({x.shape()[0], m_classes}));

    for (auto r = 0; r < x.shape()[0]; r++)
        result (r, m_root->decide(xt::view(x, xt::range(r, r + 1), xt::all()))) = 1.0;
    return result;
}

void decision_tree::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    m_root = new decision_tree_node(0, x, y, m_classes, nullptr, true);
    m_root->split(static_cast<size_t>(get_param("max_depth")), static_cast<size_t>(get_param("min_leaf_items")), static_cast<size_t>(get_param("ignored_features")));
}

void decision_tree::set_weights(const xt::xarray<double>& weights)
{
}

xt::xarray<double> decision_tree::weights()
{
    return xt::xarray<double>();
}