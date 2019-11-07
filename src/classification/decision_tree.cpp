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
    auto s = x.shape();
    s[1] = 1;
    xt::xarray<double> result (s);

    for (auto r = 0; r < x.shape()[0]; r++)
        result (r, 0) = m_root->decide(xt::view(x, xt::range(r, r + 1), xt::all()));
    return result;
}

double decision_tree::score(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    xt::xarray<double> p = predict(x);
    xt::xarray<size_t> target_class;
    
    if (y.shape()[1] > 1)
        target_class = xt::argmax(p, {1});
    else
        target_class = y;
    target_class.reshape(p.shape());
    return xt::sum(xt::equal(p, target_class))(0) / static_cast<double>(y.shape()[0]);
}

void decision_tree::init_classes(size_t number_of_classes)
{
    m_number_of_classes = number_of_classes;
}

void decision_tree::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    m_root = new decision_tree_node(0, x, y, m_number_of_classes, nullptr, true);
    m_root->split(static_cast<size_t>(get_param("max_depth")), static_cast<size_t>(get_param("min_leaf_items")), static_cast<size_t>(get_param("ignored_features")));
}

void decision_tree::set_weights(const xt::xarray<double>& weights)
{
}

xt::xarray<double> decision_tree::weights()
{
    return xt::xarray<double>();
}