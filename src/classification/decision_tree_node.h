#ifndef _DECISION_TREE_NODE_H_
#define _DECISION_TREE_NODE_H_

#include <xtensor/xarray.hpp>
#include <stdexcept>

class invalid_data_exception : public std::exception
{
public:
    virtual char const* what() const noexcept override  { return "Decision tree requires normalized data."; }
};

class decision_tree_node
{
public:
    decision_tree_node(const xt::xarray<double>& x, const xt::xarray<double>& y, const xt::xarray<size_t>& node_indices, size_t layer, size_t max_depth = 0, size_t min_leaf_items = 1, size_t randomly_ignored_features = 0);
    decision_tree_node(const decision_tree_node&) = delete;
    ~decision_tree_node();
    
    double entropy() const;

    
    size_t count() const;

    size_t decide(const xt::xarray<double>& x) const;

protected:
    bool filter(const xt::xarray<double>& row);

private:
    void split(const xt::xarray<double>& x, const xt::xarray<double>& y, const xt::xarray<size_t>& node_indices, size_t max_depth, size_t min_leaf_items, size_t randomly_ignored_features);
    double calc_entropy(const xt::xarray<double>& x, const xt::xarray<double>& y, const xt::xarray<size_t>& node_indices);
    
    double m_entropy;
    size_t m_class;
    size_t m_layer;

    size_t m_split_feature;
    double m_split_threshold;

    decision_tree_node *m_child[2];
};

#endif