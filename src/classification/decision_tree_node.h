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
    decision_tree_node(size_t layer, const xt::xarray<double>& x, const xt::xarray<double>& y, size_t classes, decision_tree_node *parent, bool positives);
    ~decision_tree_node();
    
    double entropy() const;

    void split(size_t max_depth = 0, size_t min_leaf_items = 1, size_t randomly_ignored_features = 0);
    size_t count();

    size_t decide(const xt::xarray<double>& x);

protected:
    bool filter(const xt::xarray<double>& row);

private:
    void init();
    
    xt::xarray<double> m_x;
    xt::xarray<double> m_y;

    double m_entropy;
    size_t m_classes;
    size_t m_class;
    size_t *m_class_counts;
    double *m_class_propabilities;
    size_t m_layer;
    size_t m_item_count;

    bool m_positives;
    size_t m_split_feature;
    double m_split_threshold;

    decision_tree_node *m_parent;
    decision_tree_node *m_child[2];
};

#endif