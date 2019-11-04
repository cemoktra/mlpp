#ifndef _DECISION_TREE_NODE_H_
#define _DECISION_TREE_NODE_H_

#include <Eigen/Dense>

class decision_tree_node
{
public:
    decision_tree_node(size_t layer, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, size_t classes, decision_tree_node *parent, bool positives);
    ~decision_tree_node();
    
    double entropy() const;

    void split(size_t max_depth = 0, size_t min_leaf_items = 1, size_t randomly_ignored_features = 0);
    size_t count();

    size_t decide(const Eigen::VectorXd& x);

protected:
    bool filter(const Eigen::VectorXd& row);

private:
    void init();
    
    Eigen::MatrixXd m_x;
    Eigen::MatrixXd m_y;

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