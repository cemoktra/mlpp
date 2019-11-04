#ifndef _RANDOM_FOREST_H_
#define _RANDOM_FOREST_H_

#include "classifier.h"

class decision_tree;

class random_forest : public classifier
{
public:
    random_forest(size_t trees, size_t max_depth, size_t min_leaf_items, size_t ignores_features = 1);
    ~random_forest();

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::vector<std::string>& classes, size_t maxIterations = 0) override;

    void set_weights(const Eigen::MatrixXd& weights) override;
    Eigen::MatrixXd weights() override;

private:
    decision_tree **m_trees;
    size_t m_tree_count;
    size_t m_classes;
};

#endif