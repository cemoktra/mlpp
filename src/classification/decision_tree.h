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

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void init_classes(size_t number_of_classes) override;

    void set_weights(const Eigen::MatrixXd& weights) override;
    Eigen::MatrixXd weights() override;

private:
    size_t m_number_of_classes;
    decision_tree_node *m_root;
};

#endif