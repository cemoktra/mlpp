#ifndef _DECISION_TREE_H_
#define _DECISION_TREE_H_

#include "classifier.h"

class decision_tree_node;

class decision_tree : public classifier
{
public:
    decision_tree();
    decision_tree(const decision_tree&) = delete;
    ~decision_tree();

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) override;
    double score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) override;
    void train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::vector<std::string>& classes, size_t maxIterations = 0) override;

    void set_weights(const Eigen::MatrixXd& weights) override;
    Eigen::MatrixXd weights() override;

private:
    decision_tree_node *m_root;
};

#endif