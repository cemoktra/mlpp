#include "decision_tree.h"
#include "decision_tree_node.h"

decision_tree::decision_tree()
    : m_root(nullptr)
{}

decision_tree::~decision_tree()
{
    delete m_root;
}

Eigen::MatrixXd decision_tree::predict(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd result (x.rows(), 1);
    return result;
}

double decision_tree::score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    return 0.0;
}

void decision_tree::train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::vector<std::string>& classes, size_t maxIterations)
{
    m_root = new decision_tree_node(0, x, y, classes.size(), nullptr, true);
    m_root->split(100);
}

void decision_tree::set_weights(const Eigen::MatrixXd& weights)
{
}

Eigen::MatrixXd decision_tree::weights()
{
    return Eigen::MatrixXd();
}