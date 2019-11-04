#include "decision_tree.h"
#include "decision_tree_node.h"

decision_tree::decision_tree(size_t max_depth, size_t min_leaf_items)
    : m_root(nullptr)
    , m_max_depth(max_depth)
    , m_min_leaf_items(min_leaf_items)
    , m_ignored_features(0)
{}

decision_tree::~decision_tree()
{
    delete m_root;
}

Eigen::MatrixXd decision_tree::predict(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd result (x.rows(), 1);

    for (auto r = 0; r < x.rows(); r++) {
        result (r, 0) = m_root->decide(x.row(r));
    }

    return result;
}

double decision_tree::score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    size_t pos = 0, neg = 0;
    auto p = predict(x);
    
    for (auto i = 0; i < p.rows(); i++) {
        auto yrow = y.row(i);
        size_t predict_class = p(i, 0);
        size_t target_class = 0;
        if (y.cols() > 1) {
            std::vector<double> yvec (yrow.data(), yrow.data() + yrow.rows() * yrow.cols());
            target_class = std::max_element(yvec.begin(), yvec.end()) - yvec.begin();
        } else
            target_class = static_cast<size_t>(yrow(0));
        
        if (predict_class == target_class)
            pos++;
        else
            neg++;        
    }
    return static_cast<double>(pos) / static_cast<double>(pos + neg);
}

void decision_tree::train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::vector<std::string>& classes, size_t maxIterations)
{
    m_root = new decision_tree_node(0, x, y, classes.size(), nullptr, true);
    m_root->split(m_max_depth, m_min_leaf_items);
}

void decision_tree::set_weights(const Eigen::MatrixXd& weights)
{
}

Eigen::MatrixXd decision_tree::weights()
{
    return Eigen::MatrixXd();
}

void decision_tree::ignore_random_features(size_t count)
{
    m_ignored_features = count;
}