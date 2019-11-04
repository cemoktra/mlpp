#include "decision_tree.h"
#include "decision_tree_node.h"

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

void decision_tree::init_classes(const std::vector<std::string>& classes)
{
    m_classes = classes.size();
}

void decision_tree::train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    m_root = new decision_tree_node(0, x, y, m_classes, nullptr, true);
    m_root->split(static_cast<size_t>(get_param("max_depth")), static_cast<size_t>(get_param("min_leaf_items")), static_cast<size_t>(get_param("ignored_features")));
}

void decision_tree::set_weights(const Eigen::MatrixXd& weights)
{
}

Eigen::MatrixXd decision_tree::weights()
{
    return Eigen::MatrixXd();
}