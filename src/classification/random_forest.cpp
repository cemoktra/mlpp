#include "random_forest.h"
#include "decision_tree.h"

random_forest::random_forest(size_t trees, size_t max_depth, size_t min_leaf_items, size_t ignores_features)
    : m_tree_count(trees)
{
    m_trees = new decision_tree*[trees];
    for (auto i = 0; i < m_tree_count; i++) {
        m_trees[i] = new decision_tree(max_depth, min_leaf_items);
        m_trees[i]->ignore_random_features(ignores_features);
    }
}

random_forest::~random_forest()
{
    for (auto i = 0; i < m_tree_count; i++)
        delete m_trees[i];
    delete [] m_trees;
}

Eigen::MatrixXd random_forest::predict(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd tmp (x.rows(), m_tree_count);
    Eigen::MatrixXd result (x.rows(), 1);

    for (auto i = 0; i < m_tree_count; i++) {
        tmp.col(i) = m_trees[i]->predict(x);
    }

    for (auto r = 0; r < x.rows(); r++) {
        auto row = tmp.row(r);
        std::vector<double> rvec (row.data(), row.data() + row.rows() * row.cols());
        size_t max = 0;
        size_t result_class = 0;
        for (auto c = 0; c < m_classes; c++) {
            auto count = std::count(rvec.begin(), rvec.end(), c);
            if (count > max) {
                max = count;
                result_class = c;
            }
        }
        result(r, 0) = result_class;
    }

    return result;    
}

double random_forest::score(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
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

void random_forest::train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::vector<std::string>& classes, size_t maxIterations)
{
    m_classes = classes.size();
    for (auto i = 0; i < m_tree_count; i++)
    {
        m_trees[i]->train(x, y, classes, maxIterations);
    }
}

void random_forest::set_weights(const Eigen::MatrixXd& weights)
{
}

Eigen::MatrixXd random_forest::weights()
{
    return Eigen::MatrixXd();
}