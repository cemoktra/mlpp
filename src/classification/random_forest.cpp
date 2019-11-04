#include "random_forest.h"
#include "decision_tree.h"

random_forest::random_forest()
    : m_trees(nullptr)
{
    register_param("trees", 5);
    register_param("max_depth", 256);
    register_param("min_leaf_items", 1);
    register_param("ignored_features", 1);
}

void random_forest::set_param(const std::string& name, double new_value)
{
    auto tree_count = static_cast<size_t>(get_param("trees"));

    if (name == "trees") {
        for (auto i = 0; i < tree_count; i++)
            delete m_trees[i];
        delete [] m_trees;
        m_trees = nullptr;
    }
    parameters::set_param(name, new_value);
}

random_forest::~random_forest()
{
    auto tree_count = static_cast<size_t>(get_param("trees"));

    for (auto i = 0; i < tree_count; i++)
        delete m_trees[i];
    delete [] m_trees;
}

Eigen::MatrixXd random_forest::predict(const Eigen::MatrixXd& x)
{
    auto tree_count = static_cast<size_t>(get_param("trees"));

    Eigen::MatrixXd tmp (x.rows(), tree_count);
    Eigen::MatrixXd result (x.rows(), 1);

    for (auto i = 0; i < tree_count; i++) {
        tmp.col(i) = m_trees[i]->predict(x);
    }

    for (auto r = 0; r < x.rows(); r++) {
        auto row = tmp.row(r);
        std::vector<double> rvec (row.data(), row.data() + row.rows() * row.cols());
        size_t max = 0;
        size_t result_class = 0;
        for (auto c = 0; c < m_class_count; c++) {
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

void random_forest::init_classes(const std::vector<std::string>& classes)
{
    m_classes = classes;
    m_class_count = classes.size();
}

void random_forest::train(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y)
{
    auto tree_count = static_cast<size_t>(get_param("trees"));
    m_trees = new decision_tree*[tree_count];
    for (auto i = 0; i < tree_count; i++) {
        m_trees[i] = new decision_tree();        
        m_trees[i]->set_param("max_depth", get_param("max_depth"));
        m_trees[i]->set_param("min_leaf_items", get_param("min_leaf_items"));
        m_trees[i]->set_param("ignored_features", get_param("ignored_features"));
        m_trees[i]->init_classes(m_classes);
        m_trees[i]->train(x, y);
    }
}

void random_forest::set_weights(const Eigen::MatrixXd& weights)
{
}

Eigen::MatrixXd random_forest::weights()
{
    return Eigen::MatrixXd();
}