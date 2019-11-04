#include "decision_tree_node.h"
#include <iostream>

decision_tree_node::decision_tree_node(size_t layer, const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, size_t classes, decision_tree_node *parent, bool positives)
    : m_entropy(1.0)
    , m_class(0)
    , m_classes(classes)
    , m_layer(layer)
    , m_x(x)
    , m_y(y)
    , m_split_feature(0)
    , m_split_threshold(0.0)
    , m_parent(parent)
    , m_positives(positives)
{
    m_class_counts = new size_t[classes];
    m_class_propabilities = new double[classes];
    memset(m_class_counts, 0, classes * sizeof(size_t));
    memset(m_class_propabilities, 0, classes * sizeof(double));

    m_child[0] = nullptr;
    m_child[1] = nullptr;

    init();
}

decision_tree_node::~decision_tree_node()
{
    delete m_child[0];
    delete m_child[1];
    delete [] m_class_counts;
    delete [] m_class_propabilities; 
}

bool decision_tree_node::filter(const Eigen::VectorXd& row)
{
    if (m_parent) {
        if (m_parent && !m_parent->filter(row))
            return false;
        double value = row(m_parent->m_split_feature);
        bool upper = value >= m_parent->m_split_threshold;
        return upper == m_positives;
    } else
        return true;
}

void decision_tree_node::init()
{
    size_t max = 0;
    m_item_count = 0;

    m_entropy = 0.0;
    for (auto i = 0; i < m_x.rows(); i++) {
        if (!filter(m_x.row(i)))
            continue;
        m_class_counts[static_cast<size_t>(m_y(i, 0))]++;
        m_item_count++;
    }

    for (auto i = 0; i < m_classes; i++) {
        if (!m_class_counts[i])
            continue;
        m_class_propabilities[i] = static_cast<double>(m_class_counts[i]) / static_cast<double>(m_item_count);
        m_entropy += (m_class_propabilities[i] * log(1.0 / m_class_propabilities[i]));
        if (m_class_counts[i] > max) {
            max = m_class_counts[i];
            m_class = i;
        }
    }
}

double decision_tree_node::entropy() const
{
    return m_entropy;
}

size_t decision_tree_node::decide(const Eigen::VectorXd& x)
{
    if (m_child[0] == nullptr && m_child[1] == nullptr)
        return m_class;
    
    auto value = x(m_split_feature);
    if (value >= m_split_threshold) 
        return m_child[0]->decide(x);
    else
        return m_child[1]->decide(x);
}

void decision_tree_node::split(size_t max_depth, size_t min_leaf_items)
{
    double bestSplit = std::numeric_limits<double>::max();
    auto bestSplitFeature = m_split_feature;
    auto bestSplitThreshold = m_split_threshold;

    if (entropy() == 0.0)
        return;
    if (max_depth > 0 && m_layer >= max_depth)
        return;

    for (auto feature = 0; feature < m_x.cols(); feature++)
    {
        auto feature_col = m_x.col(feature);
        auto min = feature_col.minCoeff();
        auto max = feature_col.maxCoeff();

        if (fabs(min) > 1.0 || fabs(max) > 1.0)
            // TODO: we require normalized data
            return;

        for (auto i = min + 0.1; i <= max; i += 0.1) {
            m_split_feature = feature;
            m_split_threshold = i;
            auto child0 = new decision_tree_node(m_layer + 1, m_x, m_y, m_classes, this, true);
            auto child1 = new decision_tree_node(m_layer + 1, m_x, m_y, m_classes, this, false);
            
            auto entropy = child0->entropy() + child1->entropy();
            if (child0->count() >= min_leaf_items && child1->count() >= min_leaf_items && !_isnan(entropy) && entropy < bestSplit) {
                delete m_child[0];
                delete m_child[1];
                m_child[0] = child0;
                m_child[1] = child1;
                bestSplitFeature = feature;
                bestSplitThreshold = i;
                bestSplit = entropy;
            } else {
                delete child0;
                delete child1;
            }
        }
    }
    m_split_feature = bestSplitFeature;
    m_split_threshold = bestSplitThreshold;

    if (m_layer < max_depth && m_child[0] && m_child[1]) {
        m_child[0]->split(max_depth, min_leaf_items);
        m_child[1]->split(max_depth, min_leaf_items);
    }
}

size_t decision_tree_node::count()
{
    return m_item_count;
}