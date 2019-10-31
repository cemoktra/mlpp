#include "decision_tree_node.h"

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
    if (m_parent && !m_parent->filter(row))
        return false;
    if (m_parent) {
        double value = row(m_split_feature);
        bool upper = value >= m_split_threshold;
        return upper == m_positives;
    } else
        return true;
}

void decision_tree_node::init()
{
    size_t max = 0;
    size_t count = 0;

    m_entropy = 0.0;
    for (auto i = 0; i < m_y.rows(); i++) {
        if (!filter(m_y.row(i)))
            continue;
        m_class_counts[static_cast<size_t>(m_y(i, 0))]++;
        count++;
    }

    for (auto i = 0; i < m_classes; i++) {
        m_class_propabilities[i] = static_cast<double>(m_class_counts[i]) / static_cast<double>(count);
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

void decision_tree_node::split(size_t max_depth)
{
    double bestSplit = std::numeric_limits<double>::max();

    if (max_depth > 0 && m_layer >= max_depth)
        return;

    for (auto feature = 0; feature < m_x.cols(); feature++)
    {
        auto feature_col = m_x.col(feature);
        auto min = feature_col.minCoeff();
        auto max = feature_col.maxCoeff();
        
        // TODO: test splits for all features and thresholds to find best split
    }
}