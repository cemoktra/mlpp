#include "decision_tree_node.h"
#include <random>
#include <numeric>
#include <thread>
#include <cmath>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

decision_tree_node::decision_tree_node(size_t layer, const xt::xarray<double>& x, const xt::xarray<double>& y, size_t classes, decision_tree_node *parent, bool positives)
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

bool decision_tree_node::filter(const xt::xarray<double>& row)
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
    for (auto i = 0; i < m_x.shape()[0]; i++) {
        if (!filter(xt::view(m_x, xt::range(i, i + 1), xt::all())))
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

size_t decision_tree_node::decide(const xt::xarray<double>& x)
{
    if (!m_child[0] && !m_child[1])
        return m_class;
    
    auto value = x(m_split_feature);
    if (value >= m_split_threshold) 
        return m_child[0]->decide(x);
    else
        return m_child[1]->decide(x);
}

void decision_tree_node::split(size_t max_depth, size_t min_leaf_items, size_t randomly_ignored_features)
{
    std::vector<size_t> ignored_features;
    double bestSplit = std::numeric_limits<double>::max();
    auto bestSplitFeature = m_split_feature;
    auto bestSplitThreshold = m_split_threshold;

    if (entropy() == 0.0)
        return;
    if (max_depth > 0 && m_layer >= max_depth)
        return;
    if (randomly_ignored_features >= m_x.shape()[1])
        // TODO: ERROR
        return;

    if (randomly_ignored_features > 0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, m_x.shape()[1] - 1);

        while (ignored_features.size() < randomly_ignored_features) {
            auto ignored_feature = dis(gen);
            if (std::find(ignored_features.begin(), ignored_features.end(), ignored_feature) == ignored_features.end())
                ignored_features.push_back(ignored_feature);
        }
    }

    for (auto feature = 0; feature < m_x.shape()[1]; feature++)
    {
        if (std::find(ignored_features.begin(), ignored_features.end(), feature) != ignored_features.end())
            continue;

        auto feature_col = xt::view(m_x, xt::all(), xt::range(feature, feature + 1));
        auto min = xt::amin(feature_col);
        auto max = xt::amax(feature_col);

        if (fabs(min(0)) > 1.0 || fabs(max(0)) > 1.0)
            // TODO: we require normalized data
            return;

        for (auto i = min(0) + 0.1; i <= max(0); i += 0.1) {
            m_split_feature = feature;
            m_split_threshold = i;
            auto child0 = new decision_tree_node(m_layer + 1, m_x, m_y, m_classes, this, true);
            auto child1 = new decision_tree_node(m_layer + 1, m_x, m_y, m_classes, this, false);
            
            auto entropy = child0->entropy() + child1->entropy();
            if (child0->count() >= min_leaf_items && child1->count() >= min_leaf_items && !std::isnan(entropy) && entropy < bestSplit) {
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

    if (m_child[0] && m_child[1]) {
        std::thread t1([&]() { m_child[0]->split(max_depth, min_leaf_items, randomly_ignored_features); });
        std::thread t2([&]() { m_child[1]->split(max_depth, min_leaf_items, randomly_ignored_features); });
        t1.join();
        t2.join();
    }
}

size_t decision_tree_node::count()
{
    return m_item_count;
}