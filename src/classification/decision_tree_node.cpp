#include "decision_tree_node.h"
#include <random>
#include <numeric>
#include <thread>
#include <cmath>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>

decision_tree_node::decision_tree_node(const xt::xarray<double>& x, const xt::xarray<double>& y, const xt::xarray<size_t>& node_indices, size_t layer, size_t max_depth, size_t min_leaf_items, size_t randomly_ignored_features)
    : m_entropy(0.0)
    , m_class(0)
    , m_layer(layer)
    , m_split_feature(0)
    , m_split_threshold(0.0)
{
    m_child[0] = nullptr;
    m_child[1] = nullptr;
    m_entropy = calc_entropy(x, y, node_indices);
    split(x, y, node_indices, max_depth, min_leaf_items, randomly_ignored_features);
}

decision_tree_node::~decision_tree_node()
{
    delete m_child[0];
    delete m_child[1];
}

double decision_tree_node::calc_entropy(const xt::xarray<double>& x, const xt::xarray<double>& y, const xt::xarray<size_t>& node_indices)
{
    auto node_x = xt::view(x, xt::keep(node_indices), xt::all());
    auto node_y = xt::view(y, xt::keep(node_indices), xt::all());

    auto unique_classes = xt::unique(node_y);
    auto max = 0.0;
    double entropy = 0.0;
    for (auto unique_class : unique_classes) {
        auto class_occurences = static_cast<double>(xt::eval(xt::sum(xt::where(xt::equal(node_y, unique_class), 1, 0)))(0));
        auto class_proba = class_occurences / node_indices.shape()[0];
        entropy += class_proba * log(1.0 / class_proba);
        if (class_occurences > max) {
            max = class_occurences;
            m_class = unique_class;
        }
    }
    return entropy;
}

double decision_tree_node::entropy() const
{
    return m_entropy;
}

size_t decision_tree_node::decide(const xt::xarray<double>& x) const
{
    if (!m_child[0] && !m_child[1])
        return m_class;
    
    auto value = x(m_split_feature);
    if (value >= m_split_threshold) 
        return m_child[0]->decide(x);
    else
        return m_child[1]->decide(x);
    return 0;
}

void decision_tree_node::split(const xt::xarray<double>& x, const xt::xarray<double>& y, const xt::xarray<size_t>& node_indices, size_t max_depth, size_t min_leaf_items, size_t randomly_ignored_features)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    double best_entropy = std::numeric_limits<double>::max();
    size_t best_feature = m_split_feature;
    double best_threshold = m_split_threshold;
    xt::xarray<size_t> best_indeces[2];

    if (entropy() == 0.0)
        return;
    if (max_depth > 0 && m_layer >= max_depth)
        return;
    if (randomly_ignored_features >= x.shape()[1])
        throw std::out_of_range ("randomly ignored feature out of range");

    xt::xarray<size_t> features_indices = xt::arange<size_t>(x.shape()[1]);

    if (randomly_ignored_features > 0) {
        xt::xarray<size_t> ignored_indices = xt::random::randint<size_t>({ randomly_ignored_features }, 0ULL, x.shape()[1], gen);
        ignored_indices = xt::sort(ignored_indices);
        features_indices = xt::eval(xt::view(features_indices, xt::drop(ignored_indices)));
    }

    for (auto feature : features_indices) {
        auto feature_col = xt::eval(xt::view(x, xt::all(), xt::range(feature, feature + 1)));
        feature_col.reshape({feature_col.shape()[0]});
        auto min = xt::amin(feature_col);
        auto max = xt::amax(feature_col);

        if (fabs(min(0)) > 1.0 || fabs(max(0)) > 1.0)
            throw invalid_data_exception();

        // TODO: make step configurable
        for (auto i = min(0) + 0.1; i < max(0); i += 0.1) {
            auto p1 = xt::flatten_indices(xt::argwhere(feature_col >= i));
            auto p2 = xt::flatten_indices(xt::argwhere(feature_col < i));
            double entropy = calc_entropy(x, y, p1) + calc_entropy(x, y, p2);
            if (entropy < best_entropy && p1.shape()[0] >= min_leaf_items && p2.shape()[0] >= min_leaf_items) {
                best_entropy = entropy;
                best_feature = feature;
                best_threshold = i;
                best_indeces[0] = p1;
                best_indeces[1] = p2;
            }
        }
    }
    m_split_feature   = best_feature;
    m_split_threshold = best_threshold;    

    std::thread t1([&]() { m_child[0] = new decision_tree_node(x, y, best_indeces[0], m_layer + 1, max_depth, min_leaf_items, randomly_ignored_features); });
    std::thread t2([&]() { m_child[1] = new decision_tree_node(x, y, best_indeces[1], m_layer + 1, max_depth, min_leaf_items, randomly_ignored_features); });
    t1.join();
    t2.join();
}