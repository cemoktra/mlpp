#include "random_forest.h"
#include "decision_tree.h"
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>

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

    if (name == "trees" && m_trees) {
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

xt::xarray<double> random_forest::predict(const xt::xarray<double>& x)
{
    auto tree_count = static_cast<size_t>(get_param("trees"));
    auto s = x.shape();
    s[1] = tree_count;
    xt::xarray<double> tmp (s);
    s[1] = 1;
    xt::xarray<double> result (s);

    for (auto i = 0; i < tree_count; i++) 
        xt::view(tmp, xt::all(), xt::range(i, i + 1)) = m_trees[i]->predict(x);

    for (auto r = 0; r < x.shape()[0]; r++) {
        auto row = xt::view(tmp, xt::range(r, r + 1), xt::all());
        std::vector<double> rvec (row.data(), row.data() + row.shape()[0] * row.shape()[1]);
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

double random_forest::score(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    xt::xarray<double> p = predict(x);
    xt::xarray<size_t> target_class;
    
    if (y.shape()[1] > 1)
        target_class = xt::argmax(y, {1});
    else
        target_class = y;
    target_class.reshape(p.shape());
    return xt::sum(xt::equal(p, target_class))(0) / static_cast<double>(y.shape()[0]);
}

void random_forest::init_classes(size_t number_of_classes)
{
    m_class_count = number_of_classes;
}

void random_forest::train(const xt::xarray<double>& x, const xt::xarray<double>& y)
{
    auto tree_count = static_cast<size_t>(get_param("trees"));
    m_trees = new decision_tree*[tree_count];
    for (auto i = 0; i < tree_count; i++) {
        m_trees[i] = new decision_tree();        
        m_trees[i]->set_param("max_depth", get_param("max_depth"));
        m_trees[i]->set_param("min_leaf_items", get_param("min_leaf_items"));
        m_trees[i]->set_param("ignored_features", get_param("ignored_features"));
        m_trees[i]->init_classes(m_class_count);
        m_trees[i]->train(x, y);
    }
}

void random_forest::set_weights(const xt::xarray<double>& weights)
{
}

xt::xarray<double> random_forest::weights()
{
    return xt::xarray<double>();
}