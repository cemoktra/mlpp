#include "random_forest.h"
#include "decision_tree.h"
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xio.hpp>
#include <thread>
#include <mutex>

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
    xt::xarray<double> tmp (std::vector<size_t> ({x.shape()[0], tree_count}));
    xt::xarray<double> result (std::vector<size_t> ({x.shape()[0], 1}));

    for (auto i = 0; i < tree_count; i++) 
        xt::view(tmp, xt::all(), xt::range(i, i + 1)) = m_trees[i]->predict(x);

    for (auto r = 0; r < x.shape()[0]; r++) {
        auto row = xt::view(tmp, xt::range(r, r + 1), xt::all());
        size_t max = 0;
        size_t result_class = 0;
        for (auto c = 0; c < m_class_count; c++) {
            auto count = std::count(row.begin(), row.end(), c);
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
    
    if (y.shape().size() > 1 && y.shape()[1] > 1)
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

    const size_t thread_count = std::thread::hardware_concurrency();
    {
        std::vector<std::thread> threads(thread_count);
        std::mutex critical;

        for(auto t = 0; t < thread_count; t++)
        {
            threads[t] = std::thread(std::bind([&](const int start, const int end)
            {
                for (auto i = start; i < end; i++) {
                    auto dtree = new decision_tree();
                    dtree->set_param("max_depth", get_param("max_depth"));
                    dtree->set_param("min_leaf_items", get_param("min_leaf_items"));
                    dtree->set_param("ignored_features", get_param("ignored_features"));
                    dtree->init_classes(m_class_count);
                    dtree->train(x, y);
                        
                    std::lock_guard<std::mutex> lock(critical);
                    m_trees[i] = dtree;
                }
            }, t * tree_count / thread_count, (t + 1) == thread_count ? tree_count : (t + 1) * tree_count / thread_count));
        }
        std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
    }
}

void random_forest::set_weights(const xt::xarray<double>& weights)
{
}

xt::xarray<double> random_forest::weights()
{
    return xt::xarray<double>();
}