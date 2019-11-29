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

    xt::xarray<double> result = xt::zeros<double>(std::vector<size_t>({x.shape()[0], m_classes}));
    for (auto i = 0; i < tree_count; i++) 
        result +=  m_trees[i]->predict(x);
    return result / tree_count;
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
                    dtree->init_classes(m_classes);
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